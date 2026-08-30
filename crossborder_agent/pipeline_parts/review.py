"""Review responsibilities for the delivery pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from ..agent_loop import AgentLoopTool, AgentToolOutcome, NativeToolAgentLoop
from ..agent_workspace import BoundedAgentWorkspace
from ..agent_tools import BoundedToolRegistry
from ..api import ApiError
from ..decision_state import (
    DependencyState,
)
from ..media import (
    MediaError,
    hash_distance,
    inspect_image,
    inspect_image_quality,
)
from ..models import (
    AgentActionResult,
    AssetResult,
    CreativePlan,
    ProductFacts,
    RunState,
    TaxonomyResult,
)
from ..qa import validate_delivery
from .common import (
    PipelineError,
    SemanticRejection,
    even_sample as _even_sample,
    unique as _unique,
)


class ReviewPipelineMixin:
    def _run_agentic_delivery_loop(
        self,
        registry: BoundedToolRegistry,
        *,
        facts: ProductFacts,
        taxonomy: TaxonomyResult,
        creative_plan: CreativePlan,
        agent_plan: dict[str, Any],
        state: RunState,
        localization_payloads: dict[str, dict[str, Any]],
        localization_sources: dict[str, str],
        work_dir: Path,
    ) -> None:
        """Give the top model direct review/repair/finish authority.

        Tool handlers enforce authorization, rollback, file integrity and factual
        safety.  They do not encode a preferred repair order or a fixed number of
        repair cycles; those decisions stay in the model transcript.
        """

        if self.client is None:
            self.trace.emit("orchestrator.delivery_skipped", reason="no-model-client")
            return

        evaluations: dict[str, Any] = {}
        latest_evaluation: Any = None
        accepted = False
        attempted_actions: set[tuple[str, str, str, str]] = set()

        def open_problems() -> list[dict[str, Any]]:
            return [
                copy.deepcopy(item)
                for item in state.defect_ledger
                if item.get("status") == "open"
            ]

        def update_problem_ledger(evaluation: Any, current: str) -> None:
            rows_by_id = {
                str(item.get("defect_id") or ""): item
                for item in state.defect_ledger
                if str(item.get("defect_id") or "")
            }
            seen: set[str] = set()
            for issue in evaluation.issues:
                if not isinstance(issue, dict):
                    continue
                raw_id = str(issue.get("defect_id") or "").strip()
                if not raw_id:
                    identity = json.dumps(
                        {
                            "dimension": issue.get("dimension"),
                            "criterion": issue.get("criterion"),
                            "target": issue.get("target"),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                    raw_id = (
                        "finding-"
                        + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
                    )
                seen.add(raw_id)
                row = rows_by_id.get(raw_id)
                if row is None:
                    row = {
                        "defect_id": raw_id,
                        "status": "open",
                        "first_seen_fingerprint": current,
                        "last_seen_fingerprint": current,
                        "review_count": 1,
                        "attempts": [],
                    }
                    state.defect_ledger.append(row)
                    rows_by_id[raw_id] = row
                else:
                    row["status"] = "open"
                    row["last_seen_fingerprint"] = current
                    row["review_count"] = int(row.get("review_count") or 0) + 1
                row["finding"] = copy.deepcopy(issue)
            for defect_id, row in rows_by_id.items():
                if row.get("status") == "open" and defect_id not in seen:
                    row["status"] = "resolved"
                    row["resolved_at_fingerprint"] = current

        def problem_state(_: dict[str, Any]) -> dict[str, Any]:
            return {
                "artifact_fingerprint": fingerprint(),
                "open_problems": open_problems(),
                "attempt_history": [
                    {
                        "tool": item.tool,
                        "target": item.target,
                        "status": item.status,
                        "changed": item.changed,
                        "defect_id": item.defect_id,
                        "before_hash": item.before_hash,
                        "after_hash": item.after_hash,
                    }
                    for item in state.agent_actions
                ],
                "remaining_seconds": round(self.deadline - time.monotonic(), 1),
                "available_tools": registry.catalog(),
            }

        def fingerprint() -> str:
            return self._delivery_fingerprint(
                state=state,
                localization_payloads=localization_payloads,
                localization_sources=localization_sources,
                work_dir=work_dir,
            )

        def validate_current() -> dict[str, Any]:
            # The strategy narrative is written after control finishes. Install a
            # temporary contract-valid stub so the existing public validator can
            # inspect the otherwise complete delivery without a second validator.
            strategy_path = work_dir / "strategy_document.md"
            created_stub = not strategy_path.exists()
            if created_stub:
                strategy_path.write_text(
                    "\n".join(
                        (
                            "# Agent control preview",
                            f"商品事实: {facts.offer_id}",
                            f"平台类目: {taxonomy.category.category_id}",
                            "本地化: en-US, ko-KR, pt-BR",
                            "质检: deterministic artifact gate",
                        )
                    ),
                    encoding="utf-8",
                )
            try:
                report = validate_delivery(work_dir, facts, taxonomy)
            finally:
                if created_stub:
                    strategy_path.unlink(missing_ok=True)
            return {
                "valid": report.valid,
                "errors": report.errors[:30],
                "warnings": report.warnings[:30],
                "artifact_fingerprint": fingerprint(),
            }

        def inspect_delivery(_: dict[str, Any]) -> dict[str, Any]:
            with self._detail_candidate_pool_lock:
                candidate_state = copy.deepcopy(self._detail_candidate_pools)
            return {
                "artifacts": [
                    {
                        "name": item.name,
                        "generated": item.generated,
                        "model": item.model,
                        "fallback_reason": item.fallback_reason,
                        "description": item.description,
                    }
                    for item in state.assets
                ],
                "visual_set_review": state.visual_set_review,
                "reconciled_fact_ledger": facts.reconciled_fact_ledger,
                "taxonomy": {
                    "category_id": taxonomy.category.category_id,
                    "category": taxonomy.category.name,
                    "path": taxonomy.category.path,
                    "schema_id": taxonomy.attribute_schema_category_id,
                    "attributes": [
                        {
                            "attr_id": item.attr_id,
                            "value_id": item.value_id,
                            "name": item.name,
                            "value": item.platform_value,
                            "source_name": item.source_name,
                            "source_value": item.source_value,
                            "evidence_pointer": item.source_evidence_pointer,
                        }
                        for item in taxonomy.attributes
                    ],
                    "missing_required": taxonomy.missing_required,
                },
                "detail_candidate_state": candidate_state,
                "problem_state": problem_state({}),
                "repair_tools": registry.catalog(),
                "deterministic_validation": validate_current(),
            }

        def review_delivery(_: dict[str, Any]) -> dict[str, Any]:
            nonlocal latest_evaluation
            current = fingerprint()
            if current in evaluations:
                latest_evaluation = evaluations[current]
            else:
                evaluation = self.agent.evaluate_delivery(
                    round_index=len(state.agent_evaluations),
                    facts=facts,
                    taxonomy=taxonomy,
                    creative_plan=creative_plan,
                    agent_plan=agent_plan,
                    assets=state.assets,
                    localization_payloads=localization_payloads,
                    localization_sources=localization_sources,
                    visual_set_review=state.visual_set_review,
                    work_dir=work_dir,
                    tools=registry,
                    artifact_fingerprint=current,
                    expected_delivery_spec=state.expected_delivery_spec,
                )
                if evaluation is None:
                    return {
                        "ok": False,
                        "error": "independent review did not return sufficient valid evidence",
                        "artifact_fingerprint": current,
                    }
                evaluation.artifact_fingerprint = current
                evaluations[current] = evaluation
                state.agent_evaluations.append(evaluation)
                latest_evaluation = evaluation
                update_problem_ledger(evaluation, current)
                self.trace.emit(
                    "orchestrator.delivery_review",
                    artifact_fingerprint=current,
                    ready=evaluation.ready_for_delivery,
                    issues=evaluation.issues,
                    evaluator_models=evaluation.evaluator_models,
                )
            dependency_state = DependencyState.from_dict(state.dependency_state)
            dependency_state.record(
                "review",
                {
                    "artifact_fingerprint": current,
                    "issues": latest_evaluation.issues,
                    "models": latest_evaluation.evaluator_models,
                },
                artifacts=current,
                delivery_spec=str(state.expected_delivery_spec.get("version") or ""),
            )
            state.dependency_state = dependency_state.to_dict()
            return {
                "artifact_fingerprint": current,
                "summary": latest_evaluation.summary,
                "issues": latest_evaluation.issues,
                "ready_for_delivery": latest_evaluation.ready_for_delivery,
                "evaluator_models": latest_evaluation.evaluator_models,
                "problem_state": problem_state({}),
                "deterministic_validation": validate_current(),
            }

        def repair_artifact(arguments: dict[str, Any]) -> dict[str, Any]:
            tool = str(arguments.get("tool") or "")
            target = str(arguments.get("target") or "")
            instruction = " ".join(str(arguments.get("instruction") or "").split())
            defect_id = str(arguments.get("defect_id") or "")[:300]
            if len(instruction) < 12:
                return {"ok": False, "error": "repair instruction is too vague"}
            action_key = (
                fingerprint(),
                tool,
                target,
                " ".join(instruction.casefold().split()),
            )
            available, reason = registry.availability(tool, target)
            if not available:
                return {"ok": False, "error": reason, "tool": tool, "target": target}
            required = registry.estimated_seconds(tool) + 120
            if self.deadline - time.monotonic() <= required:
                return {
                    "ok": False,
                    "error": "insufficient time for this repair plus validation reserve",
                    "required_seconds": required,
                }
            if action_key in attempted_actions:
                return {
                    "ok": False,
                    "error": "the exact same action was already attempted on this artifact fingerprint",
                    "problem_state": problem_state({}),
                }
            attempted_actions.add(action_key)
            checkpoint = self._capture_repair_checkpoint(
                state=state,
                localization_payloads=localization_payloads,
                localization_sources=localization_sources,
                work_dir=work_dir,
            )
            target_path = work_dir / target
            before_hash = (
                self._artifact_hash(target_path)
                if target_path.is_file()
                else fingerprint()
            )
            result = registry.execute(tool, target, instruction)
            after_hash = (
                self._artifact_hash(target_path)
                if target_path.is_file()
                else fingerprint()
            )
            changed = result.status == "completed" and before_hash != after_hash
            status = (
                result.status
                if changed or result.status != "completed"
                else "no_change"
            )
            detail = (
                result.detail
                if changed or result.status != "completed"
                else "tool completed without changing the target"
            )
            if changed:
                synchronized = self._synchronize_repair_dependencies(
                    round_index=len(state.agent_evaluations),
                    changed_targets={target},
                    registry=registry,
                    facts=facts,
                    taxonomy=taxonomy,
                    creative_plan=creative_plan,
                    state=state,
                    localization_payloads=localization_payloads,
                    work_dir=work_dir,
                )
                consistent, consistency_detail = (
                    self._repair_batch_consistent(
                        {target},
                        state=state,
                        localization_payloads=localization_payloads,
                        work_dir=work_dir,
                    )
                    if synchronized
                    else (False, "dependency synchronization failed")
                )
                if not consistent:
                    self._restore_repair_checkpoint(
                        checkpoint,
                        state=state,
                        localization_payloads=localization_payloads,
                        localization_sources=localization_sources,
                        work_dir=work_dir,
                    )
                    changed = False
                    status = "rolled_back"
                    detail = consistency_detail
                    after_hash = (
                        self._artifact_hash(target_path)
                        if target_path.is_file()
                        else fingerprint()
                    )
            elif after_hash != before_hash or result.status == "completed":
                self._restore_repair_checkpoint(
                    checkpoint,
                    state=state,
                    localization_payloads=localization_payloads,
                    localization_sources=localization_sources,
                    work_dir=work_dir,
                )
                after_hash = (
                    self._artifact_hash(target_path)
                    if target_path.is_file()
                    else fingerprint()
                )
            shutil.rmtree(checkpoint["directory"], ignore_errors=True)
            state.agent_actions.append(
                AgentActionResult(
                    round_index=len(state.agent_evaluations),
                    tool=tool,
                    target=target,
                    status=status,
                    detail=detail,
                    defect_id=defect_id,
                    before_hash=before_hash,
                    after_hash=after_hash,
                    changed=changed,
                    metadata=dict(result.metadata),
                )
            )
            if defect_id:
                ledger_row = next(
                    (
                        item
                        for item in state.defect_ledger
                        if item.get("defect_id") == defect_id
                    ),
                    None,
                )
                if ledger_row is not None:
                    ledger_row.setdefault("attempts", []).append(
                        {
                            "tool": tool,
                            "target": target,
                            "status": status,
                            "changed": changed,
                            "before_hash": before_hash,
                            "after_hash": after_hash,
                        }
                    )
            if changed:
                dependency_state = DependencyState.from_dict(state.dependency_state)
                dependency_state.record(
                    "artifacts", {"artifact_fingerprint": fingerprint()}
                )
                dependency_state.invalidate(
                    "artifacts", ["review"], f"repair changed {target}"
                )
                state.dependency_state = dependency_state.to_dict()
            return {
                "tool": tool,
                "target": target,
                "status": status,
                "detail": detail,
                "changed": changed,
                "artifact_fingerprint": fingerprint(),
                "must_review_again": changed,
                "problem_state": problem_state({}),
            }

        def finish_delivery(arguments: dict[str, Any]) -> AgentToolOutcome:
            nonlocal accepted
            validation = validate_current()
            current = fingerprint()
            if not validation["valid"]:
                return AgentToolOutcome(
                    {
                        "ok": False,
                        "error": "deterministic delivery contract failed",
                        **validation,
                    }
                )
            required_sources = state.expected_delivery_spec.get(
                "required_mapping_sources", []
            )
            actual_sources = {
                (
                    "sales" if item.sales_attribute else "product",
                    item.source_name,
                    item.source_value,
                )
                for item in taxonomy.attributes
            }
            mapping_gaps = [
                item
                for item in required_sources
                if isinstance(item, dict)
                and (
                    str(item.get("scope") or ""),
                    str(item.get("source_name") or ""),
                    str(item.get("source_value") or ""),
                )
                not in actual_sources
            ]
            if mapping_gaps:
                return AgentToolOutcome(
                    {
                        "ok": False,
                        "error": "taxonomy repair dropped frozen source coverage",
                        "missing_mapping_sources": mapping_gaps,
                    }
                )
            stale_dependencies = DependencyState.from_dict(
                state.dependency_state
            ).stale_nodes()
            if stale_dependencies:
                return AgentToolOutcome(
                    {
                        "ok": False,
                        "error": "downstream projections or review are stale",
                        "stale_dependencies": stale_dependencies,
                    }
                )
            if (
                latest_evaluation is None
                or latest_evaluation.artifact_fingerprint != current
            ):
                return AgentToolOutcome(
                    {
                        "ok": False,
                        "error": "review_delivery is required for the current artifact state",
                    }
                )
            hard_issues = [
                item
                for item in latest_evaluation.issues
                if str(item.get("severity") or "").casefold() in {"blocker", "critical"}
                or (
                    str(item.get("dimension") or "") in {"A1", "A2", "A5"}
                    and str(item.get("severity") or "").casefold() == "major"
                )
            ]
            if hard_issues:
                return AgentToolOutcome(
                    {
                        "ok": False,
                        "error": "unresolved safety, integrity, or product-grounding findings",
                        "issues": hard_issues,
                    }
                )
            accepted = True
            state.accepted_artifact_fingerprint = current
            return AgentToolOutcome(
                {
                    "accepted": True,
                    "reason": str(arguments.get("reason") or "")[:1000],
                    "artifact_fingerprint": current,
                    "remaining_soft_issues": latest_evaluation.issues,
                },
                terminate=True,
            )

        catalog = registry.catalog()
        tool_names = [item["name"] for item in catalog]
        targets = sorted(
            {target for item in catalog for target in item.get("allowed_targets", [])}
        )
        empty_schema = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        runtime_dir = Path(
            tempfile.mkdtemp(prefix=".agent-runtime-", dir=work_dir)
        )
        runtime_prefix = runtime_dir.relative_to(work_dir).as_posix()
        workspace = BoundedAgentWorkspace(
            work_dir,
            staging_dir=f"{runtime_prefix}/staging",
        )

        def refresh_agent_workspace() -> None:
            workspace.host_write_json(
                f"{runtime_prefix}/workspace/index.json",
                {
                    "current_state": f"{runtime_prefix}/state/delivery.json",
                    "problem_state": f"{runtime_prefix}/state/problems.json",
                    "product_evidence": f"{runtime_prefix}/evidence/product.json",
                    "expected_delivery_spec": f"{runtime_prefix}/evidence/expected_delivery_spec.json",
                    "repair_capabilities": f"{runtime_prefix}/state/repair_tools.json",
                    "artifact_files": sorted(
                        path.name
                        for path in work_dir.iterdir()
                        if path.is_file() and not path.name.startswith(".")
                    ),
                    "notes": (
                        "Artifact files at workspace root are read-only to the agent. "
                        "Use file/ffprobe through restricted bash for local physical inspection."
                    ),
                },
            )
            workspace.host_write_json(
                f"{runtime_prefix}/evidence/product.json", facts.compact_dict()
            )
            workspace.host_write_json(
                f"{runtime_prefix}/evidence/expected_delivery_spec.json",
                state.expected_delivery_spec,
            )
            workspace.host_write_json(
                f"{runtime_prefix}/state/delivery.json", inspect_delivery({})
            )
            workspace.host_write_json(
                f"{runtime_prefix}/state/problems.json", problem_state({})
            )
            workspace.host_write_json(
                f"{runtime_prefix}/state/repair_tools.json", catalog
            )

        def workspace_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            refresh_agent_workspace()
            return workspace.execute(name, arguments)

        refresh_agent_workspace()
        loop_tools = [
            *[
                AgentLoopTool(
                    str(item["function"]["name"]),
                    str(item["function"]["description"]),
                    dict(item["function"]["parameters"]),
                    lambda arguments, name=str(item["function"]["name"]): workspace_tool(
                        name, arguments
                    ),
                )
                for item in BoundedAgentWorkspace.openai_tools()
            ],
            AgentLoopTool(
                "inspect_delivery",
                "Inspect the full current delivery snapshot. Prefer workspace read/search when only one evidence section is needed.",
                empty_schema,
                inspect_delivery,
            ),
            AgentLoopTool(
                "inspect_problem_state",
                "Inspect open evidence-backed problems, prior attempts and no-change outcomes, current fingerprint, remaining time, and tool costs.",
                empty_schema,
                problem_state,
            ),
            AgentLoopTool(
                "review_delivery",
                "Run independent evidence-based review for the current artifact fingerprint. Cached for unchanged artifacts.",
                empty_schema,
                review_delivery,
            ),
            AgentLoopTool(
                "repair_artifact",
                "Execute one targeted reversible repair. Choose the tool, target, and correction from review evidence.",
                {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string", "enum": tool_names},
                        "target": {"type": "string", "enum": targets},
                        "instruction": {
                            "type": "string",
                            "minLength": 12,
                            "maxLength": 3000,
                        },
                        "defect_id": {"type": "string", "maxLength": 300},
                    },
                    "required": ["tool", "target", "instruction"],
                    "additionalProperties": False,
                },
                repair_artifact,
            ),
            AgentLoopTool(
                "validate_delivery",
                "Run deterministic file, format, schema, localization, and data-integrity checks.",
                empty_schema,
                lambda _: validate_current(),
            ),
            AgentLoopTool(
                "finish_delivery",
                "Accept the current delivery and end the run. Call it alone. The host rejects stale review or unresolved hard safety/integrity findings.",
                {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "minLength": 5, "maxLength": 1000}
                    },
                    "required": ["reason"],
                    "additionalProperties": False,
                },
                finish_delivery,
                terminal=True,
            ),
        ]
        system_prompt = self.agent.orchestrator_system_prompt or (
            "You are the top-level delivery orchestrator. Use evidence and bounded tools; never invent product facts."
        )
        loop = NativeToolAgentLoop(
            self.client,
            system_prompt=system_prompt,
            messages=self.agent.orchestrator_messages,
            trace=self.trace,
        )
        # A fixed ten-turn ceiling previously terminated healthy runs while a
        # large share of the wall-clock budget was still available.  Turns are
        # only a runaway-call guard here; the real production constraint is the
        # deadline and the per-tool reserve enforced by handlers.
        available_loop_seconds = max(
            0.0, self.deadline - time.monotonic() - 120.0
        )
        max_orchestrator_turns = max(
            10, min(64, int(available_loop_seconds // 20.0) or 10)
        )
        try:
            try:
                result = loop.run(
                    f"Initial production is complete. Start from {runtime_prefix}/workspace/index.json. Inspect and review it, choose any valuable targeted repairs, "
                    "review changed states, and call finish_delivery when the evidence is sufficient. Avoid cosmetic "
                    "churn; prioritize product identity, factual grounding, buyer-facing compliance, and artifact integrity. "
                    "Use the workspace or inspect_problem_state whenever attempt history, no-change results, remaining time, or tool costs "
                    "would help you decide the next action. You own the repair strategy; no fixed repair order is implied.",
                    loop_tools,
                    max_turns=max_orchestrator_turns,
                    deadline=self.deadline,
                    reserve_seconds=120,
                )
            finally:
                shutil.rmtree(runtime_dir, ignore_errors=True)
        except ApiError as exc:
            self.warnings.append(f"顶层交付编排器提前停止: {exc}")
            self.trace.emit("orchestrator.delivery_failed", error=str(exc))
            # Some OpenAI-compatible endpoints expose chat but not native tool
            # calls. Preserve the protocol smoke path with a clearly degraded,
            # deterministic contract gate; never pretend an LLM review occurred.
            if "未调用任何可用工具" not in str(exc):
                raise PipelineError(f"顶层交付编排器未能接受当前交付: {exc}") from exc
            report = validate_delivery(work_dir, facts, taxonomy)
            if not report.valid:
                raise PipelineError(
                    "顶层工具协议不可用且确定性交付校验失败: "
                    + "; ".join(report.errors)
                ) from exc
            current = self._delivery_fingerprint(
                state=state,
                localization_payloads=localization_payloads,
                localization_sources=localization_sources,
                work_dir=work_dir,
            )
            state.accepted_artifact_fingerprint = current
            warning = "模型端不支持原生工具调用：本次仅通过确定性交付契约门禁"
            if warning not in self.warnings:
                self.warnings.append(warning)
            self.trace.emit(
                "orchestrator.delivery_degraded_acceptance",
                artifact_fingerprint=current,
                reason="native-tool-protocol-unavailable",
            )
            return
        self.agent.orchestrator_messages = result.messages

        # The model owns review and repair strategy, but acceptance itself is a
        # deterministic host contract.  If a turn/deadline boundary lands after
        # the final review instead of after an explicit finish call, re-run only
        # the missing current-state review when time permits, then apply the same
        # finish gate the model would have called.  This prevents an execution
        # bookkeeping boundary from discarding an otherwise accepted delivery.
        final_gate: AgentToolOutcome | None = None
        if not accepted:
            current = fingerprint()
            review_is_current = (
                latest_evaluation is not None
                and latest_evaluation.artifact_fingerprint == current
            )
            if (
                not review_is_current
                and self.deadline - time.monotonic() > 180.0
            ):
                try:
                    review_delivery({})
                except (ApiError, ValueError, TypeError) as exc:
                    self.trace.emit(
                        "orchestrator.delivery_final_review_failed",
                        error=str(exc),
                    )
            final_gate = finish_delivery(
                {
                    "reason": (
                        "Host applied the documented acceptance gate after the "
                        f"agent loop stopped with {result.stop_reason}."
                    )
                }
            )
            self.trace.emit(
                "orchestrator.delivery_host_finalization",
                stop_reason=result.stop_reason,
                accepted=accepted,
                observation=final_gate.observation,
            )
        self.trace.emit(
            "orchestrator.delivery_complete",
            stop_reason=result.stop_reason,
            turns=result.turns,
            accepted=accepted,
        )
        if not accepted:
            gate_observation = final_gate.observation if final_gate else {}
            gate_error = str(
                gate_observation.get("error")
                or "current delivery did not satisfy the acceptance contract"
            )
            raise PipelineError(
                f"交付未通过宿主验收门禁（{result.stop_reason}）: {gate_error}"
            )

    def _copy_revision_is_safe(
        self,
        language: str,
        facts: ProductFacts,
        incumbent: dict[str, Any],
        candidate: dict[str, Any],
    ) -> bool:
        if self.client is None:
            return False
        system = (
            "You are a conservative cross-border listing A/B judge. Return JSON only. "
            "Compare factual completeness, source support, shopper-facing fluency, title quality, "
            "conversion usefulness and source-script contamination. Never reward invented claims."
        )
        prompt = f"""
Language: {language}
Candidate 0 is the current accepted copy. Candidate 1 is a proposed repair.
Return selected_index plus exactly two candidates containing index, score (0-100),
facts_supported, complete, native_and_natural, has_source_script_contamination, and reason.
Candidate 1 is an evaluator-requested repair. Mark its facts and language properties conservatively.
A confirmed factual correction must not be rejected merely because its style score is close to candidate 0.
The reconciled fact ledger inside Verified facts is authoritative for appearance conflicts.

Verified facts:
{json.dumps(facts.compact_dict(), ensure_ascii=False)}

Candidate 0:
{json.dumps(incumbent, ensure_ascii=False)}

Candidate 1:
{json.dumps(candidate, ensure_ascii=False)}
""".strip()
        try:
            review = self.client.chat_json(
                system, prompt, model=self.client.config.review_model
            )
        except ApiError as exc:
            self.logger.warning("%s 文案 A/B 评审不可用，保留旧文案: %s", language, exc)
            return False
        rows = review.get("candidates")
        if not isinstance(rows, list):
            return False
        by_index = {
            row.get("index"): row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("index"), int)
        }
        old = by_index.get(0)
        new = by_index.get(1)
        if not isinstance(old, dict) or not isinstance(new, dict):
            return False
        return bool(
            new.get("facts_supported") is True
            and new.get("complete") is not False
            and new.get("native_and_natural") is not False
            and new.get("has_source_script_contamination") is not True
        )

    @staticmethod
    def _video_revision_improves(
        review: dict[str, Any], *, has_incumbent: bool
    ) -> bool:
        if has_incumbent:
            rows = review.get("candidates")
            if not isinstance(rows, list):
                return False
            by_index = {
                row.get("index"): row
                for row in rows
                if isinstance(row, dict) and isinstance(row.get("index"), int)
            }
            old = by_index.get(0)
            new = by_index.get(1)
            if not isinstance(old, dict) or not isinstance(new, dict):
                return False
            old_score = old.get("score")
            new_score = new.get("score")
            if (
                review.get("selected_index") != 1
                or not isinstance(old_score, (int, float))
                or not isinstance(new_score, (int, float))
                or float(new_score) < float(old_score) + 6.0
            ):
                return False
            candidate = new
        else:
            candidate = review
        return bool(
            candidate.get("usable") is True
            and candidate.get("identity_consistent") is not False
            and candidate.get("construction_consistent") is not False
            and candidate.get("color_and_pattern_consistent") is not False
            and candidate.get("motion_stable") is not False
            and candidate.get("unwanted_text") is not True
            and candidate.get("prohibited_visual") is not True
            and candidate.get("major_artifacts") is not True
        )

    @staticmethod
    def _candidate_soft_score(item: dict[str, Any], *, selected_index: Any) -> float:
        raw_score = item.get("score")
        score = float(raw_score) if isinstance(raw_score, (int, float)) else 70.0
        if item.get("index") == selected_index:
            score += 2.0
        if item.get("usable") is False:
            score -= 8.0
        # A close-up may omit the full category outline while still delivering
        # its assigned local feature. Identity and slot mismatch remain hard.
        if item.get("critical_structure_unambiguous") is False:
            score -= 6.0
        coverage = str(item.get("product_coverage") or "").casefold()
        if coverage == "low":
            score -= 10.0
        elif coverage == "medium":
            score -= 3.0
        return max(0.0, min(100.0, score))

    def _choose_monotonic_candidate(
        self,
        label: str,
        candidate_urls: list[str],
        ranked: list[tuple[float, int, dict[str, Any]]],
        *,
        incumbent_index: int | None,
        minimum_improvement: float,
        hard_reasons: list[str],
    ) -> str:
        incumbent_valid = isinstance(
            incumbent_index, int
        ) and 0 <= incumbent_index < len(candidate_urls)
        if not ranked:
            if incumbent_valid:
                self.logger.warning("%s 新候选均有语义硬伤，保留当前资产", label)
                return candidate_urls[incumbent_index]
            feedback = "; ".join(hard_reasons[:4]) or "评审未返回可比较候选"
            raise SemanticRejection(
                f"{label} 候选均未通过语义质检（存在硬伤）", feedback=feedback
            )

        best_score, best_index, _ = max(ranked, key=lambda row: (row[0], -row[1]))
        if incumbent_valid:
            incumbent_row = next(
                (row for row in ranked if row[1] == incumbent_index), None
            )
            # If the judge omitted or hard-rejected the incumbent, the old artifact
            # is still safer than an unproven replacement unless a new candidate is
            # clearly acceptable at a high absolute score.
            incumbent_score = incumbent_row[0] if incumbent_row else 80.0
            if best_index == incumbent_index:
                return candidate_urls[incumbent_index]
            required = incumbent_score + max(0.0, minimum_improvement)
            if best_score < required:
                self.logger.info(
                    "%s 修复候选提升不足: old=%.1f new=%.1f required=%.1f，保留旧资产",
                    label,
                    incumbent_score,
                    best_score,
                    required,
                )
                return candidate_urls[incumbent_index]
        self.logger.info(
            "%s 候选选优: 选择 %d/%d，软评分 %.1f",
            label,
            best_index + 1,
            len(candidate_urls),
            best_score,
        )
        return candidate_urls[best_index]

    def _select_main_candidate(
        self,
        facts: ProductFacts,
        source_urls: list[str],
        candidate_urls: list[str],
        *,
        incumbent_index: int | None = None,
        minimum_improvement: float = 0.0,
    ) -> str:
        if not candidate_urls:
            raise ApiError("主图模型未返回候选")
        # A single generated candidate still needs semantic acceptance. Skipping
        # review here lets structure drift pass merely because no alternative
        # candidate was requested for this storyboard slot.
        if self.client is None:
            return candidate_urls[0]
        try:
            review = self.client.select_best_generated_image(
                json.dumps(facts.compact_dict(), ensure_ascii=False),
                source_urls,
                candidate_urls,
            )
        except ApiError as exc:
            keep = (
                incumbent_index
                if isinstance(incumbent_index, int)
                and 0 <= incumbent_index < len(candidate_urls)
                else 0
            )
            self.logger.warning(
                "主图语义评审不可用，保留确定性安全候选 %d，避免误回退: %s",
                keep,
                exc,
            )
            self.warnings.append(f"主图候选评审不可用，保留候选: {exc}")
            return candidate_urls[keep]

        candidates = review.get("candidates")
        selected = review.get("selected_index")
        self.trace.emit(
            "image.hero_review",
            source_urls=source_urls,
            candidate_urls=candidate_urls,
            selected_index=selected,
            review=review,
            incumbent_index=incumbent_index,
            minimum_improvement=minimum_improvement,
        )
        if not isinstance(candidates, list):
            keep = incumbent_index if incumbent_index is not None else 0
            self.warnings.append("主图语义评审结构不完整，采用确定性候选")
            return candidate_urls[keep]
        wearer_supported = self._hero_wearer_supported(facts, source_urls)
        ranked: list[tuple[float, int, dict[str, Any]]] = []
        hard_reasons: list[str] = []
        for item in candidates:
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                continue
            index = item["index"]
            if not 0 <= index < len(candidate_urls):
                continue
            hard_fields = {
                "identity_consistent": False,
                "construction_consistent": False,
                "correct_color": False,
                "single_product": False,
                "product_complete": False,
                "unwanted_text": True,
                "unwanted_brand_or_logo": True,
                "major_artifacts": True,
            }
            failed = [
                key for key, value in hard_fields.items() if item.get(key) is value
            ]
            if item.get("has_person") is True and not wearer_supported:
                failed.append("unsupported_wearer")
            if item.get("has_person") is True and item.get("anatomy_natural") is False:
                failed.append("anatomy_natural")
            if failed:
                hard_reasons.append(
                    f"candidate {index}: {','.join(failed)}; {item.get('reason', '')}"
                )
                continue
            score = self._candidate_soft_score(item, selected_index=selected)
            if item.get("clean_neutral_background") is False:
                score -= 8
            if item.get("has_unrelated_props") is True:
                score -= 5
            if item.get("has_person") is True:
                score -= 2
            ranked.append((score, index, item))
        return self._choose_monotonic_candidate(
            "主图",
            candidate_urls,
            ranked,
            incumbent_index=incumbent_index,
            minimum_improvement=minimum_improvement,
            hard_reasons=hard_reasons,
        )

    def _select_detail_candidate(
        self,
        index: int,
        facts: ProductFacts,
        source_urls: list[str],
        candidate_urls: list[str],
        purpose: str,
        *,
        incumbent_index: int | None = None,
        minimum_improvement: float = 0.0,
    ) -> str:
        if not candidate_urls:
            raise ApiError(f"详情图 {index} 模型未返回候选")
        # A single generated detail candidate still needs semantic acceptance;
        # otherwise structure drift can pass just because there is no alternative.
        if self.client is None:
            return candidate_urls[0]
        try:
            review = self.client.select_best_detail_image(
                json.dumps(facts.compact_dict(), ensure_ascii=False),
                source_urls,
                candidate_urls,
                asset_name=f"detail_image_{index}.jpeg",
                purpose=purpose,
            )
        except ApiError as exc:
            self._detail_candidate_reviews[index] = {
                "status": "unavailable",
                "error": str(exc)[:1000],
                "candidates": [],
            }
            keep = (
                incumbent_index
                if isinstance(incumbent_index, int)
                and 0 <= incumbent_index < len(candidate_urls)
                else 0
            )
            self.logger.warning(
                "详情图 %d 语义评审不可用，保留确定性安全候选 %d: %s",
                index,
                keep,
                exc,
            )
            self.warnings.append(f"详情图 {index} 评审不可用，保留候选: {exc}")
            return candidate_urls[keep]
        self._detail_candidate_reviews[index] = copy.deepcopy(review)
        candidates = review.get("candidates")
        selected = review.get("selected_index")
        self.trace.emit(
            "image.detail_review",
            asset=f"detail_image_{index}.jpeg",
            purpose=purpose,
            source_urls=source_urls,
            candidate_urls=candidate_urls,
            selected_index=selected,
            review=review,
            incumbent_index=incumbent_index,
            minimum_improvement=minimum_improvement,
        )
        if not isinstance(candidates, list):
            keep = incumbent_index if incumbent_index is not None else 0
            self.warnings.append(f"详情图 {index} 评审结构不完整，采用确定性候选")
            return candidate_urls[keep]
        ranked: list[tuple[float, int, dict[str, Any]]] = []
        hard_reasons: list[str] = []
        for item in candidates:
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                continue
            candidate_index = item["index"]
            if not 0 <= candidate_index < len(candidate_urls):
                continue
            hard_fields = {
                "identity_consistent": False,
                "construction_consistent": False,
                "color_consistent": False,
                "pattern_consistent": False,
                "unwanted_text": True,
                "unwanted_brand_or_logo": True,
                "prohibited_visual": True,
                "major_artifacts": True,
                "unexpected_collage": True,
                "single_composition": False,
            }
            purpose_text = purpose.casefold()
            if any(
                token in purpose_text
                for token in ("wearer", "person", "adult", "man", "woman", "body")
            ):
                hard_fields["anatomy_natural"] = False
            failed = [
                key for key, value in hard_fields.items() if item.get(key) is value
            ]
            if failed:
                hard_reasons.append(
                    f"candidate {candidate_index}: {','.join(failed)}; {item.get('reason', '')}"
                )
                continue
            if item.get("slot_match") is False:
                hard_reasons.append(
                    f"candidate {candidate_index}: slot_match; {item.get('reason', '')}"
                )
                continue
            score = self._candidate_soft_score(item, selected_index=selected)
            ranked.append((score, candidate_index, item))
        return self._choose_monotonic_candidate(
            f"详情图 {index}",
            candidate_urls,
            ranked,
            incumbent_index=incumbent_index,
            minimum_improvement=minimum_improvement,
            hard_reasons=hard_reasons,
        )

    def _record_visual_delivery_quality(self, assets: list[AssetResult]) -> None:
        """Expose rubric-level fallback risks that physical file QA cannot see."""

        image_assets = [asset for asset in assets if asset.name.endswith(".jpeg")]
        hashes: list[tuple[str, int]] = []
        for asset in image_assets:
            try:
                quality = inspect_image_quality(Path(asset.path))
            except MediaError:
                quality = None
            if quality is not None:
                hashes.append((asset.name, quality.difference_hash))
        distinct_names: set[str] = set()
        distinct_hashes: list[int] = []
        for name, image_hash in hashes:
            if all(hash_distance(image_hash, seen) > 10 for seen in distinct_hashes):
                distinct_names.add(name)
                distinct_hashes.append(image_hash)
        for left_index, (left_name, left_hash) in enumerate(hashes):
            for right_name, right_hash in hashes[left_index + 1 :]:
                if hash_distance(left_hash, right_hash) <= 10:
                    warning = f"最终商品图近重复: {left_name}, {right_name}"
                    if warning not in self.warnings:
                        self.warnings.append(warning)

        usable = 0
        risky_names: list[str] = []
        for asset in image_assets:
            if hashes and asset.name not in distinct_names:
                continue
            if asset.generated or asset.model == "deterministic-size-chart":
                usable += 1
                continue
            observation = self._source_image_observations.get(asset.source_url, {})
            safe_key = (
                "safe_for_main_image"
                if asset.name == "main_image.jpeg"
                else "safe_for_listing_fallback"
            )
            if observation.get(safe_key) is True:
                usable += 1
            elif observation:
                risky_names.append(asset.name)
            else:
                # Explicit offline mode cannot run a semantic listing-readiness
                # review. Count only physically valid, perceptually distinct
                # fallbacks and label the estimate accordingly below.
                usable += 1

        if risky_names:
            warning = "最终视觉兜底未达到直接发布门禁: " + ", ".join(risky_names)
            if warning not in self.warnings:
                self.warnings.append(warning)
        if image_assets:
            usable_rate = usable / len(image_assets)
            estimate_basis = (
                "视觉门禁与感知差异"
                if self._source_image_observations
                else "物理规格与感知差异（未执行模型语义门禁）"
            )
            if usable_rate < 0.8:
                warning = f"按{estimate_basis}估算的出图可用率为 {usable_rate:.0%}，低于 A6 的 80% 阈值"
                if warning not in self.warnings:
                    self.warnings.append(warning)
            elif not self._source_image_observations:
                warning = (
                    f"离线图片物理规格与感知差异检查通过 {usable}/{len(image_assets)}；"
                    "语义可用率未评估，正式交付仍需确认背景、人物道具、分镜任务与商品身份"
                )
                if warning not in self.warnings:
                    self.warnings.append(warning)

        video = next(
            (asset for asset in assets if asset.name == "product_video.mp4"), None
        )
        if video and not video.generated and risky_names:
            warning = "回退视频继承了未通过直接发布门禁的静态图片"
            if warning not in self.warnings:
                self.warnings.append(warning)

    def _review_visual_set(
        self,
        facts: ProductFacts,
        assets: list[AssetResult],
    ) -> dict[str, Any]:
        """Judge hero + five details as one set, without mutating accepted files."""

        if self.client is None:
            self.trace.emit(
                "image.set_review_skipped", reason="offline-or-no-review-client"
            )
            return {}
        remaining = self.deadline - time.monotonic()
        if remaining <= 3 * 60:
            self.warnings.append("剩余时间不足，跳过六图集合语义评审")
            self.trace.emit(
                "image.set_review_skipped",
                reason="insufficient-stage-budget",
                remaining_seconds=round(remaining, 1),
            )
            return {}
        ordered_names = ["main_image.jpeg"] + [
            f"detail_image_{index}.jpeg" for index in range(1, 6)
        ]
        by_name = {asset.name: asset for asset in assets}
        if any(name not in by_name for name in ordered_names):
            self.warnings.append("六图集合不完整，无法执行集合级语义评审")
            self.trace.emit("image.set_review_skipped", reason="incomplete-image-set")
            return {}
        reviewable_names = [
            name
            for name in ordered_names
            if by_name[name].generated and by_name[name].source_url
        ]
        review_inputs = [by_name[name].source_url for name in reviewable_names]

        if self.fast_mode:
            source_references = _unique(
                facts.product_image_urls[:2] + _even_sample(facts.sku_image_urls, 1)
            )
        else:
            source_references = _unique(
                facts.product_image_urls[:3]
                + _even_sample(facts.sku_image_urls, 1)
                + _even_sample(facts.description_image_urls, 1)
            )
        expected_assets = [
            {
                "name": name,
                "purpose": by_name[name].description or name,
                "representation": "final model output before lossless delivery normalization",
            }
            for name in reviewable_names
        ]
        if review_inputs:
            try:
                review = self.client.review_generated_images(
                    json.dumps(facts.compact_dict(), ensure_ascii=False),
                    source_references,
                    review_inputs,
                    expected_assets,
                )
            except ApiError as exc:
                # A judge outage is not evidence that an already accepted image is bad.
                self.logger.warning("六图集合语义评审不可用，保留当前素材: %s", exc)
                self.warnings.append(f"六图集合语义评审未完成: {exc}")
                self.trace.emit(
                    "image.set_review_failed",
                    category=exc.category,
                    retryable=exc.retryable,
                    status_code=exc.status_code,
                    error=str(exc),
                )
                return {}
        else:
            review = {
                "assets": [],
                "set_usable": True,
                "coherent": True,
                "near_duplicate_pairs": [],
                "missing_roles": [],
                "repair_targets": [],
                "summary": "All final images are local deterministic artifacts; semantic proxy review was intentionally skipped.",
            }

        rows = review.get("assets")
        if not isinstance(rows, list) or len(rows) != len(reviewable_names):
            self.warnings.append("六图集合评审返回结构异常，忽略该评审而不判定素材失败")
            self.trace.emit("image.set_review_invalid", review=review)
            return {}
        remote_by_name: dict[str, dict[str, Any]] = {}
        for remote_index, name in enumerate(reviewable_names):
            item = next(
                (
                    row
                    for row in rows
                    if isinstance(row, dict) and row.get("index") == remote_index
                ),
                {},
            )
            remote_by_name[name] = item
        normalized_rows: list[dict[str, Any]] = []
        for index, name in enumerate(ordered_names):
            if name in remote_by_name:
                item = remote_by_name[name]
                normalized_rows.append({"name": name, **item, "index": index})
            else:
                normalized_rows.append(
                    self._local_visual_review_row(by_name[name], index=index)
                )
        review["assets"] = normalized_rows
        review["usable_count"] = sum(
            item.get("usable") is True for item in normalized_rows
        )
        review["distinct_commercial_roles"] = len(
            {
                str(item.get("actual_role") or "other")
                for item in normalized_rows
                if item.get("usable") is True
            }
        )
        review["set_usable"] = bool(
            review.get("set_usable") is not False
            and review["usable_count"] >= max(1, len(ordered_names) - 1)
        )
        review["reviewed_names"] = ordered_names
        review["review_model"] = getattr(
            self.client,
            "visual_review_model",
            self.client.config.review_model,
        )

        duplicate_pairs = review.get("near_duplicate_pairs")
        if isinstance(duplicate_pairs, list) and len(reviewable_names) != len(
            ordered_names
        ):
            remote_to_ordered = {
                remote_index: ordered_names.index(name)
                for remote_index, name in enumerate(reviewable_names)
            }
            duplicate_pairs = [
                [remote_to_ordered[pair[0]], remote_to_ordered[pair[1]]]
                for pair in duplicate_pairs
                if isinstance(pair, list)
                and len(pair) == 2
                and pair[0] in remote_to_ordered
                and pair[1] in remote_to_ordered
            ]
            review["near_duplicate_pairs"] = duplicate_pairs
            repair_targets = review.get("repair_targets")
            if isinstance(repair_targets, list):
                review["repair_targets"] = [
                    remote_to_ordered[target]
                    for target in repair_targets
                    if isinstance(target, int) and target in remote_to_ordered
                ]
        if isinstance(duplicate_pairs, list) and duplicate_pairs:
            readable_pairs: list[str] = []
            for pair in duplicate_pairs[:6]:
                if (
                    isinstance(pair, list)
                    and len(pair) == 2
                    and all(isinstance(item, int) for item in pair)
                    and all(0 <= item < len(ordered_names) for item in pair)
                ):
                    readable_pairs.append(
                        f"{ordered_names[pair[0]]}/{ordered_names[pair[1]]}"
                    )
            if readable_pairs:
                self.warnings.append(
                    "六图集合存在语义重复: " + ", ".join(readable_pairs)
                )
        missing_roles = review.get("missing_roles")
        if isinstance(missing_roles, list) and missing_roles:
            self.warnings.append(
                "六图集合商业任务覆盖不足: "
                + ", ".join(str(item)[:80] for item in missing_roles[:5])
            )
        self.trace.emit("image.set_review", review=review)
        return review

    def _local_visual_review_row(
        self, asset: AssetResult, *, index: int
    ) -> dict[str, Any]:
        """Describe a local final image without substituting its provenance pixels."""

        observation = self._source_image_observations.get(asset.source_url, {})
        try:
            info = inspect_image(Path(asset.path))
            quality = inspect_image_quality(Path(asset.path))
            physically_usable = bool(
                info.width > 260
                and info.height > 260
                and info.size_bytes <= 5 * 1024 * 1024
                and (
                    quality is None
                    or (quality.entropy >= 0.8 and quality.luminance_stddev >= 2)
                )
            )
        except (MediaError, OSError):
            physically_usable = False

        if asset.model == "deterministic-size-chart":
            actual_role = "size_chart"
            observed_features = [
                "locally rendered English size guide",
                "seller-provided size, bust, and garment-length measurements",
                "no Chinese text is rendered by the deterministic template",
            ]
            media_descriptions = {
                "en": "Size guide with seller-provided garment measurements.",
                "ko": "판매자가 제공한 의류 실측 사이즈 안내표입니다.",
                "pt": "Guia de tamanhos com as medidas da peça informadas pelo vendedor.",
            }
            source_safe = True
            unwanted_text = False
        else:
            description = asset.description.casefold()
            actual_role = (
                "construction_detail"
                if any(
                    token in description
                    for token in ("crop", "close-up", "upper", "lower")
                )
                else "alternate_view"
            )
            observed_features = [
                "normalized or cropped seller-source product image",
                asset.description or "source-backed final image",
            ]
            media_descriptions = {
                "en": "Seller-source product detail normalized for the listing.",
                "ko": "판매자 원본을 상품 상세 이미지 규격에 맞게 정리한 이미지입니다.",
                "pt": "Detalhe do produto da fonte do vendedor, normalizado para o anúncio.",
            }
            source_safe = bool(
                not observation or observation.get("safe_for_listing_fallback") is True
            )
            unwanted_text = bool(observation.get("has_overlay_text") is True)

        usable = bool(physically_usable and source_safe and not unwanted_text)
        return {
            "name": asset.name,
            "index": index,
            "usable": usable,
            "identity_consistent": source_safe,
            "construction_consistent": source_safe,
            "color_consistent": source_safe,
            "pattern_consistent": source_safe,
            "slot_match": True,
            "unwanted_text": unwanted_text,
            "prohibited_visual": False,
            "major_artifacts": not physically_usable,
            "unexpected_collage": False,
            "product_coverage": "high",
            "actual_role": actual_role,
            "description_confidence": "medium",
            "observed_features": observed_features,
            "media_descriptions": media_descriptions,
            "reason": (
                "Final local artifact was physically inspected; provenance pixels were not substituted for the final image."
            ),
            "evidence_mode": "local-final-inspection",
        }
