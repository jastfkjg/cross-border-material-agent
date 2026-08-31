# Cross-border Material Agent

Python 3.12 agent for the Qwen AI Arena cross-border e-commerce localization task.

## Runtime

The evaluator invokes:

```bash
python agent.py --prompt "...input path...output path..."
```

The agent discovers the product, category and attribute JSON files by schema; builds an evidence-backed fact ledger; resolves the AliExpress leaf category and enumerated attributes; creates English, Korean and Brazilian Portuguese descriptions; produces one main image, five deliberately distinct detail images and one video; then evaluates and validates the complete delivery before exiting.

Operational output is English: `strategy_document.md`, CLI diagnostics, validation messages, warnings, and `agent.log` use English wording. The Korean and Brazilian Portuguese product-description files remain in their required target languages. Source-language values are retained only where the machine appendix must preserve grounded listing data. The hero image is normalized to at least 800×800 pixels and strictly below 1 MB; detail images remain within the task's 5 MB limit.

The runtime is organized around a versioned decision state, not a growing list of product rules. It first collects structured, SKU, taxonomy-snapshot and per-image visual evidence. Independent model reconcilers turn conflicts into a canonical product state before taxonomy is selected. Taxonomy, publishable claims, localization and visual plans are projections of that state; a frozen expected-delivery specification records the accepted claim, mapping, locale, file and visual-evidence coverage before production begins. The top-level native tool-calling orchestrator then chooses the storyboard, exact safe source-image indexes for each job, candidate breadth, locale priorities and production order. After production it can request independent review, invoke reversible repairs, reopen evidence or taxonomy decisions, and decide when to finish. Repairs update the upstream version, rebuild affected projections and invalidate downstream review. Semantic findings are advisory repair evidence rather than a score gate: model disagreement, an unavailable reviewer, a stale review, or a remaining quality defect cannot convert a complete artifact snapshot into a process failure. Host code validates protocol, provenance and file integrity, then commits the best available result; a final availability transaction fills missing slots from surviving local or seller-source material after unexpected failures.

`crossborder_agent/pipeline.py` is the stable orchestration facade. Its implementation is split under `crossborder_agent/pipeline_parts/`: `evidence` owns source inspection and canonical fact repair, `taxonomy` owns category/schema decisions, `planning` owns the bounded repair surface, `production` owns copy and media construction, `review` owns candidate and delivery evaluation, and `transaction` owns fingerprints, rollback, dependency synchronization and final commit. Shared safety constants and exceptions live in `pipeline_parts/common.py`. The split preserves the public `Pipeline` entry point and keeps model-facing behavior independent from module layout.

In full mode, taxonomy resolution uses two bounded ReAct transactions rather than a pre-ranked candidate classifier: the first commits a leaf category and the second receives that accepted result in a fresh context to resolve its attribute schema and mappings. An attribute failure can therefore fall back only the schema/mapping projection while preserving the model-selected category. Each task has its own independent 50-turn budget; neither task consumes or inherits the other's turns. Every turn exposes that task's remaining budget, and the last two turns of each task expose only its validated finish tool so one rejected finish can still be corrected. An identical deterministically rejected finish stops the phase on its second submission instead of consuming the rest of the budget.

Like a constrained Pi-style workspace, the model can compose `list`, bounded `read`, regex `search`, restricted `bash`, and `write_staging` over normalized `evidence/product.json` plus `taxonomy/categories.jsonl`, `schemas.jsonl`, `attributes.jsonl`, and `values.jsonl`. `bash` runs one argv vector without a shell and is limited to `rg`, `jq`, `find`, `file`, and `ffprobe`; paths stay inside the workspace, subprocesses receive no secret environment variables, and writes stay below `staging/`. The same primitives are available to the top-level planner and final delivery orchestrator, while media generation, reversible repair, deterministic validation, and final commit remain host-authorized boundary tools.

Every taxonomy and source evidence row carries a stable `ref`. Attribute submissions use `source_ref` instead of copying source kind/name/value strings, and the host resolves provenance after validation. A rejected `finish_attributes` response contains accepted rows, rejected row indexes, machine-readable reason codes, expected taxonomy refs, compatible source refs, and a concrete correction contract so the model can repair only the failed delta. Before finishing, every required attribute and every sales dimension with a spelling-level SKU/schema evidence match must be mapped or recorded as unresolved with an evidence-based reason. The older lexical scorer remains only as an offline or API-failure fallback and is not included as an online model hint.

Independent evaluators and their disagreement adjudicator receive one shared, versioned evidence bundle containing the complete verified facts, category ID/name/path, attribute schema and provenance, frozen delivery target, localized copy evidence, visual review, and artifact manifest. The host defines this bounded evidence universe once; models decide which evidence is relevant to a finding. Evaluators return evidence-backed defects, not acceptance scores. One valid report is sufficient to guide repair, while additional reports and adjudication improve confidence without becoming availability dependencies.

Localized copy uses a writer pass followed by a factual-auditor pass and locale guards for en-US, ko-KR and pt-BR. The models author only the buyer-facing title, overview, highlights and fit note; code independently builds the machine appendix from verified taxonomy, attribute, SKU and media contracts, so prose revisions cannot mutate parseable fields. Every description leads with two substantive shopper-facing paragraphs and natural feature highlights, followed by compact localized listing tables. Category IDs, platform attribute/value IDs, SKU IDs and Spec IDs remain parseable; localized source labels distinguish seller facts from platform mappings, while raw Chinese values and JSON pointers stay in the private fact ledger. Legible source-image tables use a domain-neutral cell protocol: the model decides their meaning, selected rows and columns, localized labels, notes and detail-image placement, while code checks exact source-cell references and renders only a complete grounded presentation. Unsupported numeric fields are repaired independently so one bad number no longer discards an otherwise valid localized draft.

Every inspected source image receives a role and risk record covering intrinsic product print versus marketing overlays, watermarks, contact details, QR codes, prices, platform marks and sensitive visuals. Unsafe references are excluded from derivative generation. The orchestrator chooses candidate counts from one to four and prioritizes source-image roles separately for the hero and every detail job. Candidate selectors enforce product identity, construction, color and hard usability constraints without redefining the model's storyboard. The complete surviving candidate state—including the actual current candidate and local-review evidence—is reviewed as a six-image set; it is never truncated by generation order or repaired by substituting a fabricated current index. The set may be changed again through the top-level repair loop. Generated video is fully decoded and silent by default. Deterministic source images and catalog video remain emergency completeness fallbacks when a generation capability is unavailable or produces no valid artifact.

Task-specific skills are shipped under `crossborder_agent/skills/` and progressively loaded into the relevant model calls: delivery planning, product grounding, AliExpress taxonomy, AliExpress content compliance, marketplace localization, commerce visuals, commerce video, and the A1-A7 rubric evaluator. They are local prompt-policy packages; taxonomy exploration uses the agent's own bounded function tools over the supplied local snapshots and does not rely on MCP, an external vector service, or an external taxonomy API.

Required environment variables:

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- `OPENAI_BASE_URL`
- `AGENT_LOG_DIR`

A normal platform run uses all configured model endpoints. If one or more model endpoint variables are absent, the runtime records the configuration problem and continues through the deterministic availability fallback instead of returning an empty failed submission. `--offline` remains the explicit development mode.

Optional model overrides:

- `AGENT_CHAT_MODEL`
- `AGENT_CHAT_FALLBACK_MODEL`
- `AGENT_REVIEW_MODEL`
- `AGENT_REVIEW_FALLBACK_MODEL`
- `AGENT_EVALUATION_MODELS` (comma-separated list of 2 or 3 evaluator models)
- `AGENT_IMAGE_MODEL`
- `AGENT_IMAGE_FALLBACK_MODEL`
- `AGENT_VIDEO_MODEL`
- `AGENT_KEEP_VIDEO_AUDIO=1` to retain source audio explicitly; the default removes unreviewed audio

Defaults are `qwen3.8-max` for planning and localization, `qwen3.8-max,qwen3.7-plus` for independent evidence-finding, `wan2.7-image-pro` for source-guided image generation, `qwen-image-3.0-pro` as the cross-model image fallback, and `wan2.7-i2v-2026-04-25` for video.

The automated suite focuses on complete protocol smoke runs, deterministic tool boundaries, safety/data-integrity contracts and structurally valid delivery output. Semantic quality and generalization belong in holdout evals rather than unit tests that freeze one model trajectory, prompt wording, call count or sample answer. Resilience tests cover rate limits, terminal client errors, malformed JSON, corrupt media, failed asynchronous tasks and deadline-aware fallback.

## Development smoke run

The input directory must contain exactly one product JSON. To use the supplied multi-product development dataset, select a product ID:

```bash
python agent.py \
  --input-dir Data_for_Users \
  --output-dir /tmp/material-agent-output \
  --product-id 3887087154767 \
  --offline
```

`--offline` skips model calls but still downloads supplied source URLs, produces localized fallback copy, normalizes images, creates a playable video, and executes all delivery gates. Distinct seller views are exhausted first; if too few remain, bounded neckline/hem crops are generated and rechecked against all accepted image hashes before the delivery is installed.

## Build the submission

```bash
python scripts/build_submission.py
```

This downloads pinned CPython 3.12 manylinux x86_64 wheels, vendors Pillow and the ffmpeg fallback runtime, and creates `dist/agent.zip`. The ZIP has `agent.py` at its root and is rejected by the build script if it exceeds 100 MB.
