from __future__ import annotations

import json
import logging
import time
import unittest
from types import SimpleNamespace

from crossborder_agent.agent_loop import (
    AgentLoopTool,
    AgentToolOutcome,
    NativeToolAgentLoop,
)
from crossborder_agent.agent_tools import BoundedToolRegistry
from crossborder_agent.api import ApiError
from crossborder_agent.bounded_agent import BoundedDeliveryAgent


class ScriptedToolClient:
    def __init__(self) -> None:
        self.turn = 0
        self.seen_messages: list[list[dict]] = []

    def chat_tool_step(self, system, messages, tools):
        self.seen_messages.append(list(messages))
        self.turn += 1
        if self.turn == 1:
            name, arguments = "inspect", "{}"
        else:
            name, arguments = "finish", '{"reason":"evidence is sufficient"}'
        return {
            "content": None,
            "tool_calls": [
                {
                    "id": f"call-{self.turn}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        }


class NativeToolAgentLoopTests(unittest.TestCase):
    def test_planner_visual_rows_exclude_heavy_source_payloads(self) -> None:
        source = {
            "index": 7,
            "url": "https://example.test/very-long-source.jpeg",
            "role": "detail",
            "dominant_color": "blue",
            "product_coverage": "high",
            "sharpness": "high",
            "inspection_complete": True,
            "safe_for_generation_reference": True,
            "has_text": True,
            "risk_reasons": ["requires cleanup"],
            "raw_provider_transcript": "x" * 100_000,
        }

        compact = BoundedDeliveryAgent._compact_source_image_detail(source)

        self.assertEqual(compact["index"], 7)
        self.assertTrue(compact["safe_reference"])
        self.assertNotIn("url", compact)
        self.assertNotIn("raw_provider_transcript", compact)

    def test_delivery_planner_tool_surface_stays_within_provider_budget(self) -> None:
        class CapturingClient:
            config = SimpleNamespace(chat_model="planner")

            def __init__(self):
                self.http = SimpleNamespace(deadline=time.monotonic() + 300)
                self.body: dict | None = None

            def chat_tool_step(self, system, messages, tools):
                self.body = {
                    "model": "planner",
                    "messages": [{"role": "system", "content": system}, *messages],
                    "tools": tools,
                    "tool_choice": "required",
                    "parallel_tool_calls": False,
                    "temperature": 0.0,
                    "enable_thinking": False,
                }
                raise ApiError("captured", category="invalid_request")

        client = CapturingClient()
        agent = BoundedDeliveryAgent(client, logging.getLogger("plan-schema-budget"))
        facts = SimpleNamespace(compact_dict=lambda: {"offer_id": "synthetic"})
        taxonomy = SimpleNamespace(
            category=SimpleNamespace(
                category_id="leaf", name="Synthetic", path="Root > Synthetic"
            ),
            attributes=[],
            missing_required=[],
        )

        agent.plan_delivery(facts, taxonomy, {}, BoundedToolRegistry())

        self.assertIsNotNone(client.body)
        body = client.body or {}
        tool_names = [item["function"]["name"] for item in body["tools"]]
        self.assertEqual(
            set(tool_names),
            {
                "list",
                "read",
                "search",
                "bash",
                "write_staging",
                "inspect_evidence",
                "submit_delivery_plan",
            },
        )
        self.assertEqual(tool_names[-1], "submit_delivery_plan")
        encoded = json.dumps(
            body, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        self.assertLess(len(encoded), 13_000)

    def test_complete_protocol_exposes_observations_and_finishes(self) -> None:
        client = ScriptedToolClient()
        loop = NativeToolAgentLoop(client, system_prompt="Choose tools from evidence.")
        tools = [
            AgentLoopTool(
                "inspect",
                "Inspect evidence.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                lambda _: {"evidence": "grounded"},
            ),
            AgentLoopTool(
                "finish",
                "Finish.",
                {
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                    "additionalProperties": False,
                },
                lambda arguments: AgentToolOutcome(
                    {"accepted": bool(arguments["reason"])}, terminate=True
                ),
            ),
        ]

        result = loop.run(
            "Handle the task.",
            tools,
            max_turns=4,
            deadline=time.monotonic() + 30,
        )

        self.assertEqual(result.stop_reason, "finish-tool")
        self.assertEqual(result.turns, 2)
        self.assertTrue(result.final_observation["accepted"])
        second_turn_messages = client.seen_messages[1]
        observation = next(item for item in second_turn_messages if item["role"] == "tool")
        self.assertIn("grounded", observation["content"])

    def test_parallel_batch_is_serialized_before_history_replay(self) -> None:
        class MixedClient:
            @staticmethod
            def chat_tool_step(system, messages, tools):
                return {
                    "content": None,
                    "tool_calls": [
                        {"id": "inspect", "function": {"name": "inspect", "arguments": "{}"}},
                        {"id": "finish", "function": {"name": "finish", "arguments": "{}"}},
                    ],
                }

        finished: list[bool] = []
        loop = NativeToolAgentLoop(MixedClient(), system_prompt="Use tools.")
        result = loop.run(
            "Handle the task.",
            [
                AgentLoopTool(
                    "inspect",
                    "Inspect.",
                    {"type": "object", "properties": {}},
                    lambda _: {"seen": True},
                ),
                AgentLoopTool(
                    "finish",
                    "Finish alone.",
                    {"type": "object", "properties": {}},
                    lambda _: (finished.append(True) or AgentToolOutcome({}, terminate=True)),
                    terminal=True,
                ),
            ],
            max_turns=1,
            deadline=time.monotonic() + 30,
        )

        self.assertEqual(result.stop_reason, "max-turns")
        self.assertEqual(finished, [])
        self.assertTrue(result.final_observation["seen"])
        assistant = next(item for item in result.messages if item["role"] == "assistant")
        self.assertEqual(len(assistant["tool_calls"]), 1)
        self.assertEqual(assistant["tool_calls"][0]["id"], "inspect")

    def test_delivery_plan_rejection_is_observed_and_corrected_before_finish(self) -> None:
        safe_prompt = (
            "Create a source-faithful product photograph with neutral lighting, one coherent composition, "
            "accurate visible construction and no invented product features."
        )
        plan_arguments = {
            "creative_direction": "Grounded marketplace presentation",
            "localization_priorities": {"en": "natural US copy", "ko": "natural Korean copy", "pt": "natural Brazilian copy"},
            "risk_priorities": ["A1", "A5"],
            "execution_order": ["hero", "details", "copy", "video"],
            "video_strategy": "Stable source-faithful product motion",
            "creative_plan": {
                "visual_theme": "Restrained neutral marketplace editorial styling",
                "main": {"prompt": safe_prompt, "candidate_count": 2, "reference_roles": ["hero"]},
                "details": [
                    {
                        "role": f"evidence_job_{index}",
                        "prompt": safe_prompt + f" Assigned evidence job {index}.",
                        "candidate_count": 1,
                        "reference_roles": ["detail"],
                    }
                    for index in range(1, 6)
                ],
                "video": {"prompt": safe_prompt + " Use stable restrained camera motion."},
                "market_angles": {
                    "en": "clear product decisions",
                    "ko": "명확한 상품 정보 전달",
                    "pt": "decisões claras sobre o produto",
                },
            },
        }

        class PlanningClient:
            config = SimpleNamespace(chat_model="planner")

            def __init__(self):
                self.http = SimpleNamespace(deadline=time.monotonic() + 300)
                self.turn = 0
                self.seen_messages: list[list[dict]] = []

            def chat_tool_step(self, system, messages, tools):
                self.turn += 1
                self.seen_messages.append(list(messages))
                arguments = json.loads(json.dumps(plan_arguments))
                if self.turn == 1:
                    arguments["creative_plan"]["main"]["prompt"] += " Include a coupon."
                    arguments["risk_priorities"] = ["A1", "A1"]
                    arguments["execution_order"] = ["copy", "hero", "details", "video"]
                return {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"submit-{self.turn}",
                            "function": {
                                "name": "submit_delivery_plan",
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }

        client = PlanningClient()
        agent = BoundedDeliveryAgent(client, logging.getLogger("plan-validation"))
        facts = SimpleNamespace(compact_dict=lambda: {"offer_id": "synthetic"})
        taxonomy = SimpleNamespace(
            category=SimpleNamespace(category_id="leaf", name="Synthetic", path="Root > Synthetic"),
            attributes=[],
            missing_required=[],
        )
        result = agent.plan_delivery(facts, taxonomy, {}, BoundedToolRegistry())

        self.assertEqual(client.turn, 2)
        rejection = next(
            item for item in client.seen_messages[1] if item.get("role") == "tool"
        )
        self.assertIn("correction_required", rejection["content"])
        self.assertIn("risk_priorities", rejection["content"])
        self.assertIn("execution_order", rejection["content"])
        self.assertIn("forbidden visual-prompt terms", rejection["content"])
        self.assertIn("creative_plan", result)
        self.assertEqual(
            result["creative_plan"]["main"]["prompt"], safe_prompt
        )


if __name__ == "__main__":
    unittest.main()
