import unittest

from scripts.diagnose_tool_call_400 import (
    make_probe_body,
    request_metadata,
    transcript_issues,
)


class ToolCallDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.body = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "initial"},
                {"role": "user", "content": "RUNTIME BUDGET: turn 1/8"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "inspect_product",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "inspect_product",
                    "content": '{"facts":"secret fixture content"}',
                },
            ],
            "tools": [],
            "parallel_tool_calls": True,
        }

    def test_valid_transcript_has_no_pairing_issues(self):
        self.assertEqual([], transcript_issues(self.body["messages"]))

    def test_metadata_does_not_copy_message_content(self):
        metadata = request_metadata(self.body)
        self.assertEqual(1, metadata["assistant_null_content_count"])
        self.assertEqual(1, metadata["tool_name_field_count"])
        self.assertNotIn("secret fixture content", repr(metadata))

    def test_probes_change_only_the_targeted_compatibility_surface(self):
        normalized = make_probe_body("normalized_assistant_content", self.body)
        self.assertEqual("", normalized["messages"][3]["content"])
        self.assertIsNone(self.body["messages"][3]["content"])

        no_budget = make_probe_body("without_budget_messages", self.body)
        self.assertFalse(
            any(
                item.get("role") == "user"
                and str(item.get("content")).startswith("RUNTIME BUDGET:")
                for item in no_budget["messages"]
            )
        )

        no_parallel = make_probe_body("without_parallel_tool_calls", self.body)
        self.assertNotIn("parallel_tool_calls", no_parallel)
        self.assertIn("parallel_tool_calls", self.body)


if __name__ == "__main__":
    unittest.main()
