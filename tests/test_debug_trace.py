from __future__ import annotations

import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from crossborder_agent.cli import build_parser
from crossborder_agent.debug_trace import DebugTrace
from crossborder_agent.logging_utils import configure_logging


class DebugTraceTests(unittest.TestCase):
    def test_debug_flag_is_opt_in(self) -> None:
        parser = build_parser()
        self.assertFalse(parser.parse_args([]).debug)
        self.assertTrue(parser.parse_args(["--debug"]).debug)

    def test_run_profile_defaults_to_full_but_honors_local_environment(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(build_parser().parse_args([]).run_profile, "full")
        with mock.patch.dict(
            os.environ, {"AGENT_RUN_PROFILE": "fast"}, clear=True
        ):
            self.assertEqual(build_parser().parse_args([]).run_profile, "fast")

    def test_local_runner_defaults_to_submission_safe_full_profile(self) -> None:
        script = (
            Path(__file__).resolve().parents[1] / "run_agent_local.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('RUN_PROFILE="${RUN_PROFILE:-full}"', script)
        self.assertIn('--run-profile "$RUN_PROFILE"', script)
        self.assertIn('",".join(config.evaluation_models)', script)

    def test_trace_is_disabled_by_default(self) -> None:
        logger = logging.getLogger("debug-disabled-test")
        with self.assertLogs(logger, level="INFO") as captured:
            logger.info("baseline")
            DebugTrace(logger, enabled=False).emit("should.not.exist", value=1)
        self.assertFalse(any("TRACE_JSON" in line for line in captured.output))

    def test_trace_is_structured_and_redacts_secrets_and_url_queries(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent-debug-log-") as temporary:
            with mock.patch.dict(
                os.environ, {"AGENT_LOG_DIR": temporary}, clear=False
            ):
                logger = configure_logging(debug=True)
                trace = DebugTrace(logger, enabled=True)
                trace.emit(
                    "unit.event",
                    elapsed_seconds=1.25,
                    api_key="do-not-log",
                    request={
                        "authorization": "Bearer secret",
                        "url": "https://example.test/model?signature=private",
                    },
                    result={"selected_index": 2},
                )
                for handler in logger.handlers:
                    handler.flush()
                text = (Path(temporary) / "agent_debug.jsonl").read_text(encoding="utf-8")
                normal_log = (Path(temporary) / "agent.log").read_text(encoding="utf-8")

        self.assertIn("TRACE_JSON", text)
        self.assertIn('"event":"unit.event"', text)
        self.assertIn('"selected_index":2', text)
        self.assertIn('"run_elapsed_seconds":', text)
        self.assertIn('"operation_duration_seconds":1.25', text)
        self.assertIn("[redacted]", text)
        self.assertNotIn("do-not-log", text)
        self.assertNotIn("Bearer secret", text)
        self.assertNotIn("signature=private", text)
        self.assertNotIn("TRACE_JSON", normal_log)


if __name__ == "__main__":
    unittest.main()
