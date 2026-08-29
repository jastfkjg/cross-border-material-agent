from __future__ import annotations

import json
import logging
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

try:
    from PIL import Image
except ImportError:
    Image = None

from crossborder_agent.api import (
    ApiConfig,
    ApiError,
    HttpJsonClient,
    QwenClient,
    _failure_category,
)
from crossborder_agent.media import (
    MediaError,
    create_catalog_video,
    inspect_image_quality,
    inspect_video,
    normalize_image,
)


class _FaultHandler(BaseHTTPRequestHandler):
    counters: dict[str, int] = {}

    def log_message(self, format, *args):  # noqa: A003
        return

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if status == 429:
            self.send_header("Retry-After", "0")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        self.counters[self.path] = self.counters.get(self.path, 0) + 1
        count = self.counters[self.path]
        if self.path == "/retry-json" and count == 1:
            self._json({"message": "rate limited"}, 429)
        elif self.path == "/retry-json":
            self._json({"ok": True})
        elif self.path == "/bad-json" and count == 1:
            body = b"{not-json"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/bad-json":
            self._json({"recovered": True})
        elif self.path == "/dash/tasks/failed-task":
            self._json(
                {
                    "output": {
                        "task_id": "failed-task",
                        "task_status": "FAILED",
                        "message": "injected failure",
                    }
                }
            )
        else:
            self._json({"message": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        if self.path == "/bad-request":
            self._json({"code": "InvalidParameter", "message": "bad input"}, 400)
        else:
            self._json({"message": "not found"}, 404)


class ResilienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.handler = type("ConfiguredFaultHandler", (_FaultHandler,), {})
        cls.handler.counters = {}
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), cls.handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.logger = logging.getLogger("resilience-test")
        cls.logger.addHandler(logging.NullHandler())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def _http(self) -> HttpJsonClient:
        return HttpJsonClient(self.logger, time.monotonic() + 30)

    def test_429_retries_and_honors_retry_after(self) -> None:
        self.handler.counters["/retry-json"] = 0
        response = self._http().request_json(
            self.base + "/retry-json", method="GET", body=None
        )
        self.assertTrue(response["ok"])
        self.assertEqual(self.handler.counters["/retry-json"], 2)

    def test_non_retryable_400_preserves_status(self) -> None:
        with self.assertRaises(ApiError) as raised:
            self._http().request_json(self.base + "/bad-request", body={})
        self.assertEqual(raised.exception.status_code, 400)
        self.assertFalse(raised.exception.retryable)

    def test_tool_serving_invalid_parameter_is_not_misclassified_as_model(self) -> None:
        message = (
            '{"error":{"code":"invalid_parameter_error",'
            '"message":"An error occurred in model serving: Invalid request parameters",'
            '"type":"invalid_request_error"}}'
        )
        self.assertEqual(_failure_category(400, message), "invalid_request")

    def test_video_configuration_failure_disables_capability_for_repairs(self) -> None:
        config = ApiConfig(
            api_key="test",
            dashscope_base_url=self.base + "/",
            openai_base_url=self.base + "/",
            video_model="missing-video-model",
        )
        client = QwenClient(config, self.logger, time.monotonic() + 300)

        with self.assertRaises(ApiError) as raised:
            client.generate_video("product motion", "https://example.test/frame.jpeg")

        self.assertEqual(raised.exception.category, "model_configuration")
        self.assertFalse(client.operation_available("video"))

    def test_malformed_json_response_is_retried(self) -> None:
        self.handler.counters["/bad-json"] = 0
        with mock.patch("crossborder_agent.api.time.sleep", return_value=None):
            response = self._http().request_json(
                self.base + "/bad-json", method="GET", body=None
            )
        self.assertTrue(response["recovered"])
        self.assertEqual(self.handler.counters["/bad-json"], 2)

    def test_chat_json_retries_malformed_model_content(self) -> None:
        config = ApiConfig(
            api_key="test",
            dashscope_base_url=self.base + "/dash",
            openai_base_url=self.base + "/openai",
        )
        client = QwenClient(config, self.logger, time.monotonic() + 300)
        with mock.patch.object(
            client,
            "_chat_response",
            side_effect=["not-json", '{"recovered": true}'],
        ) as response:
            payload = client.chat_json("system", "prompt")
        self.assertTrue(payload["recovered"])
        self.assertEqual(response.call_count, 2)

    def test_chat_json_with_trace_does_not_require_tool_metadata(self) -> None:
        config = ApiConfig(
            api_key="test",
            dashscope_base_url=self.base + "/dash",
            openai_base_url=self.base + "/openai",
        )
        trace = mock.Mock()
        client = QwenClient(
            config, self.logger, time.monotonic() + 300, trace=trace
        )
        with mock.patch.object(client, "_chat_response", return_value='{"ok": true}'):
            payload = client.chat_json("system", "prompt")
        self.assertTrue(payload["ok"])

    def test_chat_tool_step_sends_native_tools_and_preserves_messages(self) -> None:
        config = ApiConfig(
            api_key="test",
            dashscope_base_url=self.base + "/dash",
            openai_base_url=self.base + "/openai",
        )
        client = QwenClient(config, self.logger, time.monotonic() + 300)
        assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "search_categories",
                        "arguments": '{"query":"工装"}',
                    },
                }
            ],
        }
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_categories",
                    "description": "search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        prior_messages = [{"role": "user", "content": "product evidence"}]

        with mock.patch.object(
            client, "_chat_tool_response", return_value=assistant
        ) as response:
            result = client.chat_tool_step("system", prior_messages, tools)

        self.assertEqual(result, assistant)
        body = response.call_args.args[0]
        self.assertEqual(body["messages"][1:], prior_messages)
        self.assertEqual(body["tools"], tools)
        self.assertEqual(body["tool_choice"], "required")
        self.assertFalse(body["parallel_tool_calls"])

    def test_chat_tool_step_traces_request_shape_without_content(self) -> None:
        config = ApiConfig(
            api_key="test",
            dashscope_base_url=self.base + "/dash",
            openai_base_url=self.base + "/openai",
        )
        trace = mock.Mock()
        client = QwenClient(
            config, self.logger, time.monotonic() + 300, trace=trace
        )
        assistant = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "inspect", "arguments": "{}"},
                }
            ],
        }
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "inspect",
                    "description": "inspect",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        with mock.patch.object(client, "_chat_tool_response", return_value=assistant):
            client.chat_tool_step("secret system content", [], tools)

        shape = next(
            call for call in trace.emit.call_args_list
            if call.args and call.args[0] == "api.tool_request_shape"
        )
        self.assertGreater(shape.kwargs["request_bytes"], 0)
        self.assertGreater(shape.kwargs["tool_schema_bytes"], 0)
        self.assertEqual(shape.kwargs["tool_count"], 1)
        self.assertNotIn("secret system content", repr(shape))

    def test_review_model_uses_its_own_fallback_chain(self) -> None:
        config = ApiConfig(
            api_key="test",
            dashscope_base_url=self.base + "/dash",
            openai_base_url=self.base + "/openai",
            chat_model="producer",
            chat_fallback_model="producer-fallback",
            review_model="reviewer",
            review_fallback_model="reviewer-fallback",
        )
        client = QwenClient(config, self.logger, time.monotonic() + 300)
        called_models: list[str] = []

        def response(body):
            called_models.append(body["model"])
            if len(called_models) == 1:
                raise ApiError("reviewer busy", retryable=True, category="queue")
            return '{"ok": true}'

        with mock.patch.object(client, "_chat_response", side_effect=response):
            payload = client.chat_json(
                "system", "prompt", model=config.review_model
            )
        self.assertTrue(payload["ok"])
        self.assertEqual(called_models, ["reviewer", "reviewer-fallback"])

    @unittest.skipIf(Image is None, "Pillow is required")
    def test_corrupt_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent-corrupt-") as temporary:
            root = Path(temporary)
            source = root / "corrupt.asset"
            source.write_bytes(b"this is not an image")
            with self.assertRaises(MediaError):
                normalize_image(source, root / "out.jpeg", canvas=(800, 800))

    @unittest.skipIf(Image is None, "Pillow is required")
    def test_focus_crop_is_bounded_and_visually_distinct(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent-focus-crop-") as temporary:
            root = Path(temporary)
            source = root / "source.png"
            image = Image.new("RGB", (900, 1200), (245, 245, 245))
            for y in range(image.height):
                color = (30, 80 + y // 10, 220 - y // 8)
                for x in range(image.width):
                    if x < image.width // 2:
                        image.putpixel((x, y), color)
            image.save(source)

            full = root / "full.jpeg"
            upper = root / "upper.jpeg"
            normalize_image(source, full, canvas=(600, 750))
            normalize_image(source, upper, canvas=(600, 750), focus_crop="upper")

            with Image.open(upper) as rendered:
                self.assertEqual(rendered.size, (600, 750))
            full_quality = inspect_image_quality(full)
            upper_quality = inspect_image_quality(upper)
            self.assertIsNotNone(full_quality)
            self.assertIsNotNone(upper_quality)
            self.assertGreater(
                (full_quality.difference_hash ^ upper_quality.difference_hash).bit_count(),
                0,
            )

    @unittest.skipIf(Image is None, "Pillow is required")
    def test_catalog_video_with_distinct_stills_decodes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent-catalog-video-") as temporary:
            root = Path(temporary)
            images = []
            for index, color in enumerate(((190, 80, 60), (50, 120, 205))):
                path = root / f"source-{index}.png"
                image = Image.new("RGB", (900, 1200), color)
                for offset in range(80, 820, 120):
                    for y in range(150 + index * 20, 1050, 160):
                        image.paste((245, 235, 210), (offset, y, offset + 55, y + 90))
                image.save(path)
                images.append(path)
            destination = root / "catalog.mp4"
            create_catalog_video(images, destination, duration=3)
            result = inspect_video(destination)
            self.assertTrue(result["decoded"])
            self.assertGreater(result["size_bytes"], 1000)

    def test_failed_async_task_is_terminal(self) -> None:
        config = ApiConfig(
            api_key="test",
            dashscope_base_url=self.base + "/dash",
            openai_base_url=self.base + "/openai",
        )
        client = QwenClient(config, self.logger, time.monotonic() + 30)
        with self.assertRaisesRegex(ApiError, "injected failure"):
            client._poll_task_for_url(
                {"output": {"task_id": "failed-task", "task_status": "PENDING"}},
                preferred_keys=("video", "url"),
                timeout_seconds=10,
            )

    def test_low_deadline_skips_slow_image_fallback(self) -> None:
        config = ApiConfig(
            api_key="test",
            dashscope_base_url=self.base + "/dash",
            openai_base_url=self.base + "/openai",
        )
        client = QwenClient(config, self.logger, time.monotonic() + 120)
        with mock.patch.object(
            client, "_generate_sync_images", side_effect=ApiError("injected")
        ) as generate:
            with self.assertRaisesRegex(ApiError, "跳过慢速图像回退模型"):
                client.generate_image_candidates(
                    "prompt", [], size="1024*1024", count=2
                )
        self.assertEqual(generate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
