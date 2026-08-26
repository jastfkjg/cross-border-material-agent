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

from PIL import Image

from crossborder_agent.api import ApiConfig, ApiError, HttpJsonClient, QwenClient
from crossborder_agent.media import MediaError, inspect_image_quality, normalize_image


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

    def test_malformed_json_response_is_retried(self) -> None:
        self.handler.counters["/bad-json"] = 0
        with mock.patch("crossborder_agent.api.time.sleep", return_value=None):
            response = self._http().request_json(
                self.base + "/bad-json", method="GET", body=None
            )
        self.assertTrue(response["recovered"])
        self.assertEqual(self.handler.counters["/bad-json"], 2)

    def test_corrupt_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent-corrupt-") as temporary:
            root = Path(temporary)
            source = root / "corrupt.asset"
            source.write_bytes(b"this is not an image")
            with self.assertRaises(MediaError):
                normalize_image(source, root / "out.jpeg", canvas=(800, 800))

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
