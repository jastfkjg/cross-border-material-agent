from __future__ import annotations

import functools
import json
import logging
import re
import shutil
import tempfile
import threading
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

from crossborder_agent.pipeline import Pipeline
from crossborder_agent.qa import EXPECTED_FILES, validate_delivery


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data_for_Users"


@unittest.skipIf(Image is None, "Pillow is required for the media integration test")
class OfflinePipelineTests(unittest.TestCase):
    def test_complete_offline_delivery(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent-e2e-") as temporary:
            root = Path(temporary)
            web = root / "web"
            input_dir = root / "input"
            output_dir = root / "output"
            web.mkdir()
            (input_dir / "product_info").mkdir(parents=True)

            fixture = Image.new("RGB", (1400, 1400), (248, 248, 248))
            draw = ImageDraw.Draw(fixture)
            draw.rounded_rectangle(
                (360, 170, 1040, 1230), radius=70, fill=(173, 145, 207)
            )
            draw.line((700, 210, 700, 1170), fill=(242, 236, 248), width=18)
            for y in range(350, 1050, 150):
                draw.ellipse((680, y, 720, y + 40), fill=(245, 245, 245))
            fixture.save(web / "source.jpg", quality=94)
            handler = functools.partial(SimpleHTTPRequestHandler, directory=str(web))
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                image_url = f"http://127.0.0.1:{server.server_port}/source.jpg"
                source_path = DATA / "product_info/product_3887087154767.json"
                source = json.loads(source_path.read_text(encoding="utf-8"))
                product = source["ret"]["result"]["result"]
                product["productImage"]["images"] = [image_url] * 5
                product["description"] = f"<img src='{image_url}'>"
                for sku in product["productSkuInfos"]:
                    for attribute in sku["skuAttributes"]:
                        if attribute.get("skuImageUrl"):
                            attribute["skuImageUrl"] = image_url
                (input_dir / "product_info/product.json").write_text(
                    json.dumps(source, ensure_ascii=False), encoding="utf-8"
                )
                shutil.copy2(
                    DATA / "clothing_categories.json",
                    input_dir / "clothing_categories.json",
                )
                shutil.copy2(
                    DATA / "clothing_attributes.json",
                    input_dir / "clothing_attributes.json",
                )

                log_path = root / "agent.log"
                logger = logging.getLogger(f"offline-pipeline-test-{id(root)}")
                log_handler = logging.FileHandler(log_path, encoding="utf-8")
                logger.handlers = [log_handler]
                logger.propagate = False
                state = Pipeline(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    logger=logger,
                    timeout_seconds=240,
                    offline=True,
                ).run()
                log_handler.flush()
                self.assertEqual(
                    {path.name for path in output_dir.iterdir()}, EXPECTED_FILES
                )
                report = validate_delivery(output_dir, state.facts, state.taxonomy)
                self.assertTrue(report.valid, report.errors)
                self.assertGreater(
                    (output_dir / "product_video.mp4").stat().st_size, 1000
                )
                self.assertLess(
                    (output_dir / "main_image.jpeg").stat().st_size,
                    1024 * 1024,
                )
                strategy = (output_dir / "strategy_document.md").read_text(
                    encoding="utf-8"
                )
                for expected in (
                    "Factual Consistency and Source-Conflict Handling",
                    "Localization Strategy for the Three Target Markets",
                    "Six-Image Narrative and Video Role",
                    "Factual Consistency, Compliance, and Asset Quality Control",
                    "Listing-Ready Delivery Assurance",
                    "United States English (en-US)",
                    "South Korean market (ko-KR)",
                    "Brazilian Portuguese (pt-BR)",
                    "main_image.jpeg | Conversion Entry Point",
                    "product_video.mp4 | Dynamic Overview",
                    "The Agent follows five stages",
                ):
                    self.assertIn(expected, strategy)
                self.assertIsNone(re.search(r"[\u4e00-\u9fff]", strategy))
                self.assertIsNone(
                    re.search(
                        r"[\u4e00-\u9fff]",
                        log_path.read_text(encoding="utf-8"),
                    )
                )
                for internal_log_section in (
                    "Claim Ledger",
                    "Model configuration:",
                    "Global evaluation trace",
                    "Targeted repair trace",
                    "API failure summary",
                    "Runtime quality-control record",
                    "Single-model evidence penalty score",
                    "raw cells were read",
                ):
                    self.assertNotIn(internal_log_section, strategy)
            finally:
                if "log_handler" in locals():
                    log_handler.close()
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
