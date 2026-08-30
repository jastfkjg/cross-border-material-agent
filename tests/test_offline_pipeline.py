from __future__ import annotations

import functools
import json
import logging
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

                logger = logging.getLogger("offline-pipeline-test")
                logger.addHandler(logging.NullHandler())
                state = Pipeline(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    logger=logger,
                    timeout_seconds=240,
                    offline=True,
                ).run()
                self.assertEqual(
                    {path.name for path in output_dir.iterdir()}, EXPECTED_FILES
                )
                report = validate_delivery(output_dir, state.facts, state.taxonomy)
                self.assertTrue(report.valid, report.errors)
                self.assertGreater(
                    (output_dir / "product_video.mp4").stat().st_size, 1000
                )
                strategy = (output_dir / "strategy_document.md").read_text(
                    encoding="utf-8"
                )
                for expected in (
                    "事实一致性与源数据冲突处理",
                    "三个目标市场的本地化策略",
                    "六图视觉叙事与视频职责",
                    "事实一致、合规与素材质检",
                    "可直接上架的交付保障",
                    "美国英语（en-US）",
                    "韩国市场（ko-KR）",
                    "巴西葡萄牙语（pt-BR）",
                    "main_image.jpeg｜转化入口",
                    "product_video.mp4｜动态总览",
                ):
                    self.assertIn(expected, strategy)
                for internal_log_section in (
                    "Claim Ledger",
                    "模型配置：",
                    "全局评估轨迹",
                    "定向修复轨迹",
                    "API 失败摘要",
                    "运行质检记录",
                    "单模型证据惩罚分",
                ):
                    self.assertNotIn(internal_log_section, strategy)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
