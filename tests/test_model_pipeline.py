from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

from crossborder_agent.media import create_slideshow_video
from crossborder_agent.pipeline import Pipeline
from crossborder_agent.qa import EXPECTED_FILES, validate_delivery


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data_for_Users"


class _ModelServiceHandler(BaseHTTPRequestHandler):
    source_image: Path
    source_video: Path

    def log_message(self, format, *args):  # noqa: A003
        return

    @property
    def root_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def _json(self, body: dict, status: int = 200) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _file(self, path: Path, content_type: str) -> None:
        content = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):  # noqa: N802
        if self.path in {"/source.jpg", "/source-2.jpg"}:
            self._file(self.source_image, "image/jpeg")
        elif self.path == "/video.mp4":
            self._file(self.source_video, "video/mp4")
        elif self.path == "/dash/tasks/video-task":
            self._json(
                {
                    "output": {
                        "task_id": "video-task",
                        "task_status": "SUCCEEDED",
                        "video_url": self.root_url + "/video.mp4",
                    }
                }
            )
        else:
            self._json({"message": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        if self.path == "/openai/chat/completions":
            self._chat(body)
        elif self.path == "/dash/services/aigc/multimodal-generation/generation":
            requested = max(1, int(body.get("parameters", {}).get("n", 1)))
            image_urls = [self.root_url + "/source.jpg"]
            if requested > 1:
                image_urls.append(self.root_url + "/source-2.jpg")
            self._json(
                {
                    "output": {
                        "choices": [
                            {
                                "message": {
                                    "content": [{"image": url} for url in image_urls]
                                }
                            }
                        ]
                    }
                }
            )
        elif self.path == "/dash/services/aigc/video-generation/video-synthesis":
            self._json({"output": {"task_id": "video-task", "task_status": "PENDING"}})
        else:
            self._json({"message": "not found"}, 404)

    def _chat(self, body: dict) -> None:
        system = str(body["messages"][0]["content"])
        user_content = body["messages"][1]["content"]
        if "creative director" in system:
            long_prompt = (
                "Create a source-faithful premium e-commerce product composition with neutral lighting, "
                "clear product visibility, no text or claims, exact construction and color preservation. "
            )
            payload = {
                "visual_theme": "clean neutral marketplace editorial",
                "main_prompt": long_prompt * 2,
                "detail_prompts": [
                    long_prompt * 2 + f" Storyboard slot {index}." for index in range(5)
                ],
                "video_prompt": long_prompt * 2,
                "market_angles": {
                    "en": "clarity",
                    "ko": "정확한 옵션",
                    "pt": "clareza",
                },
            }
        elif "visual inspector" in system:
            image_count = sum(
                1 for item in user_content if item.get("type") == "image_url"
            )
            payload = {
                "product_type": "shirt",
                "visible_colors": ["purple"],
                "visible_design_features": ["long sleeves"],
                "best_hero_image_index": 0,
                "image_quality_notes": [],
                "prohibited_or_risky_visuals": [],
                "preservation_constraints": ["preserve shape"],
                "images": [
                    {
                        "index": index,
                        "role": "hero" if index == 0 else "detail",
                        "dominant_color": "purple",
                        "product_coverage": "high",
                        "sharpness": "high",
                        "has_text": False,
                        "has_logo": False,
                        "has_watermark": False,
                        "has_contact_info": False,
                        "has_qr_code": False,
                        "has_price_or_discount": False,
                        "has_review_graphic": False,
                        "has_certification_seal": False,
                        "has_platform_mark": False,
                        "has_third_party_brand": False,
                        "has_before_after": False,
                        "adult_or_sensitive_visual": False,
                        "product_obscured": False,
                        "low_sharpness": False,
                        "safe_for_generation_reference": True,
                        "risk_reasons": [],
                    }
                    for index in range(image_count)
                ],
            }
        elif "hero-image selector" in system:
            payload = {
                "selected_index": 1,
                "candidates": [
                    {
                        "index": index,
                        "usable": True,
                        "identity_consistent": True,
                        "construction_consistent": True,
                        "correct_color": True,
                        "single_product": True,
                        "product_complete": True,
                        "clean_neutral_background": True,
                        "has_person": False,
                        "has_unrelated_props": False,
                        "unwanted_text": False,
                        "unwanted_brand_or_logo": False,
                        "major_artifacts": False,
                        "unexpected_collage": False,
                        "product_coverage": "high",
                        "score": 90 + index,
                        "reason": "source-faithful",
                    }
                    for index in range(2)
                ],
            }
        elif "image quality gate" in system:
            image_count = sum(
                1 for item in user_content if item.get("type") == "image_url"
            )
            payload = {
                "assets": [
                    {
                        "index": index,
                        "usable": True,
                        "identity_consistent": True,
                        "construction_consistent": True,
                        "slot_match": True,
                        "unwanted_text": False,
                        "major_artifacts": False,
                        "product_coverage": "high",
                        "reason": "ok",
                    }
                    for index in range(image_count)
                ]
            }
        elif "product-video quality gate" in system:
            payload = {
                "usable": True,
                "identity_consistent": True,
                "unwanted_text": False,
                "major_artifacts": False,
                "reason": "ok",
            }
        elif "multimodal evidence reconciler" in system:
            payload = {
                "seller_title_decision": "publish",
                "attribute_decisions": [],
                "canonical_visual_claims": [],
                "conflicts": [],
            }
        elif "multimodal acceptance evaluator" in system:
            payload = {
                "ready_for_delivery": True,
                "weighted_score": 94,
                "dimension_scores": {
                    "A1": 94,
                    "A2": 94,
                    "A3": 94,
                    "A4": 94,
                    "A5": 94,
                    "A6": 94,
                    "A7": 94,
                },
                "summary": "All delivery dimensions pass the acceptance threshold.",
                "issues": [],
                "repair_actions": [],
            }
        else:
            if "ko-KR" in system:
                title, overview, fit = (
                    "여성 셔츠",
                    "원본 정보를 바탕으로 구성한 상품입니다.",
                    "판매자 사이즈 안내를 확인해 주세요.",
                )
            elif "pt-BR" in system:
                title, overview, fit = (
                    "Camisa feminina",
                    "Produto descrito com base nos dados de origem.",
                    "Consulte a orientação de tamanho do vendedor.",
                )
            else:
                title, overview, fit = (
                    "Women's Shirt",
                    "A product listing grounded in the supplied source data.",
                    "Review the seller-provided size guidance.",
                )
            payload = {
                "title": title,
                "overview": overview,
                "highlights": [
                    "Source-grounded details",
                    "Verified variant matrix",
                    "Conservative sizing guidance",
                ],
                "fit_note": fit,
                "media_descriptions": {
                    "main_image.jpeg": "Primary product image",
                    "detail_image_1.jpeg": "Overall product view",
                    "detail_image_2.jpeg": "Visible construction details",
                    "detail_image_3.jpeg": "Verified feature view",
                    "detail_image_4.jpeg": "Source-grounded variants",
                    "detail_image_5.jpeg": "Practical product context",
                    "product_video.mp4": "Short product video",
                },
            }
        content = json.dumps(payload, ensure_ascii=False)
        self._json(
            {"choices": [{"message": {"role": "assistant", "content": content}}]}
        )


@unittest.skipIf(
    Image is None, "Pillow is required for the model protocol integration test"
)
class ModelPipelineTests(unittest.TestCase):
    def test_model_protocol_and_complete_delivery(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent-model-e2e-") as temporary:
            root = Path(temporary)
            source_image = root / "source.jpg"
            source_video = root / "video.mp4"
            fixture = Image.new("RGB", (1400, 1400), (248, 248, 248))
            draw = ImageDraw.Draw(fixture)
            draw.rounded_rectangle(
                (360, 170, 1040, 1230), radius=70, fill=(173, 145, 207)
            )
            draw.line((700, 210, 700, 1170), fill=(242, 236, 248), width=18)
            for y in range(350, 1050, 150):
                draw.ellipse((680, y, 720, y + 40), fill=(245, 245, 245))
            fixture.save(source_image, quality=94)
            create_slideshow_video(source_image, source_video, duration=2)

            handler = type("ConfiguredModelHandler", (_ModelServiceHandler,), {})
            handler.source_image = source_image
            handler.source_video = source_video
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                input_dir = root / "input"
                output_dir = root / "output"
                (input_dir / "product_info").mkdir(parents=True)
                source = json.loads(
                    (DATA / "product_info/product_3887087154767.json").read_text(
                        encoding="utf-8"
                    )
                )
                product = source["ret"]["result"]["result"]
                product["productImage"]["images"] = [base_url + "/source.jpg"] * 5
                product["description"] = f"<img src='{base_url}/source.jpg'>"
                for sku in product["productSkuInfos"]:
                    for attribute in sku["skuAttributes"]:
                        if attribute.get("skuImageUrl"):
                            attribute["skuImageUrl"] = base_url + "/source.jpg"
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

                environment = {
                    "DASHSCOPE_API_KEY": "test-key",
                    "DASHSCOPE_BASE_URL": base_url + "/dash",
                    "OPENAI_BASE_URL": base_url + "/openai",
                }
                logger = logging.getLogger("model-pipeline-test")
                logger.addHandler(logging.NullHandler())
                with mock.patch.dict(os.environ, environment, clear=False):
                    state = Pipeline(
                        input_dir=input_dir,
                        output_dir=output_dir,
                        logger=logger,
                        timeout_seconds=900,
                    ).run()
                self.assertEqual(
                    {path.name for path in output_dir.iterdir()}, EXPECTED_FILES
                )
                generated_names = {
                    asset.name for asset in state.assets if asset.generated
                }
                self.assertIn("main_image.jpeg", generated_names)
                self.assertIn("product_video.mp4", generated_names)
                self.assertTrue(
                    any("近重复" in warning for warning in state.warnings),
                    state.warnings,
                )
                report = validate_delivery(output_dir, state.facts, state.taxonomy)
                self.assertTrue(report.valid, report.errors)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
