"""Small HTTP adapters for the model APIs allowed by the evaluation platform."""

from __future__ import annotations

import json
import logging
import os
import random
import ssl
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from .debug_trace import DebugTrace


class ApiError(RuntimeError):
    """A retryable or terminal model service failure."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool | None = None,
        category: str = "unknown",
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.category = category


def _failure_category(status_code: int | None, message: str) -> str:
    """Classify provider failures so upper layers can choose a useful recovery."""

    lowered = message.casefold()
    if status_code == 429 or any(
        token in lowered
        for token in ("rate limit", "ratelimit", "throttl", "quota exceeded")
    ):
        return "rate_limit"
    if status_code in {409, 425} or any(
        token in lowered
        for token in ("queue", "queued", "capacity", "overloaded", "busy")
    ):
        return "queue"
    if status_code == 408 or any(
        token in lowered for token in ("timeout", "timed out", "deadline exceeded")
    ):
        return "timeout"
    if status_code in {500, 502, 503, 504}:
        return "server"
    if status_code in {401, 403}:
        return "authorization"
    if status_code in {400, 404} and any(
        token in lowered
        for token in ("model", "not found", "does not exist", "unsupported")
    ):
        return "model_configuration"
    if status_code is not None and 400 <= status_code < 500:
        return "invalid_request"
    return "unknown"


@dataclass(frozen=True, slots=True)
class ApiConfig:
    api_key: str
    dashscope_base_url: str
    openai_base_url: str
    chat_model: str = "qwen3.8-max"
    chat_fallback_model: str = "qwen3.7-plus"
    review_model: str = "qwen3.8-max"
    review_fallback_model: str = "qwen3.7-plus"
    visual_review_model: str = ""
    visual_review_fallback_model: str = ""
    image_model: str = "wan2.7-image-pro"
    image_fallback_model: str = "qwen-image-3.0-pro"
    video_model: str = "wan2.7-i2v-2026-04-25"

    @classmethod
    def from_environment(cls) -> "ApiConfig | None":
        api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        dashscope = os.environ.get("DASHSCOPE_BASE_URL", "").strip()
        openai = os.environ.get("OPENAI_BASE_URL", "").strip()
        if not api_key or not dashscope or not openai:
            return None
        return cls(
            api_key=api_key,
            dashscope_base_url=dashscope,
            openai_base_url=openai,
            chat_model=os.environ.get("AGENT_CHAT_MODEL", "qwen3.8-max"),
            chat_fallback_model=os.environ.get(
                "AGENT_CHAT_FALLBACK_MODEL", "qwen3.7-plus"
            ),
            review_model=os.environ.get("AGENT_REVIEW_MODEL", "qwen3.8-max"),
            review_fallback_model=os.environ.get(
                "AGENT_REVIEW_FALLBACK_MODEL", "qwen3.7-plus"
            ),
            visual_review_model=os.environ.get(
                "AGENT_VISUAL_REVIEW_MODEL",
                os.environ.get("AGENT_REVIEW_MODEL", "qwen3.8-max"),
            ),
            visual_review_fallback_model=os.environ.get(
                "AGENT_VISUAL_REVIEW_FALLBACK_MODEL",
                os.environ.get("AGENT_REVIEW_FALLBACK_MODEL", "qwen3.7-plus"),
            ),
            image_model=os.environ.get("AGENT_IMAGE_MODEL", "wan2.7-image-pro"),
            image_fallback_model=os.environ.get(
                "AGENT_IMAGE_FALLBACK_MODEL", "qwen-image-3.0-pro"
            ),
            video_model=os.environ.get("AGENT_VIDEO_MODEL", "wan2.7-i2v-2026-04-25"),
        )


def _endpoint(base_url: str, suffix: str) -> str:
    base = base_url.rstrip("/") + "/"
    suffix_clean = suffix.lstrip("/")
    if suffix_clean.startswith("chat/completions") and base.rstrip("/").endswith(
        "chat/completions"
    ):
        return base.rstrip("/")
    return urljoin(base, suffix_clean)


def _collect_urls(value: Any, *, preferred_keys: tuple[str, ...] = ()) -> list[str]:
    preferred: list[str] = []
    other: list[str] = []

    def visit(node: Any, key: str = "") -> None:
        if isinstance(node, dict):
            for child_key, child_value in node.items():
                visit(child_value, str(child_key).lower())
        elif isinstance(node, list):
            for child in node:
                visit(child, key)
        elif isinstance(node, str) and node.startswith(("https://", "http://")):
            if any(token in key for token in preferred_keys):
                preferred.append(node)
            else:
                other.append(node)

    visit(value)
    result: list[str] = []
    for url in preferred + other:
        if url not in result:
            result.append(url)
    return result


class HttpJsonClient:
    def __init__(
        self,
        logger: logging.Logger,
        deadline: float,
        trace: DebugTrace | None = None,
    ):
        self.logger = logger
        self.deadline = deadline
        self.trace = trace
        self.ssl_context = ssl.create_default_context()

    def _remaining(self, maximum: float) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ApiError("全局运行截止时间已到")
        return max(1.0, min(maximum, remaining))

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def request_json(
        self,
        url: str,
        *,
        method: str = "POST",
        headers: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
        timeout: float = 180,
        attempts: int = 4,
    ) -> dict[str, Any]:
        payload = (
            None
            if body is None
            else json.dumps(body, ensure_ascii=False).encode("utf-8")
        )
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        last_error = ""
        for attempt in range(attempts):
            if self.trace:
                self.trace.emit(
                    "http.request.attempt",
                    method=method,
                    url=url,
                    attempt=attempt + 1,
                    max_attempts=attempts,
                    remaining_seconds=round(self.remaining_seconds, 1),
                )
            request = Request(url, data=payload, headers=request_headers, method=method)
            try:
                with urlopen(
                    request, timeout=self._remaining(timeout), context=self.ssl_context
                ) as response:
                    data = response.read()
                parsed = json.loads(data.decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise ApiError(
                        "API 返回值不是 JSON 对象",
                        retryable=True,
                        category="response_format",
                    )
                if self.trace:
                    self.trace.emit(
                        "http.request.success",
                        method=method,
                        url=url,
                        attempt=attempt + 1,
                    )
                return parsed
            except HTTPError as exc:
                detail = exc.read(8192).decode("utf-8", errors="replace")
                last_error = f"HTTP {exc.code}: {detail}"
                retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
                if self.trace:
                    self.trace.emit(
                        "http.request.failure",
                        method=method,
                        url=url,
                        attempt=attempt + 1,
                        status_code=exc.code,
                        retryable=retryable,
                        category=_failure_category(exc.code, last_error),
                        error=last_error,
                    )
                if not retryable or attempt + 1 >= attempts:
                    raise ApiError(
                        last_error,
                        status_code=exc.code,
                        retryable=retryable,
                        category=_failure_category(exc.code, last_error),
                    ) from exc
                retry_after = exc.headers.get("Retry-After", "")
                try:
                    delay = min(30.0, max(0.0, float(retry_after)))
                except ValueError:
                    delay = min(18.0, (2**attempt) + random.random() * 1.5)
            except ApiError as exc:
                last_error = str(exc)
                if self.trace:
                    self.trace.emit(
                        "http.request.failure",
                        method=method,
                        url=url,
                        attempt=attempt + 1,
                        status_code=exc.status_code,
                        retryable=exc.retryable,
                        category=exc.category,
                        error=last_error,
                    )
                if exc.retryable is not True or attempt + 1 >= attempts:
                    raise
                delay = min(8.0, (2**attempt) + random.random())
            except (
                URLError,
                TimeoutError,
                OSError,
                json.JSONDecodeError,
                UnicodeError,
            ) as exc:
                last_error = str(exc)
                category = (
                    "response_format"
                    if isinstance(exc, (json.JSONDecodeError, UnicodeError))
                    else "timeout"
                    if isinstance(exc, TimeoutError)
                    else "network"
                )
                if self.trace:
                    self.trace.emit(
                        "http.request.failure",
                        method=method,
                        url=url,
                        attempt=attempt + 1,
                        retryable=True,
                        category=category,
                        error=last_error,
                    )
                if attempt + 1 >= attempts:
                    raise ApiError(
                        last_error, retryable=True, category=category
                    ) from exc
                delay = min(18.0, (2**attempt) + random.random() * 1.5)
            if self.trace:
                self.trace.emit(
                    "http.request.retry_scheduled",
                    method=method,
                    url=url,
                    attempt=attempt + 1,
                    delay_seconds=round(delay, 3),
                    error=last_error,
                )
            self.logger.warning(
                "API 请求失败，%.1f 秒后重试: %s", delay, last_error[:500]
            )
            time.sleep(min(delay, self._remaining(delay)))
        raise ApiError(
            last_error or "API 请求失败", retryable=True, category="unknown"
        )

    def download(
        self, url: str, path: Path, *, max_bytes: int, timeout: float = 180
    ) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ApiError(f"拒绝下载非法 URL: {url!r}")
        last_error = ""
        for attempt in range(4):
            try:
                request = Request(
                    url, headers={"User-Agent": "crossborder-material-agent/1.1"}
                )
                with urlopen(
                    request, timeout=self._remaining(timeout), context=self.ssl_context
                ) as response:
                    content_length = response.headers.get("Content-Length")
                    if content_length and int(content_length) > max_bytes:
                        raise ApiError(f"下载文件超过限制: {content_length} bytes")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    total = 0
                    with path.open("wb") as handle:
                        while True:
                            block = response.read(1024 * 256)
                            if not block:
                                break
                            total += len(block)
                            if total > max_bytes:
                                raise ApiError(f"下载文件超过限制: {max_bytes} bytes")
                            handle.write(block)
                return
            except (
                HTTPError,
                URLError,
                TimeoutError,
                OSError,
                ValueError,
                ApiError,
            ) as exc:
                last_error = str(exc)
                path.unlink(missing_ok=True)
                if attempt == 3:
                    raise ApiError(f"下载失败: {last_error}") from exc
                delay = min(10.0, (2**attempt) + random.random())
                self.logger.warning(
                    "下载失败，%.1f 秒后重试: %s", delay, last_error[:500]
                )
                time.sleep(min(delay, self._remaining(delay)))


class QwenClient:
    """Explicit model adapters; no model-built-in tools are used."""

    def __init__(
        self,
        config: ApiConfig,
        logger: logging.Logger,
        deadline: float,
        trace: DebugTrace | None = None,
    ):
        self.config = config
        self.logger = logger
        self.trace = trace
        self.http = HttpJsonClient(logger, deadline, trace)
        self._chat_slots = threading.BoundedSemaphore(3)
        self._image_slots = threading.BoundedSemaphore(2)
        self._video_slots = threading.BoundedSemaphore(1)
        self._metrics: list[dict[str, Any]] = []
        self._metrics_lock = threading.Lock()
        self._disabled_models: set[str] = set()
        self._disabled_models_lock = threading.Lock()

    @property
    def metrics(self) -> list[dict[str, Any]]:
        with self._metrics_lock:
            return [dict(item) for item in self._metrics]

    def _record_metric(
        self,
        *,
        operation: str,
        model: str,
        started: float,
        status: str,
        error: str = "",
    ) -> None:
        item: dict[str, Any] = {
            "operation": operation,
            "model": model,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "status": status,
            "remaining_seconds": round(self.http.remaining_seconds, 1),
        }
        if error:
            item["error"] = error[:300]
        with self._metrics_lock:
            self._metrics.append(item)
        if self.trace:
            self.trace.emit("api.operation", **item)

    @property
    def model_summary(self) -> dict[str, str]:
        return {
            "chat": self.config.chat_model,
            "review": self.config.review_model,
            "review_fallback": self.config.review_fallback_model,
            "visual_review": self.visual_review_model,
            "visual_review_fallback": self.visual_review_fallback_model,
            "image": self.config.image_model,
            "image_fallback": self.config.image_fallback_model,
            "video": self.config.video_model,
        }

    @property
    def visual_review_model(self) -> str:
        return self.config.visual_review_model or self.config.review_model

    @property
    def visual_review_fallback_model(self) -> str:
        return (
            self.config.visual_review_fallback_model
            or self.config.review_fallback_model
        )

    def _chat_response(self, body: dict[str, Any]) -> str:
        started = time.monotonic()
        selected_model = str(body.get("model") or "")
        endpoint = _endpoint(self.config.openai_base_url, "chat/completions")
        try:
            response = self.http.request_json(
                endpoint,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                body=body,
                timeout=180,
            )
        except ApiError as exc:
            self._record_metric(
                operation="chat",
                model=selected_model,
                started=started,
                status="error",
                error=str(exc),
            )
            raise
        self._record_metric(
            operation="chat",
            model=selected_model,
            started=started,
            status="ok",
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ApiError(
                f"Chat API 返回结构异常: {json.dumps(response, ensure_ascii=False)[:1000]}",
                retryable=True,
                category="response_format",
            ) from exc
        if isinstance(content, list):
            text_parts = [
                str(item.get("text", "")) for item in content if isinstance(item, dict)
            ]
            return "".join(text_parts).strip()
        return str(content).strip()

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        images: Iterable[str] = (),
        videos: Iterable[str] = (),
        model: str = "",
        fallback_model: str = "",
    ) -> dict[str, Any]:
        user_content: list[dict[str, Any]] = []
        for url in images:
            if url:
                user_content.append({"type": "image_url", "image_url": {"url": url}})
        for url in videos:
            if url:
                user_content.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": url},
                        "fps": 1.0,
                        "min_pixels": 65_536,
                        "max_pixels": 655_360,
                    }
                )
        user_content.append({"type": "text", "text": user_prompt})
        selected_model = model or self.config.chat_model
        body = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            # Classification, factual copy, and semantic QA should be as stable
            # as the provider allows; creative variation is delegated to the
            # dedicated image/video models instead of these JSON decisions.
            "temperature": 0.0,
            "enable_thinking": False,
        }
        models = [selected_model]
        selected_fallback = fallback_model
        if not selected_fallback:
            selected_fallback = (
                self.config.review_fallback_model
                if selected_model == self.config.review_model
                else self.config.chat_fallback_model
            )
        if selected_fallback not in models:
            models.append(selected_fallback)
        last_error: ApiError | None = None
        with self._chat_slots:
            for candidate_model in models:
                body["model"] = candidate_model
                # Invalid JSON is a judge-output failure, not evidence that the
                # reviewed artifact is bad. Retry structure once before changing judge.
                for format_attempt in range(2):
                    try:
                        text = self._chat_response(body)
                    except ApiError as exc:
                        last_error = exc
                        if exc.category in {"authorization", "invalid_request"}:
                            raise
                        break
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        last_error = ApiError(
                            f"模型未返回合法 JSON: {text[:1000]}",
                            retryable=True,
                            category="response_format",
                        )
                        if format_attempt == 0 and self.http.remaining_seconds > 90:
                            self.logger.warning(
                                "模型 %s 返回非法 JSON，重试一次结构化评审",
                                candidate_model,
                            )
                            continue
                        break
                    if not isinstance(parsed, dict):
                        last_error = ApiError(
                            "模型 JSON 顶层不是对象",
                            retryable=True,
                            category="response_format",
                        )
                        if format_attempt == 0 and self.http.remaining_seconds > 90:
                            continue
                        break
                    return parsed
                if candidate_model != models[-1] and self.http.remaining_seconds > 60:
                    self.logger.warning(
                        "模型 %s 的结构化调用失败，切换评审回退模型: %s",
                        candidate_model,
                        last_error,
                    )
        raise last_error or ApiError("结构化模型调用失败")

    def analyze_product_images(
        self,
        facts_json: str,
        image_urls: list[str],
        *,
        skill_instructions: str = "",
    ) -> dict[str, Any]:
        system = (
            "You are a conservative e-commerce visual inspector. Return JSON only. "
            "Never infer fabric composition, performance, measurements, brand authorization, or care instructions "
            "from appearance. Separate direct observations from uncertain impressions."
            + ("\n\n" + skill_instructions if skill_instructions else "")
        )
        prompt = f"""
Inspect the supplied source product images and the verified source facts below.
Return a JSON object with keys:
- product_type: string
- visible_colors: string[]
- visible_design_features: string[]
- best_hero_image_index: integer, zero-based
- image_quality_notes: string[]
- prohibited_or_risky_visuals: string[]
- preservation_constraints: string[]
- size_chart_rows: an array containing only rows directly readable from supplied size-chart images. Each object must contain:
  - size_label: exact size code such as S, M, XL or 2XL
  - bust_cm: numeric string in centimeters, or empty string when not shown
  - length_cm: numeric string in centimeters, or empty string when not shown
  - weight_guidance: exact visible seller guidance including its unit, or empty string
  - source_image_index: zero-based index of the supplied image containing the row
- images: an array of exactly {len(image_urls[:12])} objects, one for every supplied image in input order.
  Every object must contain:
  - index: zero-based integer
  - role: one of hero, front, back, side, detail, variant, size_chart, lifestyle, packaging, unknown
  - dominant_color: concise observed color or empty string
  - product_coverage: one of high, medium, low, unknown
  - sharpness: one of high, medium, low, unknown
  - has_text: boolean
  - has_overlay_text: boolean (marketing copy, captions or text placed over/background around the product)
  - has_intrinsic_product_text: boolean (text physically printed, embroidered or sewn on the sellable product)
  - has_logo: boolean
  - has_watermark: boolean
  - has_contact_info: boolean
  - has_qr_code: boolean
  - has_price_or_discount: boolean
  - has_review_graphic: boolean
  - has_certification_seal: boolean
  - has_platform_mark: boolean
  - has_third_party_brand: boolean (true only when a third-party brand is visible or strongly suspected)
  - has_before_after: boolean
  - adult_or_sensitive_visual: boolean
  - has_hate_or_extremism: boolean
  - has_violence_or_weapon: boolean
  - has_drugs_tobacco_or_alcohol: boolean
  - product_obscured: boolean
  - low_sharpness: boolean
  - has_person: boolean
  - has_unrelated_props: boolean (bags, newspapers, furniture or styling objects not part of the product)
  - multiple_products: boolean
  - product_complete: boolean (the entire sellable item is visible without cropping or obstruction)
  - clean_neutral_background: boolean (white or near-white studio background without a lifestyle scene)
  - safe_for_generation_reference: boolean
  - risk_reasons: string[]

Read all visible text as carefully as possible. A product's own sewn label or print still counts as
has_text and has_intrinsic_product_text, but not has_overlay_text. Marketing captions around the garment
count as has_overlay_text. Contact details, QR codes, marketplace marks, watermarks, price/discount badges,
review graphics, certification seals and sensitive/prohibited content make safe_for_generation_reference false.
A person, ordinary prop, background text or suspected third-party styling mark makes the source unsuitable
for direct listing, but it may remain a generation reference when the product is clear and the final generated
asset removes those elements.
For size_chart_rows, transcribe only complete, clearly legible rows. Do not estimate obscured digits, convert
units, rename size codes, or infer measurements from the garment. Return an empty array when no table is legible.
For a marketplace hero, a person, unrelated prop, multiple products, incomplete product or lifestyle
background makes the source unsuitable even when it remains usable as a detail reference.

Verified source facts:
{facts_json}
""".strip()
        return self.chat_json(
            system,
            prompt,
            images=image_urls[:12],
            model=self.visual_review_model,
            fallback_model=self.visual_review_fallback_model,
        )

    def review_generated_images(
        self,
        facts_json: str,
        source_image_urls: list[str],
        generated_image_urls: list[str],
        expected_assets: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        system = (
            "You are a strict product-listing image quality gate. Return JSON only. "
            "Reject product identity drift, changed colors or construction, invented text or claims, watermarks, "
            "visual artifacts, distorted anatomy, unreadable layouts, and images where the product is obscured."
        )
        prompt = f"""
The first {len(source_image_urls)} supplied images are trusted source references. The next
{len(generated_image_urls)} images are generated listing assets. Compare every generated asset
against the source references and verified facts. Do not reject a faithful crop or a changed neutral
background; reject changes to the product itself.

Return JSON with these top-level keys:
- assets: an array of exactly {len(generated_image_urls)} objects. Each object must contain:
- index: zero-based integer
- usable: boolean
- identity_consistent: boolean
- construction_consistent: boolean (check visible collar, sleeves, pockets, buttons/fasteners, hem and pattern)
- color_consistent: boolean
- pattern_consistent: boolean
- slot_match: boolean (whether this asset fulfills its intended storyboard purpose)
- unwanted_text: boolean
- prohibited_visual: boolean (price, discount, contact details, QR code, review badge, certification seal, platform mark or adult/sensitive content)
- major_artifacts: boolean
- unexpected_collage: boolean (true for montage, grid, split-screen or repeated panels; detail slot 4 may intentionally show a clean variant lineup)
- product_coverage: one of high, medium, low
- reason: concise string
- set_usable: boolean; false only for a hard defect or a materially redundant/unfinished set
- usable_count: integer from 0 to {len(generated_image_urls)}
- distinct_commercial_roles: integer from 0 to {len(generated_image_urls)}
- coherent: boolean (consistent product identity and campaign style without making all slots alike)
- near_duplicate_pairs: array of two-index arrays, for example [[1,2]]
- missing_roles: array of intended purposes that are absent or materially under-delivered
- repair_targets: array of generated indices whose replacement would most improve the set
- summary: concise set-level diagnosis

Judge the six assets as one commercial collection after judging each image. Penalize repeated camera
angles, repeated crops, repeated backgrounds or two slots that communicate the same shopper value even
when each image is individually usable. Do not call faithful consistency a duplicate when framing and
commercial purpose are clearly distinct. A deterministic size chart may be visually different by design.

Verified facts:
{facts_json}

Expected asset purposes in generated-image order:
{json.dumps(expected_assets or [], ensure_ascii=False)}
""".strip()
        return self.chat_json(
            system,
            prompt,
            images=[*source_image_urls, *generated_image_urls],
            model=self.visual_review_model,
            fallback_model=self.visual_review_fallback_model,
        )

    def select_best_generated_image(
        self,
        facts_json: str,
        source_image_urls: list[str],
        candidate_urls: list[str],
    ) -> dict[str, Any]:
        system = (
            "You are a strict e-commerce hero-image selector. Return JSON only. "
            "The source images define product identity. Favor exact construction, color and pattern, "
            "a single unobscured product, clean background, high product coverage and no text or marks."
        )
        prompt = f"""
The first {len(source_image_urls)} images are trusted source references. The next
{len(candidate_urls)} images are generated hero candidates. Compare every candidate with the source.

Return JSON with:
- selected_index: zero-based candidate index, or -1 if every candidate is unusable
- candidates: exactly {len(candidate_urls)} objects, each containing index, usable,
  identity_consistent, construction_consistent, correct_color, single_product,
  product_complete, clean_neutral_background, has_person, has_unrelated_props,
  anatomy_natural, unwanted_text, unwanted_brand_or_logo, major_artifacts, product_coverage
  (high/medium/low), score (0-100), and reason.

Explicitly inspect visible product type, silhouette, collar/neckline, sleeve or leg length,
pockets, button/fastener count where visible, hem, pattern and any product logo.
Reject any candidate that reproduces a background/styling brand, character, store text or logo
that is not intrinsic to the sellable product.
Prefer a product-only hero. For adult apparel, a single fully clothed adult wearer is acceptable only
when a trusted source already shows a wearer, the garment remains complete and unobstructed, anatomy is
natural, and the clean near-white hero composition is preserved. Never add a wearer for children's goods.

Verified facts:
{facts_json}
""".strip()
        return self.chat_json(
            system,
            prompt,
            images=[*source_image_urls, *candidate_urls],
            model=self.visual_review_model,
            fallback_model=self.visual_review_fallback_model,
        )

    def select_best_detail_image(
        self,
        facts_json: str,
        source_image_urls: list[str],
        candidate_urls: list[str],
        *,
        asset_name: str,
        purpose: str,
    ) -> dict[str, Any]:
        system = (
            "You are a strict e-commerce detail-image selector. Return JSON only. "
            "The source images define product identity. Reject product drift, invented variants, "
            "unwanted text, prohibited commerce elements, anatomy errors and storyboard mismatch."
        )
        prompt = f"""
The first {len(source_image_urls)} images are trusted source references. The next
{len(candidate_urls)} images are candidates for {asset_name}.

Intended storyboard purpose:
{purpose}

Return JSON with:
- selected_index: zero-based candidate index, or -1 if every candidate is unusable
- candidates: exactly {len(candidate_urls)} objects, each containing index, usable,
  identity_consistent, construction_consistent, color_consistent, pattern_consistent,
  slot_match, critical_structure_unambiguous, anatomy_natural, single_composition,
  unexpected_collage, unwanted_text,
  unwanted_brand_or_logo, prohibited_visual, major_artifacts,
  product_coverage (high/medium/low), score (0-100), and reason.

For a variant lineup, every shown variant must be visibly supported by a source reference.
Evaluate critical_structure_unambiguous as a quality signal relative to the intended storyboard
purpose, not as an automatic rejection. A close-up does not need to show the complete garment: it must
clearly show its assigned local feature plus enough identity anchors to connect it to the verified
product. Reject only when the crop causes actual product-identity drift or fails the assigned slot.
Full-view, variant and wearer slots should keep the category-defining structure recognizable. For a variant lineup, each item must remain
unambiguously the same product; folded long sleeves must still be visibly present rather than making
the item appear sleeveless.
For a wearer scene, reject malformed hands, limbs, faces, garment fit or body proportions.
Every non-variant detail must use one coherent full-frame composition. Mark unexpected_collage true
for a montage, grid, split screen, inset, repeated panel, or a hybrid close-up/full-product layout.
The variant slot may show several complete verified variants in one physical catalog composition, but
must not use dividers, inset panels, duplicated views, or unrelated close-ups.
Reject copied background/styling brands, characters, store text or logos unrelated to the product.

Verified facts:
{facts_json}
""".strip()
        return self.chat_json(
            system,
            prompt,
            images=[*source_image_urls, *candidate_urls],
            model=self.visual_review_model,
            fallback_model=self.visual_review_fallback_model,
        )

    def review_generated_video(
        self,
        facts_json: str,
        source_image_urls: list[str],
        video_url: str,
        *,
        current_video_url: str = "",
    ) -> dict[str, Any]:
        system = (
            "You are a strict e-commerce product-video quality gate. Return JSON only. "
            "Compare the generated video with the trusted source product images. Reject identity drift, "
            "changed color or construction, morphing, duplicated limbs or garments, unreadable or unwanted "
            "text, watermarks, violent camera motion, severe flicker, and frames where the product is obscured."
        )
        if current_video_url:
            prompt = f"""
The supplied images are trusted source references. Video 0 is the current accepted asset and video 1
is a proposed repair. Review both completely. Return JSON with selected_index and exactly two candidates.
Each candidate must contain index, usable, identity_consistent, construction_consistent,
color_and_pattern_consistent, motion_stable, unwanted_text, prohibited_visual, major_artifacts,
score (0-100), and reason. Select video 1 only when it is at least 6 points better and has no hard defect.

Verified facts:
{facts_json}
""".strip()
            videos = [current_video_url, video_url]
        else:
            prompt = f"""
The supplied images are trusted source references. The supplied video is a generated listing asset.
Review the entire video and return a JSON object with exactly these keys:
- usable: boolean
- identity_consistent: boolean
- construction_consistent: boolean
- color_and_pattern_consistent: boolean
- motion_stable: boolean (no morphing, severe flicker, sudden cut or camera shake)
- unwanted_text: boolean
- prohibited_visual: boolean
- major_artifacts: boolean
- score: number from 0 to 100
- reason: concise string

Verified facts:
{facts_json}
""".strip()
            videos = [video_url]
        return self.chat_json(
            system,
            prompt,
            images=source_image_urls,
            videos=videos,
            model=self.visual_review_model,
            fallback_model=self.visual_review_fallback_model,
        )

    def generate_image(
        self,
        prompt: str,
        reference_urls: list[str],
        *,
        size: str,
        negative_prompt: str = "",
    ) -> tuple[str, str]:
        urls, model = self.generate_image_candidates(
            prompt,
            reference_urls,
            size=size,
            negative_prompt=negative_prompt,
            count=1,
        )
        return urls[0], model

    def generate_image_candidates(
        self,
        prompt: str,
        reference_urls: list[str],
        *,
        size: str,
        negative_prompt: str = "",
        count: int = 2,
    ) -> tuple[list[str], str]:
        errors: list[str] = []
        requested = max(1, min(4, int(count)))
        with self._image_slots:
            if not self._model_disabled(self.config.image_model):
                try:
                    return self._generate_images_with_strategy(
                        self.config.image_model,
                        prompt,
                        reference_urls,
                        size=size,
                        negative_prompt=negative_prompt,
                        count=requested,
                    ), self.config.image_model
                except ApiError as exc:
                    errors.append(f"{self.config.image_model}: {exc}")
                    self._disable_model_on_configuration_error(
                        self.config.image_model, exc
                    )
                    self.logger.warning("主图像模型失败，切换回退模型: %s", exc)
            else:
                errors.append(f"{self.config.image_model}: 本轮已因配置错误熔断")
            if self.http.remaining_seconds < 300:
                errors.append("剩余时间不足 300 秒，跳过慢速图像回退模型")
                raise ApiError("; ".join(errors), retryable=False)
            if self._model_disabled(self.config.image_fallback_model):
                errors.append(
                    f"{self.config.image_fallback_model}: 本轮已因配置错误熔断"
                )
                raise ApiError("; ".join(errors), retryable=False)
            try:
                return self._generate_images_with_strategy(
                    self.config.image_fallback_model,
                    prompt,
                    reference_urls,
                    size=size,
                    negative_prompt=negative_prompt,
                    count=1,
                ), self.config.image_fallback_model
            except ApiError as exc:
                errors.append(f"{self.config.image_fallback_model}: {exc}")
                self._disable_model_on_configuration_error(
                    self.config.image_fallback_model, exc
                )
        raise ApiError("; ".join(errors))

    def _model_disabled(self, model: str) -> bool:
        with self._disabled_models_lock:
            return model in self._disabled_models

    def _disable_model_on_configuration_error(
        self, model: str, error: ApiError
    ) -> None:
        if error.category not in {"authorization", "model_configuration"}:
            return
        with self._disabled_models_lock:
            self._disabled_models.add(model)
        self.logger.error(
            "模型 %s 因不可恢复的 %s 错误在本轮熔断",
            model,
            error.category,
        )

    def _generate_images_with_strategy(
        self,
        model: str,
        prompt: str,
        reference_urls: list[str],
        *,
        size: str,
        negative_prompt: str,
        count: int,
    ) -> list[str]:
        last_error: ApiError | None = None
        for submission_attempt in range(2):
            try:
                return self._generate_sync_images(
                    model,
                    prompt,
                    reference_urls,
                    size=size,
                    negative_prompt=negative_prompt,
                    count=count,
                )
            except ApiError as exc:
                last_error = exc
                transient = exc.retryable is True or exc.category in {
                    "rate_limit",
                    "queue",
                    "timeout",
                    "server",
                    "network",
                    "response_format",
                }
                if (
                    not transient
                    or submission_attempt > 0
                    or self.http.remaining_seconds < 480
                ):
                    raise
                delay = 4.0 if exc.category in {"rate_limit", "queue"} else 2.0
                self.logger.warning(
                    "图像模型 %s 在底层重试后仍遇到 %s，%.1f 秒后重新提交一次",
                    model,
                    exc.category,
                    delay,
                )
                time.sleep(min(delay, self.http._remaining(delay)))
        raise last_error or ApiError("图像生成失败")

    def _generate_sync_images(
        self,
        model: str,
        prompt: str,
        reference_urls: list[str],
        *,
        size: str,
        negative_prompt: str,
        count: int,
    ) -> list[str]:
        reference_limit = 9 if model.startswith("wan2.7-image") else 3
        content: list[dict[str, str]] = [
            {"image": url} for url in reference_urls[:reference_limit]
        ]
        content.append({"text": prompt})
        parameters: dict[str, Any] = {
            "size": size,
            "watermark": False,
            "n": count,
        }
        if not model.startswith("wan2.7-image"):
            parameters["prompt_extend"] = True
            if negative_prompt:
                parameters["negative_prompt"] = negative_prompt
        body = {
            "model": model,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": parameters,
        }
        started = time.monotonic()
        try:
            response = self.http.request_json(
                _endpoint(
                    self.config.dashscope_base_url,
                    "services/aigc/multimodal-generation/generation",
                ),
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                body=body,
                timeout=360,
            )
        except ApiError as exc:
            self._record_metric(
                operation="image",
                model=model,
                started=started,
                status="error",
                error=str(exc),
            )
            raise
        self._record_metric(
            operation="image",
            model=model,
            started=started,
            status="ok",
        )
        urls = _collect_urls(response, preferred_keys=("image", "url"))
        if not urls:
            raise ApiError(
                f"图像响应没有 URL: {json.dumps(response, ensure_ascii=False)[:1000]}",
                retryable=True,
                category="response_format",
            )
        return urls[:count]

    def generate_video(
        self, prompt: str, first_frame_url: str, *, negative_prompt: str = ""
    ) -> tuple[str, str]:
        body = {
            "model": self.config.video_model,
            "input": {
                "prompt": prompt,
                "media": [{"type": "first_frame", "url": first_frame_url}],
            },
            "parameters": {
                "resolution": "720P",
                "duration": 8,
                "prompt_extend": False,
                "watermark": False,
            },
        }
        if negative_prompt:
            body["input"]["negative_prompt"] = negative_prompt
        started = time.monotonic()
        with self._video_slots:
            last_error: ApiError | None = None
            for submission_attempt in range(2):
                try:
                    response = self.http.request_json(
                        _endpoint(
                            self.config.dashscope_base_url,
                            "services/aigc/video-generation/video-synthesis",
                        ),
                        headers={
                            "Authorization": f"Bearer {self.config.api_key}",
                            "X-DashScope-Async": "enable",
                        },
                        body=body,
                        timeout=180,
                    )
                    url = self._poll_task_for_url(
                        response,
                        preferred_keys=("video", "url"),
                        timeout_seconds=720,
                    )
                    break
                except ApiError as exc:
                    last_error = exc
                    transient = exc.retryable is True or exc.category in {
                        "rate_limit",
                        "queue",
                        "timeout",
                        "server",
                        "network",
                        "response_format",
                    }
                    if (
                        not transient
                        or submission_attempt > 0
                        or self.http.remaining_seconds < 480
                    ):
                        self._record_metric(
                            operation="video",
                            model=self.config.video_model,
                            started=started,
                            status="error",
                            error=str(exc),
                        )
                        raise
                    self.logger.warning(
                        "视频任务因 %s 失败，保留时间预算后重新提交一次: %s",
                        exc.category,
                        exc,
                    )
                    time.sleep(min(4.0, self.http._remaining(4.0)))
            else:
                raise last_error or ApiError("视频生成失败")
        self._record_metric(
            operation="video",
            model=self.config.video_model,
            started=started,
            status="ok",
        )
        return url, self.config.video_model

    def _poll_task_for_url(
        self,
        initial_response: dict[str, Any],
        *,
        preferred_keys: tuple[str, ...],
        timeout_seconds: int,
    ) -> str:
        output = initial_response.get("output") or {}
        task_id = output.get("task_id") if isinstance(output, dict) else None
        if not task_id:
            urls = _collect_urls(initial_response, preferred_keys=preferred_keys)
            if urls:
                return urls[0]
            raise ApiError(
                f"异步响应缺少 task_id: {json.dumps(initial_response, ensure_ascii=False)[:1000]}",
                retryable=True,
                category="response_format",
            )

        started = time.monotonic()
        delay = 3.0
        task_endpoint = _endpoint(self.config.dashscope_base_url, f"tasks/{task_id}")
        while time.monotonic() - started < timeout_seconds:
            response = self.http.request_json(
                task_endpoint,
                method="GET",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                body=None,
                timeout=60,
            )
            output = response.get("output") or {}
            status = str(
                output.get("task_status") or response.get("task_status") or ""
            ).upper()
            if status == "SUCCEEDED":
                urls = _collect_urls(response, preferred_keys=preferred_keys)
                if not urls:
                    raise ApiError(
                        "任务成功但没有返回产物 URL",
                        retryable=True,
                        category="response_format",
                    )
                return urls[0]
            if status in {"FAILED", "CANCELED", "CANCELLED", "UNKNOWN"}:
                message = output.get("message") or response.get("message") or status
                category = _failure_category(None, str(message))
                raise ApiError(
                    f"异步任务失败: {message}",
                    retryable=category
                    in {"rate_limit", "queue", "timeout", "server", "network"},
                    category=category,
                )
            time.sleep(min(delay, self.http._remaining(delay)))
            delay = min(15.0, delay * 1.35)
        raise ApiError(
            f"异步任务轮询超时: {task_id}",
            retryable=True,
            category="timeout",
        )
