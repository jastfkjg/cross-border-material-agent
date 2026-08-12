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


class ApiError(RuntimeError):
    """A retryable or terminal model service failure."""


@dataclass(frozen=True, slots=True)
class ApiConfig:
    api_key: str
    dashscope_base_url: str
    openai_base_url: str
    chat_model: str = "qwen3.8-max"
    chat_fallback_model: str = "qwen3.7-plus"
    image_model: str = "qwen-image-3.0-pro"
    image_fallback_model: str = "wan2.7-image"
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
            image_model=os.environ.get("AGENT_IMAGE_MODEL", "qwen-image-3.0-pro"),
            image_fallback_model=os.environ.get(
                "AGENT_IMAGE_FALLBACK_MODEL", "wan2.7-image"
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
    def __init__(self, logger: logging.Logger, deadline: float):
        self.logger = logger
        self.deadline = deadline
        self.ssl_context = ssl.create_default_context()

    def _remaining(self, maximum: float) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ApiError("全局运行截止时间已到")
        return max(1.0, min(maximum, remaining))

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
            request = Request(url, data=payload, headers=request_headers, method=method)
            try:
                with urlopen(
                    request, timeout=self._remaining(timeout), context=self.ssl_context
                ) as response:
                    data = response.read()
                parsed = json.loads(data.decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise ApiError("API 返回值不是 JSON 对象")
                return parsed
            except HTTPError as exc:
                detail = exc.read(8192).decode("utf-8", errors="replace")
                last_error = f"HTTP {exc.code}: {detail}"
                retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
                if not retryable or attempt + 1 >= attempts:
                    raise ApiError(last_error) from exc
                retry_after = exc.headers.get("Retry-After", "")
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = min(18.0, (2**attempt) + random.random() * 1.5)
            except (
                URLError,
                TimeoutError,
                OSError,
                json.JSONDecodeError,
                UnicodeError,
            ) as exc:
                last_error = str(exc)
                if attempt + 1 >= attempts:
                    raise ApiError(last_error) from exc
                delay = min(18.0, (2**attempt) + random.random() * 1.5)
            self.logger.warning(
                "API 请求失败，%.1f 秒后重试: %s", delay, last_error[:500]
            )
            time.sleep(min(delay, self._remaining(delay)))
        raise ApiError(last_error or "API 请求失败")

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
                    url, headers={"User-Agent": "crossborder-material-agent/1.0"}
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

    def __init__(self, config: ApiConfig, logger: logging.Logger, deadline: float):
        self.config = config
        self.logger = logger
        self.http = HttpJsonClient(logger, deadline)
        self._chat_slots = threading.BoundedSemaphore(3)
        self._image_slots = threading.BoundedSemaphore(2)
        self._video_slots = threading.BoundedSemaphore(1)

    @property
    def model_summary(self) -> dict[str, str]:
        return {
            "chat": self.config.chat_model,
            "image": self.config.image_model,
            "image_fallback": self.config.image_fallback_model,
            "video": self.config.video_model,
        }

    def _chat_response(self, body: dict[str, Any]) -> str:
        endpoint = _endpoint(self.config.openai_base_url, "chat/completions")
        response = self.http.request_json(
            endpoint,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            body=body,
            timeout=180,
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ApiError(
                f"Chat API 返回结构异常: {json.dumps(response, ensure_ascii=False)[:1000]}"
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
        model: str = "",
    ) -> dict[str, Any]:
        user_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for url in images:
            if url:
                user_content.append({"type": "image_url", "image_url": {"url": url}})
        selected_model = model or self.config.chat_model
        body = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "enable_thinking": False,
        }
        with self._chat_slots:
            try:
                text = self._chat_response(body)
            except ApiError:
                if selected_model == self.config.chat_fallback_model:
                    raise
                body["model"] = self.config.chat_fallback_model
                text = self._chat_response(body)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ApiError(f"模型未返回合法 JSON: {text[:1000]}") from exc
        if not isinstance(parsed, dict):
            raise ApiError("模型 JSON 顶层不是对象")
        return parsed

    def analyze_product_images(
        self, facts_json: str, image_urls: list[str]
    ) -> dict[str, Any]:
        system = (
            "You are a conservative e-commerce visual inspector. Return JSON only. "
            "Never infer fabric composition, performance, measurements, brand authorization, or care instructions "
            "from appearance. Separate direct observations from uncertain impressions."
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

Verified source facts:
{facts_json}
""".strip()
        return self.chat_json(system, prompt, images=image_urls[:12])

    def review_generated_images(
        self, facts_json: str, image_urls: list[str]
    ) -> dict[str, Any]:
        system = (
            "You are a strict product-listing image quality gate. Return JSON only. "
            "Reject product identity drift, changed colors or construction, invented text or claims, watermarks, "
            "visual artifacts, distorted anatomy, unreadable layouts, and images where the product is obscured."
        )
        prompt = f"""
Review each generated image in the supplied order against the verified facts.
Return JSON with key assets, an array of exactly {len(image_urls)} objects. Each object must contain:
- index: zero-based integer
- usable: boolean
- identity_consistent: boolean
- unwanted_text: boolean
- major_artifacts: boolean
- reason: concise string

Verified facts:
{facts_json}
""".strip()
        return self.chat_json(system, prompt, images=image_urls)

    def generate_image(
        self, prompt: str, reference_urls: list[str], *, size: str
    ) -> tuple[str, str]:
        errors: list[str] = []
        with self._image_slots:
            try:
                return self._generate_qwen_image(
                    prompt, reference_urls, size=size
                ), self.config.image_model
            except ApiError as exc:
                errors.append(f"{self.config.image_model}: {exc}")
                self.logger.warning("主图像模型失败，切换回退模型: %s", exc)
            try:
                return self._generate_wan_image(
                    prompt, reference_urls, size=size
                ), self.config.image_fallback_model
            except ApiError as exc:
                errors.append(f"{self.config.image_fallback_model}: {exc}")
        raise ApiError("; ".join(errors))

    def _generate_qwen_image(
        self, prompt: str, reference_urls: list[str], *, size: str
    ) -> str:
        content: list[dict[str, str]] = [{"text": prompt}]
        content.extend({"image": url} for url in reference_urls[:3])
        body = {
            "model": self.config.image_model,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": {"size": size, "prompt_extend": True, "watermark": False},
        }
        response = self.http.request_json(
            _endpoint(
                self.config.dashscope_base_url,
                "services/aigc/multimodal-generation/generation",
            ),
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            body=body,
            timeout=240,
        )
        urls = _collect_urls(response, preferred_keys=("image", "url"))
        if not urls:
            raise ApiError(
                f"图像响应没有 URL: {json.dumps(response, ensure_ascii=False)[:1000]}"
            )
        return urls[0]

    def _generate_wan_image(
        self, prompt: str, reference_urls: list[str], *, size: str
    ) -> str:
        content: list[dict[str, str]] = [{"text": prompt}]
        content.extend({"image": url} for url in reference_urls[:9])
        body = {
            "model": self.config.image_fallback_model,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": {
                "n": 1,
                "size": size,
                "watermark": False,
                "thinking_mode": True,
            },
        }
        response = self.http.request_json(
            _endpoint(
                self.config.dashscope_base_url,
                "services/aigc/image-generation/generation",
            ),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "X-DashScope-Async": "enable",
            },
            body=body,
            timeout=180,
        )
        return self._poll_task_for_url(
            response, preferred_keys=("image", "url"), timeout_seconds=480
        )

    def generate_video(self, prompt: str, first_frame_url: str) -> tuple[str, str]:
        body = {
            "model": self.config.video_model,
            "input": {
                "prompt": prompt,
                "media": [{"type": "first_frame", "url": first_frame_url}],
            },
            "parameters": {
                "resolution": "720P",
                "ratio": "16:9",
                "duration": 8,
                "prompt_extend": False,
                "watermark": False,
            },
        }
        with self._video_slots:
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
                response, preferred_keys=("video", "url"), timeout_seconds=720
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
                f"异步响应缺少 task_id: {json.dumps(initial_response, ensure_ascii=False)[:1000]}"
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
                    raise ApiError("任务成功但没有返回产物 URL")
                return urls[0]
            if status in {"FAILED", "CANCELED", "CANCELLED", "UNKNOWN"}:
                message = output.get("message") or response.get("message") or status
                raise ApiError(f"异步任务失败: {message}")
            time.sleep(min(delay, self.http._remaining(delay)))
            delay = min(15.0, delay * 1.35)
        raise ApiError(f"异步任务轮询超时: {task_id}")
