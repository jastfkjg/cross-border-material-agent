"""Prompt path parsing and product source normalization."""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

from .models import ProductAttribute, ProductFacts, SizeConversion, Sku, SkuAttribute


class InputError(RuntimeError):
    """Raised when the evaluation input cannot be resolved safely."""


@dataclass(frozen=True, slots=True)
class ParsedPaths:
    input_dir: Path
    output_dir: Path


_PATH_RE = re.compile(r"(?P<path>/(?:[^\s`'\"，。；：、<>|]+/?)+)")
_QUOTED_PATH_RE = re.compile(r"[`'\"](?P<path>/[^`'\"]+)[`'\"]")


def _clean_candidate(value: str) -> str:
    return value.strip().rstrip(".,;:，。；：)）]】>")


def parse_prompt_paths(prompt: str) -> ParsedPaths:
    """Extract input and output paths from the natural-language evaluation prompt.

    The official prompt contains absolute Unix paths, but their exact wording and
    quoting may vary. Selection is based on the nearby semantic label first and
    on path names as a secondary signal.
    """

    candidates: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for pattern in (_QUOTED_PATH_RE, _PATH_RE):
        for match in pattern.finditer(prompt):
            value = _clean_candidate(match.group("path"))
            if value not in seen:
                candidates.append((value, match.start(), match.end()))
                seen.add(value)

    if not candidates:
        raise InputError("--prompt 中没有找到绝对输入/输出路径")

    def score(item: tuple[str, int, int], role: str) -> int:
        value, start, end = item
        window = prompt[max(0, start - 80) : min(len(prompt), end + 50)].lower()
        lowered = value.lower().rstrip("/")
        if role == "input":
            semantic = ("输入", "input", "数据", "dataset", "读取", "source")
            opposing = ("输出", "output", "保存", "result")
            basename = ("input", "dataset", "data")
        else:
            semantic = ("输出", "output", "保存", "result", "产出")
            opposing = ("输入", "input", "读取", "dataset")
            basename = ("output", "result", "out")
        value_score = (
            25
            if any(
                lowered.endswith(f"/{token}") or lowered.endswith(token)
                for token in basename
            )
            else 0
        )
        return (
            value_score
            + sum(12 for token in semantic if token in window)
            - sum(7 for token in opposing if token in window)
        )

    input_item = max(candidates, key=lambda item: score(item, "input"))
    remaining = [item for item in candidates if item[0] != input_item[0]]
    if not remaining:
        raise InputError("--prompt 中只解析到一个路径，无法区分输入和输出目录")
    output_item = max(remaining, key=lambda item: score(item, "output"))

    input_dir = Path(input_item[0]).expanduser().resolve()
    output_dir = Path(output_item[0]).expanduser().resolve()
    if input_dir.suffix.lower() in {".json", ".md", ".txt"}:
        input_dir = input_dir.parent
    if output_dir.suffix.lower() in {".json", ".md", ".txt"}:
        output_dir = output_dir.parent
    if input_dir == output_dir:
        raise InputError("输入目录与输出目录不能相同")
    return ParsedPaths(input_dir=input_dir, output_dir=output_dir)


def _nested_product(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    current: Any = data
    for key in ("ret", "result", "result"):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, dict) and ("offerId" in current or "subject" in current):
        return current
    return None


def _classify_json(data: Any) -> str:
    if _nested_product(data) is not None:
        return "product"
    if not isinstance(data, dict) or not isinstance(data.get("categories"), list):
        return "unknown"
    if "rootCategory" in data or "source" in data:
        return "categories"
    if "generatedAt" in data or "stats" in data:
        return "attributes"
    first = data["categories"][0] if data["categories"] else {}
    if isinstance(first, dict) and "categoryMetadata" in first:
        return "attributes"
    return "categories"


def discover_input_files(
    input_dir: Path, product_id: str = ""
) -> tuple[Path, Path, Path]:
    if not input_dir.is_dir():
        raise InputError(f"输入目录不存在或不可读: {input_dir}")

    classified: dict[str, list[Path]] = {
        "product": [],
        "categories": [],
        "attributes": [],
    }
    for path in sorted(input_dir.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                kind = _classify_json(json.load(handle))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if kind in classified:
            classified[kind].append(path)

    products = classified["product"]
    if product_id:
        products = [
            path
            for path in products
            if product_id in path.name or _file_offer_id(path) == product_id
        ]
    if len(products) != 1:
        raise InputError(f"期望恰好一个商品 JSON，实际找到 {len(products)} 个")
    if not classified["categories"]:
        raise InputError("未找到平台类目 JSON")
    if not classified["attributes"]:
        raise InputError("未找到平台属性 JSON")
    return products[0], classified["categories"][0], classified["attributes"][0]


def _file_offer_id(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as handle:
            product = _nested_product(json.load(handle)) or {}
        return str(product.get("offerId", ""))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ""


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(f"无法读取 JSON {path}: {exc}") from exc


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _deduplicate(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _normalize_url(url: str) -> str:
    """Keep signed query parameters intact while stripping HTML escaping."""

    cleaned = html.unescape(url).replace("\\/", "/").strip()
    parts = urlsplit(cleaned)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return ""
    return urlunsplit(parts)


def _description_urls(description: str) -> list[str]:
    urls = re.findall(r"https?://[^\s'\"<>]+", html.unescape(description or ""))
    return _deduplicate(filter(None, (_normalize_url(url) for url in urls)))


_JIN_RANGE_RE = re.compile(
    r"(?P<low>\d+(?:\.\d+)?)\s*[-~～至]\s*(?P<high>\d+(?:\.\d+)?)\s*斤"
)


def _decimal_string(value: Decimal, places: str) -> str:
    quantized = value.quantize(Decimal(places), rounding=ROUND_HALF_UP)
    return format(quantized.normalize(), "f")


def _size_conversions(attributes: list[ProductAttribute]) -> list[SizeConversion]:
    result: list[SizeConversion] = []
    for item in attributes:
        if "尺码" not in item.name and "身高" not in item.name:
            continue
        match = _JIN_RANGE_RE.search(item.value)
        if not match:
            continue
        low_jin = Decimal(match.group("low"))
        high_jin = Decimal(match.group("high"))
        low_kg, high_kg = low_jin / 2, high_jin / 2
        kg = f"{_decimal_string(low_kg, '0.1')}–{_decimal_string(high_kg, '0.1')} kg"
        lb = f"{_decimal_string(low_kg * Decimal('2.2046226218'), '0.1')}–{_decimal_string(high_kg * Decimal('2.2046226218'), '0.1')} lb"
        result.append(
            SizeConversion(
                source_label=item.value,
                kilograms=kg,
                pounds=lb,
                evidence_pointer=item.evidence_pointer,
            )
        )
    return result


def load_product_facts(product_path: Path) -> ProductFacts:
    raw_bytes = product_path.read_bytes()
    raw = json.loads(raw_bytes.decode("utf-8"))
    product = _nested_product(raw)
    if product is None:
        raise InputError(f"商品 JSON 结构不符合预期: {product_path}")

    attributes: list[ProductAttribute] = []
    for index, item in enumerate(product.get("productAttribute") or []):
        if not isinstance(item, dict):
            continue
        value = _safe_text(item.get("valueTrans") or item.get("value"))
        name = _safe_text(item.get("attributeNameTrans") or item.get("attributeName"))
        if not name or not value:
            continue
        attributes.append(
            ProductAttribute(
                attribute_id=_safe_text(item.get("attributeId")),
                name=name,
                value=value,
                value_translated=_safe_text(item.get("valueTrans")),
                evidence_pointer=f"/ret/result/result/productAttribute/{index}",
            )
        )

    skus: list[Sku] = []
    sku_images: list[str] = []
    for sku_index, sku_raw in enumerate(product.get("productSkuInfos") or []):
        if not isinstance(sku_raw, dict):
            continue
        sku_attrs: list[SkuAttribute] = []
        for item in sku_raw.get("skuAttributes") or []:
            if not isinstance(item, dict):
                continue
            image_url = _normalize_url(_safe_text(item.get("skuImageUrl")))
            if image_url:
                sku_images.append(image_url)
            sku_attrs.append(
                SkuAttribute(
                    attribute_id=_safe_text(item.get("attributeId")),
                    name=_safe_text(
                        item.get("attributeNameTrans") or item.get("attributeName")
                    ),
                    value=_safe_text(item.get("valueTrans") or item.get("value")),
                    image_url=image_url,
                )
            )
        skus.append(
            Sku(
                sku_id=_safe_text(sku_raw.get("skuId")) or f"sku-{sku_index + 1}",
                spec_id=_safe_text(sku_raw.get("specId")),
                attributes=sku_attrs,
            )
        )

    image_container = product.get("productImage") or {}
    product_images_raw = (
        image_container.get("images") if isinstance(image_container, dict) else []
    )
    product_images = _deduplicate(
        filter(
            None,
            (_normalize_url(_safe_text(url)) for url in (product_images_raw or [])),
        )
    )
    description_images = _description_urls(_safe_text(product.get("description")))
    fingerprint = hashlib.sha256(raw_bytes).hexdigest()

    facts = ProductFacts(
        platform=_safe_text(product.get("platform")),
        source_url=_safe_text(product.get("url")),
        offer_id=_safe_text(product.get("offerId")),
        source_title=_safe_text(product.get("subjectTrans") or product.get("subject")),
        source_category_id=_safe_text(product.get("categoryId")),
        source_category_name=_safe_text(product.get("category_name")),
        attributes=attributes,
        skus=skus,
        product_image_urls=product_images,
        sku_image_urls=_deduplicate(sku_images),
        description_image_urls=description_images,
        size_conversions=[],
        input_file=str(product_path),
        fingerprint=fingerprint,
    )
    facts.size_conversions = _size_conversions(attributes)
    if not facts.offer_id or not facts.source_title:
        raise InputError("商品 JSON 缺少 offerId 或 subject")
    if not facts.all_image_urls():
        raise InputError("商品 JSON 未提供任何可用图片 URL")
    return facts
