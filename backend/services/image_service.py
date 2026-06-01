from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import HTTPException
from raganything import RAGAnything

from backend.constants import (
    IMAGE_MEDIA_TYPES,
    INLINE_IMAGE_REF_PATTERN,
    MULTI_DOCUMENT_RELATED_IMAGE_LIMIT,
    MULTI_DOCUMENT_RELATED_IMAGE_PER_DOCUMENT_LIMIT,
    SINGLE_DOCUMENT_RELATED_IMAGE_LIMIT,
)
from backend.core.ids import safe_document_id
from backend.core.models import ImageAsset
from backend.core.paths import (
    content_list_path,
    document_path,
    is_relative_to_path,
    load_content_list,
    normalized_path_key,
)
from backend.core.text import first_non_empty_text
from backend.schemas import ImageAssetPublic

logger = logging.getLogger("api_server")

def build_image_id(
    document_id: str, image_path: Path, content_root: Path | None = None
) -> str:
    path_key = image_path.as_posix()
    if content_root is not None:
        try:
            path_key = image_path.resolve(strict=False).relative_to(
                content_root.resolve(strict=False)
            ).as_posix()
        except (OSError, RuntimeError, ValueError):
            path_key = image_path.name
    digest_input = f"{safe_document_id(document_id)}\0{path_key}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


def build_image_url(document_id: str, image_id: str) -> str:
    return f"/api/documents/{quote(safe_document_id(document_id), safe='')}/images/{image_id}"


def resolve_registered_image_path(raw_path: Any, content_root: Path) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    candidate = Path(raw_path.strip())
    if not candidate.is_absolute():
        candidate = content_root / candidate

    try:
        resolved = candidate.resolve(strict=False)
        root = content_root.resolve(strict=False)
    except (OSError, RuntimeError):
        return None

    if not is_relative_to_path(resolved, root):
        return None
    if resolved.suffix.lower() not in IMAGE_MEDIA_TYPES:
        return None
    return resolved


def page_from_content_item(item: dict[str, Any]) -> int | None:
    page_idx = item.get("page_idx")
    if isinstance(page_idx, int) and not isinstance(page_idx, bool) and page_idx >= 0:
        return page_idx + 1
    if isinstance(page_idx, str) and page_idx.isdigit():
        return int(page_idx) + 1

    page = item.get("page")
    if isinstance(page, int) and not isinstance(page, bool) and page > 0:
        return page
    if isinstance(page, str) and page.isdigit() and int(page) > 0:
        return int(page)
    return None


def content_item_search_text(item: dict[str, Any]) -> str:
    return first_non_empty_text(
        item,
        "text",
        "content",
        "caption",
        "image_caption",
        "img_caption",
        "table_caption",
        "table_body",
        "equation",
        "latex",
        "footnote",
        "image_footnote",
        "img_footnote",
    )


def nearby_image_context(items: list[dict[str, Any]], index: int) -> str:
    parts: list[str] = []
    for neighbor in items[max(0, index - 2) : min(len(items), index + 3)]:
        text = content_item_search_text(neighbor)
        if text:
            parts.append(text)
    return "\n".join(parts)


def image_asset_source_type(item_type: str) -> str | None:
    if item_type == "equation":
        return "equation_image"
    if item_type in {"image", "table"}:
        return item_type
    return None


def get_document_images(document_id: str) -> list[ImageAsset]:
    normalized_document_id = safe_document_id(document_id)
    pdf_path = document_path(normalized_document_id)
    path = content_list_path(pdf_path)
    if not path:
        return []

    try:
        items = load_content_list(path)
    except Exception as exc:
        logger.warning(
            "Failed to load image registry content_list document_id=%s path=%s error=%s",
            normalized_document_id,
            path,
            exc,
        )
        return []

    content_root = path.parent
    assets: list[ImageAsset] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        item_type = str(item.get("type", "")).strip().lower()
        if item_type not in {"image", "table", "equation"}:
            continue
        image_path = resolve_registered_image_path(item.get("img_path"), content_root)
        if image_path is None:
            logger.debug(
                "Skipping unsafe or unsupported image asset document_id=%s item_index=%s",
                normalized_document_id,
                index,
            )
            continue

        image_id = build_image_id(normalized_document_id, image_path, content_root)
        if image_id in seen_ids:
            continue
        seen_ids.add(image_id)

        caption = first_non_empty_text(
            item,
            "caption",
            "image_caption",
            "img_caption",
            "table_caption",
            "equation",
            "latex",
            "text",
        )
        footnote = first_non_empty_text(
            item,
            "footnote",
            "image_footnote",
            "img_footnote",
            "table_footnote",
        )
        context_text = nearby_image_context(items, index)
        assets.append(
            ImageAsset(
                image_id=image_id,
                document_id=normalized_document_id,
                document_name=normalized_document_id,
                filename=image_path.name,
                caption=caption or None,
                footnote=footnote or None,
                page=page_from_content_item(item),
                bbox=item.get("bbox") or item.get("layout_bbox"),
                absolute_path=image_path,
                url=build_image_url(normalized_document_id, image_id),
                content_index=index,
                context_text=context_text,
                source_type=image_asset_source_type(item_type),
            )
        )

    return assets


def get_document_image_asset(document_id: str, image_id: str) -> ImageAsset | None:
    if not re.fullmatch(r"[a-f0-9]{64}", image_id):
        return None
    for asset in get_document_images(document_id):
        if asset.image_id == image_id:
            return asset
    return None


def image_file_exists(asset: ImageAsset) -> bool:
    return (
        asset.absolute_path.suffix.lower() in IMAGE_MEDIA_TYPES
        and asset.absolute_path.exists()
        and asset.absolute_path.is_file()
    )


def public_image_asset(asset: ImageAsset, relevance_reason: str | None = None) -> ImageAssetPublic:
    return ImageAssetPublic(
        image_id=asset.image_id,
        document_id=asset.document_id,
        document_name=asset.document_name,
        url=asset.url,
        filename=asset.filename,
        caption=asset.caption,
        page=asset.page,
        bbox=asset.bbox,
        source_type=asset.source_type,
        relevance_reason=relevance_reason,
    )


def configure_vlm_image_registry(rag: RAGAnything, document_ids: list[str]) -> None:
    registry: dict[str, dict[str, Any]] = {}
    for document_id in document_ids:
        for asset in get_document_images(document_id):
            if not image_file_exists(asset):
                continue
            registry[normalized_path_key(asset.absolute_path)] = {
                "image_id": asset.image_id,
                "document_id": asset.document_id,
                "document_name": asset.document_name,
                "caption": asset.caption,
                "page": asset.page,
                "source_type": asset.source_type,
            }
    setattr(rag, "_image_assets_for_vlm", registry)


def extract_inline_image_refs(answer: str) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for match in INLINE_IMAGE_REF_PATTERN.finditer(answer):
        image_id = match.group(1).strip()
        if image_id and image_id not in seen:
            refs.append(image_id)
            seen.add(image_id)
    return refs


def public_images_for_documents(document_ids: list[str]) -> list[ImageAssetPublic]:
    images: list[ImageAssetPublic] = []
    seen: set[str] = set()
    for document_id in document_ids:
        for asset in get_document_images(document_id):
            if asset.image_id in seen or not image_file_exists(asset):
                continue
            images.append(public_image_asset(asset))
            seen.add(asset.image_id)
    return images


def image_aliases(image: ImageAssetPublic) -> list[str]:
    aliases = [image.url, image.image_id]
    if image.filename:
        aliases.insert(0, image.filename)
    return [alias for alias in aliases if alias]


def line_mentions_image_alias(line: str, image: ImageAssetPublic) -> bool:
    return any(alias in line for alias in image_aliases(image))


def replace_image_aliases_with_label(line: str, image: ImageAssetPublic) -> str:
    for alias in image_aliases(image):
        escaped_alias = re.escape(alias)
        line = re.sub(rf"`[^`]*{escaped_alias}[^`]*`", "下图", line)
        line = re.sub(rf"(?:[A-Za-z]:\\|/)[^\n`]*?{escaped_alias}", "下图", line)
        line = re.sub(
            rf"(?:[A-Za-z]:\\|/)[^\s`，。；;、)）\]]*{escaped_alias}",
            "下图",
            line,
        )
        line = line.replace(alias, "下图")
    line = re.sub(r"一张名为\s*下图\s*的图片", "下图", line)
    line = re.sub(r"名为\s*下图\s*的图片", "下图", line)
    line = re.sub(r"文件名为\s*下图\s*的图片", "下图", line)
    return line


def normalize_image_aliases_to_placeholders(
    answer: str, allowed_images: list[ImageAssetPublic]
) -> str:
    if not answer.strip() or not allowed_images:
        return answer

    used_refs = set(extract_inline_image_refs(answer))
    lines = answer.splitlines(keepends=True)
    normalized_lines: list[str] = []

    for line in lines:
        if "[[image:" in line:
            normalized_lines.append(line)
            continue

        line_refs: list[str] = []
        normalized_line = line
        for image in allowed_images:
            if image.image_id in used_refs:
                continue
            if not line_mentions_image_alias(normalized_line, image):
                continue
            normalized_line = replace_image_aliases_with_label(normalized_line, image)
            line_refs.append(image.image_id)
            used_refs.add(image.image_id)

        normalized_lines.append(normalized_line)
        for image_id in line_refs:
            normalized_lines.append(f"\n[[image:{image_id}]]\n")

    return "".join(normalized_lines)


def validate_inline_image_refs(
    answer: str, allowed_images: list[ImageAssetPublic]
) -> tuple[str, list[str], list[str]]:
    allowed_ids = {image.image_id for image in allowed_images}
    used_refs: list[str] = []
    seen: set[str] = set()
    warnings: list[str] = []

    def replace_ref(match: re.Match[str]) -> str:
        image_id = match.group(1).strip()
        if image_id not in allowed_ids:
            warnings.append(f"Removed unavailable inline image reference: {image_id}")
            return "[图片不可用]"
        if image_id in seen:
            warnings.append(f"Removed duplicate inline image reference: {image_id}")
            return ""
        seen.add(image_id)
        used_refs.append(image_id)
        return f"[[image:{image_id}]]"

    validated_answer = INLINE_IMAGE_REF_PATTERN.sub(replace_ref, answer)
    return validated_answer, used_refs, warnings


def related_images_for_inline_answer(
    allowed_images: list[ImageAssetPublic], inline_image_refs: list[str]
) -> list[ImageAssetPublic]:
    if not inline_image_refs:
        return []

    images_by_id = {image.image_id: image for image in allowed_images}
    return [
        images_by_id[image_id]
        for image_id in inline_image_refs
        if image_id in images_by_id
    ]


def validate_answer_images(
    answer: str,
    related_images: list[ImageAssetPublic],
    allowed_images: list[ImageAssetPublic],
) -> tuple[str, list[ImageAssetPublic], list[str]]:
    normalized_answer = normalize_image_aliases_to_placeholders(answer, allowed_images)
    validated_answer, inline_refs, warnings = validate_inline_image_refs(
        normalized_answer, allowed_images
    )
    if warnings:
        for warning in warnings:
            logger.warning("Inline image reference validation: %s", warning)
    if not inline_refs:
        return validated_answer, related_images, []
    return (
        validated_answer,
        related_images_for_inline_answer(allowed_images, inline_refs),
        inline_refs,
    )


def clear_vlm_image_tracking(rag: RAGAnything) -> None:
    setattr(rag, "_current_image_paths_for_vlm", [])
    setattr(rag, "_current_image_refs_for_vlm", [])


def current_vlm_image_paths(rag: RAGAnything) -> list[str]:
    raw_paths = getattr(rag, "_current_image_paths_for_vlm", [])
    if not isinstance(raw_paths, list):
        return []
    return [str(path) for path in raw_paths if str(path).strip()]


def query_keywords(question: str) -> set[str]:
    normalized = question.lower()
    keywords = {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) > 1
    }
    cjk_stop = {
        "什么",
        "如何",
        "说明",
        "解释",
        "文档",
        "中的",
        "相关",
        "内容",
        "过程",
        "请解",
    }
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", question):
        for size in (4, 3, 2):
            if len(run) < size:
                continue
            for start in range(0, len(run) - size + 1):
                term = run[start : start + size]
                if term not in cjk_stop:
                    keywords.add(term)
    return keywords


def score_image_asset(asset: ImageAsset, keywords: set[str]) -> int:
    if not keywords:
        return 0
    caption_text = " ".join(
        part for part in (asset.caption, asset.footnote) if part
    ).lower()
    context_text = asset.context_text.lower()
    score = 0
    for keyword in keywords:
        if keyword in caption_text:
            score += len(keyword) * 3
        elif keyword in context_text:
            score += len(keyword)
    return score


def related_images_from_vlm_paths(
    document_ids: list[str],
    image_paths: list[str],
) -> list[ImageAssetPublic]:
    if not image_paths:
        return []

    assets_by_path: dict[str, ImageAsset] = {}
    for document_id in document_ids:
        try:
            for asset in get_document_images(document_id):
                if image_file_exists(asset):
                    assets_by_path[normalized_path_key(asset.absolute_path)] = asset
        except HTTPException:
            continue

    total_limit = (
        MULTI_DOCUMENT_RELATED_IMAGE_LIMIT
        if len(document_ids) > 1
        else SINGLE_DOCUMENT_RELATED_IMAGE_LIMIT
    )
    per_document_limit = (
        MULTI_DOCUMENT_RELATED_IMAGE_PER_DOCUMENT_LIMIT
        if len(document_ids) > 1
        else SINGLE_DOCUMENT_RELATED_IMAGE_LIMIT
    )

    related_images: list[ImageAssetPublic] = []
    per_document_counts: dict[str, int] = {}
    seen: set[str] = set()
    for image_path in image_paths:
        asset = assets_by_path.get(normalized_path_key(image_path))
        if asset is None or asset.image_id in seen:
            continue
        if per_document_counts.get(asset.document_id, 0) >= per_document_limit:
            continue
        related_images.append(
            public_image_asset(asset, "Retrieved with VLM image context")
        )
        seen.add(asset.image_id)
        per_document_counts[asset.document_id] = (
            per_document_counts.get(asset.document_id, 0) + 1
        )
        if len(related_images) >= total_limit:
            break

    return related_images


def related_images_from_keywords(
    document_ids: list[str],
    question: str,
) -> list[ImageAssetPublic]:
    keywords = query_keywords(question)
    if not keywords:
        return []

    scored_assets: list[tuple[int, int, int, str, ImageAsset]] = []
    for document_id in document_ids:
        try:
            for asset in get_document_images(document_id):
                if not image_file_exists(asset):
                    continue
                score = score_image_asset(asset, keywords)
                if score <= 0:
                    continue
                page_sort = asset.page if asset.page is not None else 1_000_000
                scored_assets.append(
                    (score, -len(asset.caption or ""), -page_sort, asset.image_id, asset)
                )
        except HTTPException:
            continue

    scored_assets.sort(reverse=True)
    total_limit = (
        MULTI_DOCUMENT_RELATED_IMAGE_LIMIT
        if len(document_ids) > 1
        else SINGLE_DOCUMENT_RELATED_IMAGE_LIMIT
    )
    per_document_limit = (
        MULTI_DOCUMENT_RELATED_IMAGE_PER_DOCUMENT_LIMIT
        if len(document_ids) > 1
        else SINGLE_DOCUMENT_RELATED_IMAGE_LIMIT
    )

    related_images: list[ImageAssetPublic] = []
    per_document_counts: dict[str, int] = {}
    seen: set[str] = set()
    for _score, _caption_sort, _page_sort, _image_id, asset in scored_assets:
        if asset.image_id in seen:
            continue
        if per_document_counts.get(asset.document_id, 0) >= per_document_limit:
            continue
        related_images.append(
            public_image_asset(asset, "Matched question keywords in image context")
        )
        seen.add(asset.image_id)
        per_document_counts[asset.document_id] = (
            per_document_counts.get(asset.document_id, 0) + 1
        )
        if len(related_images) >= total_limit:
            break
    return related_images


def select_related_images(
    document_ids: list[str],
    question: str,
    vlm_image_paths: list[str],
) -> list[ImageAssetPublic]:
    related_images = related_images_from_vlm_paths(document_ids, vlm_image_paths)
    if related_images:
        return related_images
    return related_images_from_keywords(document_ids, question)
