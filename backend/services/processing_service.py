from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from raganything import RAGAnything
from raganything.utils import insert_text_content, separate_content

from backend.config import (
    OUTPUT_DIR,
    formula_processing_enabled,
    generic_processing_enabled,
    mineru_runtime_kwargs,
    multimodal_enabled,
)
from backend.constants import MULTIMODAL_INDEXED_MESSAGE, TEXT_ONLY_INDEXED_MESSAGE
from backend.core.ids import document_cache_key, safe_document_id, safe_document_key
from backend.core.paths import document_path, document_storage_dir, lightrag_workspace
from backend.core.runtime import (
    _global_process_lock,
    _rag_instances,
    _rag_lock,
    document_lock,
    evict_rag_instance,
    mark_document_processing,
    unmark_document_processing,
)
from backend.core.text import first_non_empty_text
from backend.rag.factory import build_fresh_document_rag
from backend.rag.readiness import document_storage_readiness
from backend.rag.storage import (
    clear_process_failure,
    ensure_document_storage_reset,
    persist_lightrag_storages,
    set_lightrag_default_workspace,
    write_process_failure,
)
from backend.schemas import ProcessResponse
from backend.services.citation_service import build_source_map
from backend.services.document_service import (
    knowledge_base_not_ready_response,
    summarize_document,
)

logger = logging.getLogger("api_server")

def canonical_multimodal_type(item: dict[str, Any]) -> str:
    raw_type = str(item.get("type", "unknown")).strip().lower()
    if raw_type in {"image", "table", "equation"}:
        return raw_type
    if raw_type in {
        "formula",
        "inline_formula",
        "interline_formula",
        "inline_equation",
        "interline_equation",
    }:
        return "formula"
    return "generic"


def new_multimodal_summary(multimodal_enabled_value: bool) -> dict[str, Any]:
    return {
        "text_indexed": False,
        "multimodal_enabled": multimodal_enabled_value,
        "multimodal_status": "disabled" if not multimodal_enabled_value else "skipped",
        "image_status": "disabled",
        "table_status": "disabled",
        "equation_status": "disabled",
        "formula_status": "disabled",
        "skipped_multimodal_items_count": 0,
        "skipped_by_reason": {
            "disabled_skip": 0,
            "noise_skip": 0,
            "invalid_skip": 0,
            "limit_skip": 0,
        },
        "multimodal_warnings_count": 0,
        "warnings_summary": [],
    }


def add_multimodal_warning(summary: dict[str, Any], message: str) -> None:
    logger.warning(message)
    summary["multimodal_warnings_count"] += 1
    warnings_summary = summary["warnings_summary"]
    if len(warnings_summary) < 20:
        warnings_summary.append(message)


def status_from_counts(
    *,
    enabled: bool,
    processed: int,
    failed: int,
    skipped: int,
    limit_skipped: int,
    attempted: int,
) -> str:
    if not enabled:
        return "disabled"
    if attempted == 0:
        return "skipped"
    if failed == 0 and skipped == 0:
        return "processed_with_limits" if limit_skipped > 0 else "processed"
    if processed > 0:
        return "partial_success_with_limits" if limit_skipped > 0 else "partial_success"
    if failed > 0:
        return "failed"
    return "skipped"


def new_multimodal_stats() -> dict[str, dict[str, int]]:
    return {
        "image": {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "limit_skipped": 0,
            "attempted": 0,
        },
        "table": {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "limit_skipped": 0,
            "attempted": 0,
        },
        "equation": {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "limit_skipped": 0,
            "attempted": 0,
        },
        "formula": {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "limit_skipped": 0,
            "attempted": 0,
        },
        "generic": {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "limit_skipped": 0,
            "attempted": 0,
        },
    }


def finalize_multimodal_summary(
    summary: dict[str, Any], stats: dict[str, dict[str, int]], rag: RAGAnything
) -> dict[str, Any]:
    summary["image_status"] = status_from_counts(
        enabled=rag.config.enable_image_processing, **stats["image"]
    )
    summary["table_status"] = status_from_counts(
        enabled=rag.config.enable_table_processing, **stats["table"]
    )
    summary["equation_status"] = status_from_counts(
        enabled=rag.config.enable_equation_processing, **stats["equation"]
    )
    summary["formula_status"] = status_from_counts(
        enabled=formula_processing_enabled()
        and (rag.config.enable_equation_processing or rag.config.enable_image_processing),
        **stats["formula"],
    )
    summary["generic_status"] = status_from_counts(
        enabled=generic_processing_enabled(),
        **stats["generic"],
    )

    attempted_total = sum(type_stats["attempted"] for type_stats in stats.values())
    failed_total = sum(type_stats["failed"] for type_stats in stats.values())
    limit_skipped_total = sum(
        type_stats["limit_skipped"] for type_stats in stats.values()
    )
    non_limit_warning_count = max(
        summary["multimodal_warnings_count"] - limit_skipped_total, 0
    )

    if attempted_total == 0:
        summary["multimodal_status"] = (
            "partial_success"
            if non_limit_warning_count > 0
            else "skipped"
        )
    elif failed_total > 0 or non_limit_warning_count > 0:
        summary["multimodal_status"] = (
            "partial_success_with_limits"
            if limit_skipped_total > 0
            else "partial_success"
        )
    elif limit_skipped_total > 0:
        summary["multimodal_status"] = "processed_with_limits"
    else:
        summary["multimodal_status"] = "processed"

    return summary


def skip_multimodal_item(
    summary: dict[str, Any], reason: str, warning: str | None = None
) -> None:
    summary["skipped_multimodal_items_count"] += 1
    summary["skipped_by_reason"][reason] += 1
    if warning:
        add_multimodal_warning(summary, warning)


def is_noise_multimodal_type(item: dict[str, Any]) -> bool:
    raw_type = str(item.get("type", "unknown")).strip().lower()
    return raw_type in {
        "discarded",
        "unknown",
        "header",
        "footer",
        "page_header",
        "page_footer",
        "page_number",
        "page_footnote",
        "list",
    }


def multimodal_item_limit(canonical_type: str) -> int | None:
    env_names = {
        "image": "MAX_IMAGE_ITEMS",
        "table": "MAX_TABLE_ITEMS",
        "equation": "MAX_EQUATION_ITEMS",
        "formula": "MAX_FORMULA_ITEMS",
        "generic": "MAX_GENERIC_ITEMS",
    }
    env_name = env_names.get(canonical_type)
    if not env_name:
        return None

    raw = os.getenv(env_name)
    if raw is None:
        return None

    normalized = raw.strip().lower()
    if normalized in {"", "0", "-1", "none", "unlimited"}:
        return None

    try:
        limit = int(normalized)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using unlimited multimodal item processing.",
            env_name,
            raw,
        )
        return None
    return limit if limit > 0 else None


def item_disabled_by_config(
    item: dict[str, Any], canonical_type: str, rag: RAGAnything
) -> bool:
    if canonical_type == "image":
        return not rag.config.enable_image_processing
    if canonical_type == "table":
        return not rag.config.enable_table_processing
    if canonical_type == "equation":
        return not rag.config.enable_equation_processing
    if canonical_type == "formula":
        if not formula_processing_enabled():
            return True
        formula_text = first_non_empty_text(item, "text", "latex", "content", "equation")
        if formula_text:
            return not rag.config.enable_equation_processing
        if item.get("img_path"):
            return not rag.config.enable_image_processing
        return False
    if canonical_type == "generic":
        return not generic_processing_enabled()
    return True


def normalize_multimodal_item(
    item: dict[str, Any],
    canonical_type: str,
    rag: RAGAnything,
    summary: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    normalized = dict(item)
    raw_type = str(item.get("type", "unknown"))
    normalized["_original_type"] = raw_type

    if canonical_type == "image":
        image_path = normalized.get("img_path")
        if not image_path:
            add_multimodal_warning(
                summary, f"Skipping image item without img_path: {item}"
            )
            return None
        if not Path(str(image_path)).exists():
            add_multimodal_warning(
                summary, f"Skipping image item with missing file: {image_path}"
            )
            return None
        if "image" not in rag.modal_processors:
            add_multimodal_warning(
                summary, "Skipping image item because image processor is disabled."
            )
            return None
        normalized["type"] = "image"
        return "image", normalized

    if canonical_type == "table":
        table_body = first_non_empty_text(normalized, "table_body")
        if not table_body:
            add_multimodal_warning(
                summary, f"Skipping table item without table_body: {item}"
            )
            return None
        if "table" not in rag.modal_processors:
            add_multimodal_warning(
                summary, "Skipping table item because table processor is disabled."
            )
            return None
        normalized["type"] = "table"
        normalized["table_body"] = table_body
        return "table", normalized

    if canonical_type in {"equation", "formula"}:
        equation_text = first_non_empty_text(
            normalized, "text", "latex", "content", "equation"
        )
        if equation_text:
            if "equation" not in rag.modal_processors:
                add_multimodal_warning(
                    summary,
                    f"Skipping {canonical_type} item because equation processor is disabled.",
                )
                return None
            normalized["type"] = "equation"
            normalized["text"] = equation_text
            if not normalized.get("text_format"):
                normalized["text_format"] = "latex" if normalized.get("latex") else "text"
            return "equation", normalized

        if canonical_type == "formula" and normalized.get("img_path"):
            image_path = normalized.get("img_path")
            if not Path(str(image_path)).exists():
                add_multimodal_warning(
                    summary, f"Skipping formula image item with missing file: {image_path}"
                )
                return None
            if "image" not in rag.modal_processors:
                add_multimodal_warning(
                    summary,
                    "Skipping formula image item because image processor is disabled.",
                )
                return None
            normalized["type"] = "image"
            normalized.setdefault("image_caption", ["Formula image"])
            return "image", normalized

        add_multimodal_warning(
            summary, f"Skipping {canonical_type} item without text/latex/content: {item}"
        )
        return None

    if canonical_type == "generic":
        if not generic_processing_enabled():
            return None
        if "generic" not in rag.modal_processors:
            add_multimodal_warning(
                summary, "Skipping generic item because generic processor is unavailable."
            )
            return None
        normalized["type"] = "generic"
        return "generic", normalized

    return None


async def get_doc_chunk_ids(rag: RAGAnything, doc_id: str) -> set[str] | None:
    try:
        doc_status = await rag.lightrag.doc_status.get_by_id(doc_id)
        if not doc_status:
            return set()
        return set(doc_status.get("chunks_list", []))
    except Exception as exc:
        logger.warning("Could not read doc_status chunks_list for %s: %s", doc_id, exc)
        return None


async def process_multimodal_item_with_tracking(
    rag: RAGAnything,
    normalized_item: dict[str, Any],
    processor_type: str,
    canonical_type: str,
    pdf_path: Path,
    doc_id: str,
    index: int,
    summary: dict[str, Any],
) -> bool:
    processor = rag.modal_processors.get(processor_type)
    if processor is None:
        add_multimodal_warning(
            summary, f"{canonical_type} item {index} has no {processor_type} processor."
        )
        return False

    original_caption_func = getattr(processor, "modal_caption_func", None)
    model_status = {"called": False, "empty": False, "exception": None}

    async def tracked_caption_func(*args, **kwargs):
        model_status["called"] = True
        try:
            response = await original_caption_func(*args, **kwargs)
        except Exception as exc:
            model_status["exception"] = exc
            raise
        if not str(response or "").strip():
            model_status["empty"] = True
        return response

    before_chunk_ids = await get_doc_chunk_ids(rag, doc_id)
    if original_caption_func is not None:
        processor.modal_caption_func = tracked_caption_func
    try:
        await rag._process_multimodal_content_individual(
            [normalized_item], str(pdf_path), doc_id
        )
    finally:
        if original_caption_func is not None:
            processor.modal_caption_func = original_caption_func

    after_chunk_ids = await get_doc_chunk_ids(rag, doc_id)
    chunk_added = (
        before_chunk_ids is None
        or after_chunk_ids is None
        or bool(after_chunk_ids - before_chunk_ids)
    )
    if not chunk_added:
        add_multimodal_warning(
            summary, f"{canonical_type} item {index} produced no multimodal chunk."
        )
        return False
    if model_status["exception"] is not None:
        add_multimodal_warning(
            summary,
            f"{canonical_type} item {index} model call failed and fallback was used: {model_status['exception']}",
        )
        return False
    if model_status["empty"]:
        add_multimodal_warning(
            summary,
            f"{canonical_type} item {index} model call returned empty content and fallback was used.",
        )
        return False
    if not model_status["called"]:
        add_multimodal_warning(
            summary, f"{canonical_type} item {index} did not call its model function."
        )
        return False
    return True


async def process_document_text_only(rag: RAGAnything, pdf_path: Path) -> None:
    content_list, doc_id = await rag.parse_document(
        file_path=str(pdf_path),
        output_dir=str(OUTPUT_DIR),
        parse_method=os.getenv("PARSE_METHOD", "auto"),
        display_stats=rag.config.display_content_stats,
        **mineru_runtime_kwargs(),
    )
    text_content, _multimodal_items = separate_content(content_list)
    if not text_content.strip():
        raise HTTPException(
            status_code=422,
            detail="No text content was extracted from the document.",
        )

    await insert_text_content(
        rag.lightrag,
        input=text_content,
        file_paths=rag._get_file_reference(str(pdf_path)),
        ids=doc_id,
    )


async def process_document_multimodal_safe(
    rag: RAGAnything, pdf_path: Path
) -> dict[str, Any]:
    summary = new_multimodal_summary(True)
    content_list, doc_id = await rag.parse_document(
        file_path=str(pdf_path),
        output_dir=str(OUTPUT_DIR),
        parse_method=os.getenv("PARSE_METHOD", "auto"),
        display_stats=rag.config.display_content_stats,
        **mineru_runtime_kwargs(),
    )
    text_content, multimodal_items = separate_content(content_list)
    if not text_content.strip():
        raise HTTPException(
            status_code=422,
            detail="No text content was extracted from the document.",
        )

    await insert_text_content(
        rag.lightrag,
        input=text_content,
        file_paths=rag._get_file_reference(str(pdf_path)),
        ids=doc_id,
    )
    summary["text_indexed"] = True
    stats = new_multimodal_stats()

    if not multimodal_items:
        return finalize_multimodal_summary(summary, stats, rag)

    if hasattr(rag, "set_content_source_for_context"):
        try:
            rag.set_content_source_for_context(content_list, rag.config.content_format)
        except Exception as exc:
            add_multimodal_warning(
                summary, f"Failed to set multimodal context source: {exc}"
            )

    for index, item in enumerate(multimodal_items):
        canonical_type = canonical_multimodal_type(item)
        tracked_type = canonical_type if canonical_type in stats else "generic"
        if is_noise_multimodal_type(item):
            stats[tracked_type]["skipped"] += 1
            skip_multimodal_item(summary, "noise_skip")
            continue
        if item_disabled_by_config(item, canonical_type, rag):
            stats[tracked_type]["skipped"] += 1
            skip_multimodal_item(summary, "disabled_skip")
            continue

        normalized = normalize_multimodal_item(item, canonical_type, rag, summary)
        if normalized is None:
            stats[tracked_type]["skipped"] += 1
            skip_multimodal_item(summary, "invalid_skip")
            continue

        processor_type, normalized_item = normalized
        item_limit = multimodal_item_limit(tracked_type)
        if item_limit is not None and stats[tracked_type]["attempted"] >= item_limit:
            stats[tracked_type]["limit_skipped"] += 1
            skip_multimodal_item(
                summary,
                "limit_skip",
                f"Skipping {canonical_type} item {index}: {tracked_type} limit {item_limit} reached.",
            )
            continue

        stats[tracked_type]["attempted"] += 1
        try:
            item_succeeded = await process_multimodal_item_with_tracking(
                rag,
                normalized_item,
                processor_type,
                canonical_type,
                pdf_path,
                doc_id,
                index,
                summary,
            )
            if item_succeeded:
                stats[tracked_type]["processed"] += 1
            else:
                stats[tracked_type]["failed"] += 1
        except Exception as exc:
            stats[tracked_type]["failed"] += 1
            add_multimodal_warning(
                summary,
                f"{canonical_type} item {index} failed in {processor_type} processor: {exc}",
            )

    return finalize_multimodal_summary(summary, stats, rag)


async def process_document(document_id: str) -> ProcessResponse | JSONResponse:
    normalized_document_id = safe_document_id(document_id)
    pdf_path = document_path(normalized_document_id)
    marked = await mark_document_processing(normalized_document_id)
    if not marked:
        return knowledge_base_not_ready_response(
            error="document_processing",
            message="当前文档正在索引，请稍后再试。",
            document_status="indexing",
        )

    lock = await document_lock(normalized_document_id)
    async with lock:
        logger.info(
            "process_wait_global_lock document_id=%s safe_document_key=%s",
            normalized_document_id,
            safe_document_key(normalized_document_id),
        )
        await _global_process_lock.acquire()
        logger.info(
            "process_acquired_global_lock document_id=%s safe_document_key=%s",
            normalized_document_id,
            safe_document_key(normalized_document_id),
        )
        storage_dir = document_storage_dir(normalized_document_id)
        cache_key = document_cache_key(normalized_document_id)
        keep_rag_instance = False
        rag: RAGAnything | None = None
        try:
            logger.info(
                "Starting document process document_id=%s safe_document_key=%s storage_dir=%s",
                normalized_document_id,
                safe_document_key(normalized_document_id),
                storage_dir,
            )
            await evict_rag_instance(normalized_document_id)
            storage_dir = ensure_document_storage_reset(normalized_document_id)
            clear_process_failure(normalized_document_id)
            set_lightrag_default_workspace(lightrag_workspace(normalized_document_id))

            rag = build_fresh_document_rag(normalized_document_id, storage_dir)
            init_result = await rag._ensure_lightrag_initialized()
            if not init_result.get("success"):
                raise HTTPException(
                    status_code=500,
                    detail=init_result.get("error", "LightRAG initialization failed"),
                )
            logger.info(
                "process_rag_initialized document_id=%s safe_document_key=%s working_dir=%s rag_id=%s lightrag_id=%s lightrag_workspace=%s",
                normalized_document_id,
                safe_document_key(normalized_document_id),
                storage_dir,
                id(rag),
                id(rag.lightrag) if rag.lightrag is not None else None,
                getattr(rag.lightrag, "workspace", None) if rag.lightrag else None,
            )

            if multimodal_enabled():
                process_summary = await process_document_multimodal_safe(rag, pdf_path)
                message = MULTIMODAL_INDEXED_MESSAGE
            else:
                await process_document_text_only(rag, pdf_path)
                process_summary = new_multimodal_summary(False)
                process_summary["text_indexed"] = True
                message = TEXT_ONLY_INDEXED_MESSAGE

            await persist_lightrag_storages(rag)
            try:
                build_source_map(normalized_document_id)
            except Exception as exc:
                logger.warning(
                    "Source map generation failed after processing document_id=%s safe_document_key=%s error=%s",
                    normalized_document_id,
                    safe_document_key(normalized_document_id),
                    exc,
                )
            readiness = document_storage_readiness(normalized_document_id)
            if readiness.ready:
                logger.info(
                    "Document process ready_for_chat document_id=%s safe_document_key=%s storage_dir=%s chunks_count=%s",
                    normalized_document_id,
                    safe_document_key(normalized_document_id),
                    storage_dir,
                    readiness.chunks_count,
                )
            else:
                logger.warning(
                    "Document process completed but storage is not ready document_id=%s safe_document_key=%s storage_dir=%s warnings=%s",
                    normalized_document_id,
                    safe_document_key(normalized_document_id),
                    storage_dir,
                    "; ".join(readiness.warnings),
                )
                message = (
                    f"{message} 知识库索引不完整，请重新解析。"
                    if readiness.warnings
                    else message
                )

            async with _rag_lock:
                _rag_instances[cache_key] = rag
            keep_rag_instance = True
            logger.info(
                "process_cached_fresh_rag document_id=%s safe_document_key=%s rag_id=%s lightrag_id=%s",
                normalized_document_id,
                cache_key,
                id(rag),
                id(rag.lightrag) if rag.lightrag is not None else None,
            )
            return ProcessResponse(
                document=summarize_document(pdf_path),
                message=message,
                sources=[],
                readiness_warnings=readiness.warnings,
                **process_summary,
            )
        except Exception as exc:
            write_process_failure(normalized_document_id, str(exc))
            await evict_rag_instance(normalized_document_id)
            raise
        finally:
            if not keep_rag_instance:
                await evict_rag_instance(normalized_document_id)
                if rag is not None:
                    try:
                        await rag.finalize_storages()
                    except Exception as exc:
                        logger.warning(
                            "Failed to finalize uncached process RAG document_id=%s safe_document_key=%s error=%s",
                            normalized_document_id,
                            safe_document_key(normalized_document_id),
                            exc,
                        )
            await unmark_document_processing(normalized_document_id)
            _global_process_lock.release()
            logger.info(
                "process_released_global_lock document_id=%s safe_document_key=%s",
                normalized_document_id,
                safe_document_key(normalized_document_id),
            )
