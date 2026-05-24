"""FastAPI entry point for the RAG teaching assistant.

The frontend must only call this API. RAG parsing, indexing, embedding, LLM
calls, and local file access all stay on the backend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
from dataclasses import fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from raganything import RAGAnything, RAGAnythingConfig
from raganything.utils import insert_text_content, separate_content


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
RAG_STORAGE_DIR = BASE_DIR / "rag_storage"
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf"}
ANSWER_LEVELS = {
    "beginner": "请用直观类比解释概念，避免堆砌术语，并补充必要背景。",
    "undergraduate": "请按定义、公式、物理含义、常见考点的顺序回答。",
    "expert": "请直接进入机制、边界条件、推导要点和局限性。",
}
TEXT_ONLY_INDEXED_MESSAGE = "Document parsed and text-only indexed. MVP text-only mode is active."
MULTIMODAL_INDEXED_MESSAGE = "Document parsed and indexed with multimodal processing enabled."

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

_rag_instance: Optional[RAGAnything] = None
_rag_lock: asyncio.Lock = asyncio.Lock()


def load_runtime_config() -> None:
    load_dotenv(BASE_DIR / ".env", override=False)
    local_cache = BASE_DIR / ".cache"
    (local_cache / "ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(local_cache / "ultralytics"))
    os.environ.setdefault("MINERU_BACKEND", "pipeline")
    os.environ.setdefault("MINERU_DEVICE", "cuda")
    os.environ.setdefault("MINERU_SOURCE", "modelscope")
    os.environ.setdefault("LLM_MODEL", "qwen-plus")
    os.environ.setdefault(
        "LLM_BINDING_HOST", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    os.environ.setdefault("ENABLE_MULTIMODAL", "false")
    os.environ.setdefault("ENABLE_IMAGE_PROCESSING", "true")
    os.environ.setdefault("ENABLE_TABLE_PROCESSING", "false")
    os.environ.setdefault("ENABLE_EQUATION_PROCESSING", "false")
    os.environ.setdefault("ENABLE_GENERIC_PROCESSING", "false")
    os.environ.setdefault("QWEN_VL_MODEL", "qwen-vl-max")
    os.environ.setdefault(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def as_int(value: str | None, default: int = 0) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


def as_bool(value: str | None, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid boolean value %r; using default %s.", value, default)
    return default


def multimodal_enabled() -> bool:
    return as_bool(os.getenv("ENABLE_MULTIMODAL"), False)


def generic_processing_enabled() -> bool:
    return as_bool(os.getenv("ENABLE_GENERIC_PROCESSING"), False)


def mineru_runtime_kwargs() -> dict[str, str]:
    kwargs: dict[str, str] = {}
    for env_name, kwarg_name, default in (
        ("MINERU_BACKEND", "backend", "pipeline"),
        ("MINERU_DEVICE", "device", "cuda"),
        ("MINERU_SOURCE", "source", "modelscope"),
    ):
        value = os.getenv(env_name, default)
        if value:
            kwargs[kwarg_name] = value
    return kwargs


def filter_rag_config_kwargs(config_kwargs: dict[str, Any]) -> dict[str, Any]:
    supported_fields = {field.name for field in fields(RAGAnythingConfig)}
    filtered = {
        key: value for key, value in config_kwargs.items() if key in supported_fields
    }
    unsupported_fields = sorted(set(config_kwargs) - supported_fields)
    if unsupported_fields:
        logger.warning(
            "Ignoring unsupported RAGAnythingConfig fields: %s",
            ", ".join(unsupported_fields),
        )
    return filtered


def validate_vision_messages(messages: Any) -> list[dict[str, Any]] | None:
    if not isinstance(messages, list) or not messages:
        logger.warning("Invalid vision messages: expected a non-empty list.")
        return None

    validated: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            logger.warning("Invalid vision message %s: expected object.", message_index)
            return None

        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            logger.warning("Invalid vision message %s: missing role.", message_index)
            return None

        if "content" not in message or message.get("content") is None:
            logger.warning("Invalid vision message %s: missing content.", message_index)
            return None

        content = message["content"]
        if isinstance(content, str):
            validated.append(dict(message))
            continue

        if not isinstance(content, list) or not content:
            logger.warning(
                "Invalid vision message %s: content must be string or non-empty list.",
                message_index,
            )
            return None

        for part_index, part in enumerate(content):
            if not isinstance(part, dict):
                logger.warning(
                    "Invalid vision message %s content part %s: expected object.",
                    message_index,
                    part_index,
                )
                return None

            part_type = part.get("type")
            if part_type == "text":
                if not isinstance(part.get("text"), str):
                    logger.warning(
                        "Invalid vision message %s content part %s: text must be string.",
                        message_index,
                        part_index,
                    )
                    return None
            elif part_type == "image_url":
                image_url = part.get("image_url")
                if (
                    not isinstance(image_url, dict)
                    or not isinstance(image_url.get("url"), str)
                    or not image_url.get("url")
                ):
                    logger.warning(
                        "Invalid vision message %s content part %s: image_url.url is required.",
                        message_index,
                        part_index,
                    )
                    return None
            else:
                logger.warning(
                    "Invalid vision message %s content part %s: unsupported type %r.",
                    message_index,
                    part_index,
                    part_type,
                )
                return None

        validated.append(dict(message))

    return validated


def response_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(part for part in parts if part)
    return str(content)


def safe_document_id(filename: str) -> str:
    return Path(filename).name


def document_path(document_id: str) -> Path:
    candidate = UPLOAD_DIR / safe_document_id(document_id)
    if candidate.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Document not found.")
    return candidate


def content_list_path(pdf_path: Path) -> Path | None:
    candidates = sorted(
        OUTPUT_DIR.glob(f"**/{pdf_path.stem}_content_list.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_content_list(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Unexpected content_list format: {path}")
    return [item for item in data if isinstance(item, dict)]


def document_status(pdf_path: Path) -> Literal["uploaded", "processed"]:
    return "processed" if content_list_path(pdf_path) else "uploaded"


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def build_embedding_func() -> EmbeddingFunc:
    model_name = os.getenv("EMBEDDING_LOCAL_MODEL", "BAAI/bge-small-zh-v1.5")
    model = load_embedding_model(model_name)

    async def embed(texts: Iterable[str]):
        return await asyncio.to_thread(lambda: model.encode(list(texts)))

    return EmbeddingFunc(embedding_dim=512, max_token_size=512, func=embed)


def build_model_functions():
    qwen_api_key = os.getenv("QWEN_API_KEY")
    llm_model = os.getenv("LLM_MODEL", "qwen-plus")
    llm_api_key = os.getenv("LLM_BINDING_API_KEY") or qwen_api_key
    llm_base_url = os.getenv(
        "LLM_BINDING_HOST", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    qwen_model = os.getenv("QWEN_VL_MODEL") or os.getenv("QWEN_MODEL", "qwen-vl-max")
    qwen_base_url = os.getenv("QWEN_BASE_URL", llm_base_url)

    if not llm_api_key:
        raise RuntimeError(
            "Missing LLM API key. Set LLM_BINDING_API_KEY or QWEN_API_KEY in .env."
        )

    async def llm_func(
        prompt,
        system_prompt="你是一名严谨的中文 AI 助教。回答必须基于已检索到的文档内容。",
        history_messages=None,
        **kwargs,
    ):
        return await openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=llm_api_key,
            base_url=llm_base_url,
            **clean_lightrag_kwargs(kwargs),
        )

    async def vision_func(
        prompt,
        system_prompt="你是一名擅长理解图像、表格、公式和文档截图的中文 AI 助教。",
        image_data=None,
        **kwargs,
    ):
        if not qwen_api_key:
            logger.warning("QWEN_API_KEY is missing; skipping vision model call.")
            return ""

        call_kwargs = dict(kwargs)
        raw_messages = call_kwargs.pop("messages", None)
        for key in ("response_format", "keyword_extraction", "image_data"):
            call_kwargs.pop(key, None)

        if raw_messages is not None:
            messages = validate_vision_messages(raw_messages)
            if messages is None:
                return ""
        else:
            if not image_data:
                logger.warning("image_data is empty; skipping vision model call.")
                return ""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or ""},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ],
                },
            ]

        client = None
        try:
            client = AsyncOpenAI(api_key=qwen_api_key, base_url=qwen_base_url)
            response = await client.chat.completions.create(
                model=qwen_model,
                messages=messages,
                **call_kwargs,
            )
            content = (
                response_content_to_text(response.choices[0].message.content)
                if response.choices
                else ""
            )
            if not content:
                logger.warning("Vision model returned empty content.")
            return content
        except Exception as exc:
            logger.warning("Vision model call failed: %s", exc)
            return ""
        finally:
            if client is not None:
                try:
                    await client.close()
                except Exception as exc:
                    logger.warning("Failed to close vision model client: %s", exc)

    return llm_func, vision_func


def clean_lightrag_kwargs(kwargs: dict) -> dict:
    cleaned = dict(kwargs)
    for key in ("response_format", "keyword_extraction", "image_data"):
        cleaned.pop(key, None)

    messages = cleaned.get("messages")
    if messages:
        normalized = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            normalized.append({"role": message.get("role", "user"), "content": content})
        cleaned["messages"] = normalized
    return cleaned


def build_rag() -> RAGAnything:
    llm_func, vision_func = build_model_functions()
    enable_multimodal = multimodal_enabled()
    enable_image_processing = enable_multimodal and as_bool(
        os.getenv("ENABLE_IMAGE_PROCESSING"), True
    )
    enable_table_processing = enable_multimodal and as_bool(
        os.getenv("ENABLE_TABLE_PROCESSING"), False
    )
    enable_equation_processing = enable_multimodal and as_bool(
        os.getenv("ENABLE_EQUATION_PROCESSING"), False
    )

    if enable_image_processing and not os.getenv("QWEN_API_KEY"):
        logger.warning(
            "ENABLE_MULTIMODAL=true and ENABLE_IMAGE_PROCESSING=true, but "
            "QWEN_API_KEY is missing; image processing is disabled."
        )
        enable_image_processing = False

    config_kwargs = {
        "working_dir": str(RAG_STORAGE_DIR),
        "parser_output_dir": str(OUTPUT_DIR),
        "parser": "mineru",
        "parse_method": os.getenv("PARSE_METHOD", "auto"),
        "mineru_backend": os.getenv("MINERU_BACKEND", "pipeline"),
        "mineru_device": os.getenv("MINERU_DEVICE", "cuda"),
        "mineru_source": os.getenv("MINERU_SOURCE", "modelscope"),
        "mineru_vram": as_int(os.getenv("MINERU_VRAM"), 0),
        "enable_image_processing": enable_image_processing,
        "enable_table_processing": enable_table_processing,
        "enable_equation_processing": enable_equation_processing,
    }
    config = RAGAnythingConfig(**filter_rag_config_kwargs(config_kwargs))
    return RAGAnything(
        config=config,
        llm_model_func=llm_func,
        vision_model_func=vision_func
        if enable_multimodal and enable_image_processing
        else None,
        embedding_func=build_embedding_func(),
    )


async def get_rag() -> RAGAnything:
    global _rag_instance
    if _rag_instance is None:
        async with _rag_lock:
            if _rag_instance is None:
                _rag_instance = build_rag()
    return _rag_instance


def knowledge_base_not_ready_response() -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "error": "knowledge_base_not_ready",
            "message": "知识库尚未就绪，请先解析文档后再提问。",
        },
    )


def first_non_empty_text(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            joined = "\n".join(str(part).strip() for part in value if str(part).strip())
            if joined:
                return joined
    return ""


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
    *, enabled: bool, processed: int, failed: int, skipped: int, attempted: int
) -> str:
    if not enabled:
        return "disabled"
    if attempted == 0:
        return "skipped"
    if failed == 0 and skipped == 0:
        return "processed"
    if processed > 0:
        return "partial_success"
    if failed > 0:
        return "failed"
    return "skipped"


def new_multimodal_stats() -> dict[str, dict[str, int]]:
    return {
        "image": {"processed": 0, "failed": 0, "skipped": 0, "attempted": 0},
        "table": {"processed": 0, "failed": 0, "skipped": 0, "attempted": 0},
        "equation": {"processed": 0, "failed": 0, "skipped": 0, "attempted": 0},
        "formula": {"processed": 0, "failed": 0, "skipped": 0, "attempted": 0},
        "generic": {"processed": 0, "failed": 0, "skipped": 0, "attempted": 0},
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
        enabled=rag.config.enable_equation_processing or rag.config.enable_image_processing,
        **stats["formula"],
    )

    attempted_total = sum(type_stats["attempted"] for type_stats in stats.values())
    failed_total = sum(type_stats["failed"] for type_stats in stats.values())

    if attempted_total == 0:
        summary["multimodal_status"] = (
            "partial_success"
            if summary["multimodal_warnings_count"] > 0
            else "skipped"
        )
    elif failed_total > 0 or summary["multimodal_warnings_count"] > 0:
        summary["multimodal_status"] = "partial_success"
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


def multimodal_item_limit(canonical_type: str) -> int:
    env_names = {
        "image": "MAX_IMAGE_ITEMS",
        "table": "MAX_TABLE_ITEMS",
        "equation": "MAX_EQUATION_ITEMS",
        "formula": "MAX_FORMULA_ITEMS",
        "generic": "MAX_GENERIC_ITEMS",
    }
    return as_int(os.getenv(env_names.get(canonical_type, "")), 0)


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
        if item_limit > 0 and stats[tracked_type]["attempted"] >= item_limit:
            stats[tracked_type]["skipped"] += 1
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


class DocumentSummary(BaseModel):
    id: str
    name: str
    size: int
    status: Literal["uploaded", "processed"]


class SourceItem(BaseModel):
    id: str
    type: str
    page: int | None = None
    text: str


class UploadResponse(BaseModel):
    document: DocumentSummary


class ProcessResponse(BaseModel):
    document: DocumentSummary
    message: str
    sources: list[SourceItem]
    text_indexed: bool = False
    multimodal_enabled: bool = False
    multimodal_status: str = "disabled"
    image_status: str = "disabled"
    table_status: str = "disabled"
    equation_status: str = "disabled"
    formula_status: str = "disabled"
    skipped_multimodal_items_count: int = 0
    skipped_by_reason: dict[str, int] = Field(default_factory=dict)
    multimodal_warnings_count: int = 0
    warnings_summary: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    document_id: str | None = None
    level: str = "undergraduate"
    mode: str = "hybrid"


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class CacheResponse(BaseModel):
    ok: bool


class HealthResponse(BaseModel):
    ok: bool
    has_api_key: bool
    mineru_backend: str
    mineru_device: str


def summarize_document(pdf_path: Path) -> DocumentSummary:
    return DocumentSummary(
        id=pdf_path.name,
        name=pdf_path.name,
        size=pdf_path.stat().st_size,
        status=document_status(pdf_path),
    )


def extract_sources(pdf_path: Path, limit: int = 12) -> list[SourceItem]:
    path = content_list_path(pdf_path)
    if not path:
        return []

    sources: list[SourceItem] = []
    for index, item in enumerate(load_content_list(path)):
        item_type = item.get("type")
        text = " ".join(
            (item.get("text") or item.get("table_body") or "").split()
        )
        if item_type not in {"text", "equation", "table"} or not text:
            continue
        page_idx = item.get("page_idx")
        page = int(page_idx) + 1 if isinstance(page_idx, int) else None
        sources.append(
            SourceItem(
                id=f"{pdf_path.stem}-{index}",
                type=str(item_type),
                page=page,
                text=text[:600],
            )
        )
        if len(sources) >= limit:
            break
    return sources


load_runtime_config()
app = FastAPI(title="RAG Teaching Assistant API")

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        ok=True,
        has_api_key=bool(os.getenv("LLM_BINDING_API_KEY") or os.getenv("QWEN_API_KEY")),
        mineru_backend=os.getenv("MINERU_BACKEND", "pipeline"),
        mineru_device=os.getenv("MINERU_DEVICE", "cuda"),
    )


@app.get("/api/documents", response_model=list[DocumentSummary])
async def list_documents() -> list[DocumentSummary]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return [summarize_document(path) for path in sorted(UPLOAD_DIR.glob("*.pdf"))]


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    filename = safe_document_id(file.filename or "")
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / filename
    with target.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return UploadResponse(document=summarize_document(target))


@app.post("/api/documents/{document_id}/process", response_model=ProcessResponse)
async def process_document(document_id: str) -> ProcessResponse:
    pdf_path = document_path(document_id)
    rag = await get_rag()
    init_result = await rag._ensure_lightrag_initialized()
    if not init_result.get("success"):
        raise HTTPException(
            status_code=500,
            detail=init_result.get("error", "LightRAG initialization failed"),
        )

    if multimodal_enabled():
        process_summary = await process_document_multimodal_safe(rag, pdf_path)
        message = MULTIMODAL_INDEXED_MESSAGE
    else:
        await process_document_text_only(rag, pdf_path)
        process_summary = new_multimodal_summary(False)
        process_summary["text_indexed"] = True
        message = TEXT_ONLY_INDEXED_MESSAGE

    return ProcessResponse(
        document=summarize_document(pdf_path),
        message=message,
        sources=[],
        **process_summary,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse | JSONResponse:
    level_prompt = ANSWER_LEVELS.get(request.level, ANSWER_LEVELS["undergraduate"])
    rag = await get_rag()
    pdf_path: Path | None = None

    if request.document_id:
        pdf_path = document_path(request.document_id)
        if document_status(pdf_path) != "processed":
            return knowledge_base_not_ready_response()
    else:
        pdfs = sorted(UPLOAD_DIR.glob("*.pdf")) if UPLOAD_DIR.exists() else []
        pdf_path = pdfs[0] if pdfs else None
        if pdf_path and document_status(pdf_path) != "processed":
            return knowledge_base_not_ready_response()

    init_result = await rag._ensure_lightrag_initialized()
    if not init_result.get("success") or rag.lightrag is None:
        return knowledge_base_not_ready_response()

    prompt = f"{request.question.strip()}\n\n回答风格：{level_prompt}"
    try:
        answer = await rag.aquery(prompt, mode=request.mode)
        answer = response_content_to_text(answer)
        if not answer.strip():
            logger.warning(
                "VLM enhanced chat query returned empty answer; retrying with vlm_enhanced=False."
            )
            try:
                answer = response_content_to_text(
                    await rag.aquery(prompt, mode=request.mode, vlm_enhanced=False)
                )
            except TypeError as error:
                logger.warning(
                    "vlm_enhanced=False fallback is unsupported: %s", error
                )
                answer = ""
    except ValueError as error:
        if "No LightRAG instance available" in str(error):
            return knowledge_base_not_ready_response()
        raise
    return ChatResponse(
        answer=answer,
        sources=[],
    )


@app.delete("/api/cache", response_model=CacheResponse)
async def clear_runtime_cache() -> CacheResponse:
    for path in (RAG_STORAGE_DIR, OUTPUT_DIR):
        if path.exists():
            shutil.rmtree(path)
    return CacheResponse(ok=True)
