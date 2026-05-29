"""FastAPI entry point for the RAG teaching assistant.

The frontend must only call this API. RAG parsing, indexing, embedding, LLM
calls, and local file access all stay on the backend.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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
    "beginner": "请用直观类比和短步骤讲清楚核心概念，先补必要背景，再给少量必要公式。",
    "undergraduate": "请兼顾概念、公式、推导、计算和应用，明确公式适用条件与常见误区。",
    "expert": "请直接进入机制、严格条件、边界情况、证明思路、反例和局限性。",
}
ANSWER_LEVEL_LABELS = {
    "beginner": "入门",
    "undergraduate": "本科",
    "expert": "专家",
}
QUIZ_LEVEL_GUIDANCE = {
    "beginner": "入门：更多概念识别和直观理解题；少量公式；解析更直观；避免复杂证明。",
    "undergraduate": "本科：概念、公式、应用并重；包含计算题；强调边界条件和公式使用条件。",
    "expert": "专家：强调严格条件、反例、证明、模型假设和综合迁移；题目更综合。",
}
TEXT_ONLY_INDEXED_MESSAGE = "Document parsed and text-only indexed. MVP text-only mode is active."
MULTIMODAL_INDEXED_MESSAGE = "Document parsed and indexed with multimodal processing enabled."
IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
SINGLE_DOCUMENT_RELATED_IMAGE_LIMIT = 4
MULTI_DOCUMENT_RELATED_IMAGE_LIMIT = 6
MULTI_DOCUMENT_RELATED_IMAGE_PER_DOCUMENT_LIMIT = 2
INLINE_IMAGE_REF_PATTERN = re.compile(r"\[\[image:\s*([A-Za-z0-9._:-]+)\s*\]\]")

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

DocumentState = Literal[
    "uploaded",
    "parsed",
    "indexing",
    "ready_for_chat",
    "partial_success",
    "failed",
]
TaskIntent = Literal[
    "summary",
    "explain",
    "quiz",
    "tutoring",
    "compare",
    "solve",
    "flashcards",
    "study_plan",
]

_rag_instances: dict[str, RAGAnything] = {}
_rag_lock: asyncio.Lock = asyncio.Lock()
_global_process_lock: asyncio.Lock = asyncio.Lock()
_document_locks: dict[str, asyncio.Lock] = {}
_document_locks_guard: asyncio.Lock = asyncio.Lock()
_documents_processing: set[str] = set()


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
    os.environ.setdefault("ENABLE_MULTIMODAL", "true")
    os.environ.setdefault("ENABLE_IMAGE_PROCESSING", "true")
    os.environ.setdefault("ENABLE_TABLE_PROCESSING", "true")
    os.environ.setdefault("ENABLE_EQUATION_PROCESSING", "true")
    os.environ.setdefault("ENABLE_FORMULA_PROCESSING", "true")
    os.environ.setdefault("ENABLE_GENERIC_PROCESSING", "false")
    os.environ.setdefault("QWEN_VL_MODEL", "qwen-vl-max")
    os.environ.setdefault(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    os.environ.setdefault("ENABLE_RERANK", "false")
    os.environ.setdefault("RERANK_MODEL", "")


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
    return as_bool(os.getenv("ENABLE_MULTIMODAL"), True)


def formula_processing_enabled() -> bool:
    return multimodal_enabled() and as_bool(
        os.getenv("ENABLE_FORMULA_PROCESSING"), True
    )


def generic_processing_enabled() -> bool:
    return as_bool(os.getenv("ENABLE_GENERIC_PROCESSING"), False)


def rerank_model_name() -> str:
    return (os.getenv("RERANK_MODEL") or "").strip()


def rerank_requested() -> bool:
    return as_bool(os.getenv("ENABLE_RERANK"), False)


RERANK_BINDING_ALIASES = {
    "aliyun": "aliyun",
    "ali": "aliyun",
    "dashscope": "aliyun",
    "jina": "jina",
    "cohere": "cohere",
}
# Backend-only rerank state; expose status flags, never provider credentials.
_rerank_runtime_state: dict[str, Any] = {}


def rerank_binding_name() -> str:
    return (os.getenv("RERANK_BINDING") or "aliyun").strip().lower()


def rerank_provider_name() -> str | None:
    return RERANK_BINDING_ALIASES.get(rerank_binding_name())


def rerank_base_url() -> str | None:
    return (
        (os.getenv("RERANK_BASE_URL") or "").strip()
        or (os.getenv("RERANK_BINDING_HOST") or "").strip()
        or None
    )


def rerank_top_n() -> int | None:
    configured_top_n = as_int(os.getenv("RERANK_TOP_N"), 0)
    return configured_top_n if configured_top_n > 0 else None


def rerank_binding_api_key(provider: str | None = None) -> str | None:
    provider = provider or rerank_provider_name()
    if provider == "aliyun":
        return (
            os.getenv("RERANK_BINDING_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("QWEN_API_KEY")
            or None
        )
    if provider == "jina":
        return os.getenv("RERANK_BINDING_API_KEY") or os.getenv("JINA_API_KEY") or None
    if provider == "cohere":
        return (
            os.getenv("RERANK_BINDING_API_KEY")
            or os.getenv("COHERE_API_KEY")
            or None
        )
    return os.getenv("RERANK_BINDING_API_KEY") or None


def update_rerank_runtime_state(**updates: Any) -> None:
    _rerank_runtime_state.update(updates)


def current_rerank_runtime_status() -> dict[str, Any]:
    requested = rerank_requested()
    model = rerank_model_name() or None
    provider = rerank_provider_name() if requested else None
    status: dict[str, Any] = {
        "requested": requested,
        "enabled": False,
        "model": model,
        "provider": provider,
        "model_loaded": False,
        "last_error": None,
        "reason": "disabled_by_config" if not requested else None,
        "top_n": rerank_top_n(),
        "base_url": rerank_base_url(),
    }
    status.update(_rerank_runtime_state)
    status["requested"] = requested
    status["model"] = model
    status["provider"] = provider
    status["top_n"] = rerank_top_n()
    status["base_url"] = rerank_base_url()
    if not requested:
        status.update(
            {
                "enabled": False,
                "model_loaded": False,
                "last_error": None,
                "reason": "disabled_by_config",
            }
        )
    elif not model:
        status.update(
            {
                "enabled": False,
                "model_loaded": False,
                "last_error": "missing_model",
                "reason": "missing_model",
            }
        )
    return status


def rerank_enabled_for_query() -> bool:
    status = current_rerank_runtime_status()
    return bool(
        status["requested"] and status["model"] and status["enabled"] and status["model_loaded"]
    )


def log_rerank_startup_status() -> None:
    if rerank_requested() and not rerank_model_name():
        logger.warning(
            "ENABLE_RERANK=true but RERANK_MODEL is empty; rerank will be disabled for queries."
        )
    elif rerank_requested():
        provider = rerank_provider_name()
        if provider is None:
            logger.warning(
                "Unsupported RERANK_BINDING=%r; rerank will be disabled for queries.",
                rerank_binding_name(),
            )
        elif not rerank_binding_api_key(provider):
            logger.warning(
                "ENABLE_RERANK=true and RERANK_MODEL=%s, but no rerank API key is available for provider=%s; rerank will be disabled for queries.",
                rerank_model_name(),
                provider,
            )
        else:
            logger.info(
                "Rerank requested: provider=%s model=%s base_url=%s top_n=%s",
                provider,
                rerank_model_name(),
                rerank_base_url() or "provider_default",
                rerank_top_n() or "QueryParam.chunk_top_k",
            )
    else:
        logger.info("Rerank disabled.")


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


def safe_document_key(document_id: str) -> str:
    filename = safe_document_id(document_id)
    stem = Path(filename).stem
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip(".-_")
    sanitized = re.sub(r"-{2,}", "-", sanitized)[:80].strip(".-_")
    if not sanitized:
        sanitized = "document"
    short_hash = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:12]
    return f"{sanitized}-{short_hash}"


def document_storage_dir(document_id: str) -> Path:
    return RAG_STORAGE_DIR / "documents" / safe_document_key(document_id)


def lightrag_working_root() -> Path:
    return RAG_STORAGE_DIR / "documents"


def lightrag_workspace(document_id: str) -> str:
    return safe_document_key(document_id)


def process_failure_path(document_id: str) -> Path:
    return document_storage_dir(document_id) / ".process_failed.json"


@dataclass
class StorageReadiness:
    ready: bool
    status: DocumentState
    warnings: list[str]
    storage_dir: Path
    chunks_count: int = 0


@dataclass
class DocumentChatContext:
    document_id: str
    name: str
    status: DocumentState
    storage_dir: Path
    readiness: StorageReadiness


@dataclass
class DocumentAnswer:
    document_id: str
    name: str
    status: DocumentState
    storage_dir: Path
    answer: str
    vlm_image_paths: list[str]
    fallback_used: bool
    answer_source_path: str
    prompt_template_used: str
    timings: dict[str, float]


@dataclass(frozen=True)
class ImageAsset:
    image_id: str
    document_id: str
    document_name: str
    filename: str
    caption: str | None
    footnote: str | None
    page: int | None
    bbox: Any | None
    absolute_path: Path
    url: str
    content_index: int
    context_text: str
    source_type: str | None = None


def same_document_file(file_path: Any, document_id: str) -> bool:
    if not isinstance(file_path, str) or not file_path.strip():
        return False
    return Path(file_path).name == safe_document_id(document_id)


def document_cache_key(document_id: str) -> str:
    return safe_document_key(safe_document_id(document_id))


def is_document_processing(document_id: str) -> bool:
    return document_cache_key(document_id) in _documents_processing


async def document_lock(document_id: str) -> asyncio.Lock:
    cache_key = document_cache_key(document_id)
    async with _document_locks_guard:
        lock = _document_locks.get(cache_key)
        if lock is None:
            lock = asyncio.Lock()
            _document_locks[cache_key] = lock
        return lock


async def mark_document_processing(document_id: str) -> bool:
    cache_key = document_cache_key(document_id)
    async with _document_locks_guard:
        if cache_key in _documents_processing:
            return False
        _documents_processing.add(cache_key)
        if cache_key not in _document_locks:
            _document_locks[cache_key] = asyncio.Lock()
        return True


async def unmark_document_processing(document_id: str) -> None:
    cache_key = document_cache_key(document_id)
    async with _document_locks_guard:
        _documents_processing.discard(cache_key)


async def evict_rag_instance(document_id: str) -> None:
    cache_key = document_cache_key(document_id)
    async with _rag_lock:
        evicted = _rag_instances.pop(cache_key, None)
    if evicted is not None:
        try:
            await evicted.finalize_storages()
        except Exception as exc:
            logger.warning(
                "Failed to finalize evicted RAG instance document_id=%s safe_document_key=%s error=%s",
                safe_document_id(document_id),
                cache_key,
                exc,
            )
        logger.info(
            "Evicted document RAG instance document_id=%s safe_document_key=%s",
            safe_document_id(document_id),
            cache_key,
        )


def clear_lightrag_shared_workspace(workspace: str) -> None:
    try:
        from lightrag.kg import shared_storage
    except Exception as exc:
        logger.warning(
            "process_clear_shared_workspace_failed workspace=%s error=%s",
            workspace,
            exc,
        )
        return

    prefix = f"{workspace}:"
    removed_keys: list[str] = []
    for attr_name in ("_shared_dicts", "_init_flags", "_update_flags"):
        mapping = getattr(shared_storage, attr_name, None)
        if mapping is None:
            continue
        for key in list(mapping.keys()):
            key_text = str(key)
            if key_text == workspace or key_text.startswith(prefix):
                try:
                    del mapping[key]
                except Exception:
                    try:
                        mapping.pop(key, None)
                    except Exception as exc:
                        logger.warning(
                            "process_clear_shared_workspace_key_failed workspace=%s mapping=%s key=%s error=%s",
                            workspace,
                            attr_name,
                            key_text,
                            exc,
                        )
                        continue
                removed_keys.append(f"{attr_name}:{key_text}")

    logger.info(
        "process_clear_shared_workspace_done workspace=%s removed_keys=%s",
        workspace,
        removed_keys,
    )


def clear_all_lightrag_shared_storage() -> None:
    try:
        from lightrag.kg import shared_storage
    except Exception as exc:
        logger.warning("clear_lightrag_shared_storage_failed error=%s", exc)
        return

    for attr_name in ("_shared_dicts", "_init_flags", "_update_flags"):
        mapping = getattr(shared_storage, attr_name, None)
        if mapping is not None:
            try:
                mapping.clear()
            except Exception as exc:
                logger.warning(
                    "clear_lightrag_shared_storage_mapping_failed mapping=%s error=%s",
                    attr_name,
                    exc,
                )
    try:
        shared_storage.set_default_workspace(None)
    except Exception as exc:
        logger.warning("clear_lightrag_default_workspace_failed error=%s", exc)


def set_lightrag_default_workspace(workspace: str) -> None:
    try:
        from lightrag.kg.shared_storage import set_default_workspace

        set_default_workspace(workspace)
    except Exception as exc:
        logger.warning(
            "set_lightrag_default_workspace_failed workspace=%s error=%s",
            workspace,
            exc,
        )


def ensure_document_storage_reset(document_id: str) -> Path:
    normalized_document_id = safe_document_id(document_id)
    storage_root = (RAG_STORAGE_DIR / "documents").resolve(strict=False)
    storage_dir = document_storage_dir(normalized_document_id)
    resolved_storage_dir = storage_dir.resolve(strict=False)
    if not is_relative_to_path(resolved_storage_dir, storage_root):
        raise RuntimeError(f"Unsafe document storage path: {storage_dir}")

    logger.info(
        "process_clean_storage_start document_id=%s safe_document_key=%s storage_dir=%s",
        normalized_document_id,
        safe_document_key(normalized_document_id),
        storage_dir,
    )
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_exists_after_delete = storage_dir.exists()
    logger.info(
        "process_clean_storage_done document_id=%s safe_document_key=%s storage_dir=%s storage_exists_after_delete=%s",
        normalized_document_id,
        safe_document_key(normalized_document_id),
        storage_dir,
        str(storage_exists_after_delete).lower(),
    )
    if storage_exists_after_delete:
        raise RuntimeError(f"Failed to delete document storage: {storage_dir}")

    clear_lightrag_shared_workspace(lightrag_workspace(normalized_document_id))
    storage_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Prepared empty document storage document_id=%s safe_document_key=%s storage_dir=%s",
        normalized_document_id,
        safe_document_key(normalized_document_id),
        storage_dir,
    )
    return storage_dir


def build_fresh_document_rag(document_id: str, storage_dir: Path) -> RAGAnything:
    normalized_document_id = safe_document_id(document_id)
    logger.info(
        "Building fresh document RAG document_id=%s safe_document_key=%s working_dir=%s",
        normalized_document_id,
        safe_document_key(normalized_document_id),
        storage_dir,
    )
    rag = build_rag(document_id=normalized_document_id, working_dir=storage_dir)
    logger.info(
        "build_fresh_document_rag_created document_id=%s safe_document_key=%s working_dir=%s rag_id=%s lightrag_id=%s",
        normalized_document_id,
        safe_document_key(normalized_document_id),
        storage_dir,
        id(rag),
        id(rag.lightrag) if rag.lightrag is not None else None,
    )
    return rag


def write_process_failure(document_id: str, message: str) -> None:
    storage_dir = document_storage_dir(document_id)
    storage_dir.mkdir(parents=True, exist_ok=True)
    payload = {"message": message, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00")}
    try:
        process_failure_path(document_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to write process failure marker for %s: %s", document_id, exc)


def clear_process_failure(document_id: str) -> None:
    marker = process_failure_path(document_id)
    if marker.exists():
        try:
            marker.unlink()
        except Exception as exc:
            logger.warning("Failed to clear process failure marker for %s: %s", document_id, exc)


def vdb_records_by_file(storage_dir: Path, filename: str) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    data = read_json_dict(storage_dir / "vdb_chunks.json")
    records = data.get("data", []) if isinstance(data.get("data"), list) else []
    current_records: list[dict[str, Any]] = []
    polluted_files: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        file_path = record.get("file_path")
        if same_document_file(file_path, filename):
            current_records.append(record)
        elif isinstance(file_path, str) and file_path.strip():
            polluted_files.add(Path(file_path).name)
    if polluted_files:
        warnings.append(
            "vdb_chunks contains chunks from other documents: "
            + ", ".join(sorted(polluted_files))
        )
    return current_records, warnings


def document_storage_readiness(document_id: str) -> StorageReadiness:
    filename = safe_document_id(document_id)
    storage_dir = document_storage_dir(filename)
    warnings: list[str] = []

    if not storage_dir.exists():
        return StorageReadiness(False, "parsed", ["document-scoped storage is missing"], storage_dir)

    required_files = (
        "kv_store_doc_status.json",
        "kv_store_text_chunks.json",
        "vdb_chunks.json",
        "graph_chunk_entity_relation.graphml",
    )
    for required_file in required_files:
        if not (storage_dir / required_file).exists():
            warnings.append(f"{required_file} is missing")

    doc_statuses = read_json_dict(storage_dir / "kv_store_doc_status.json")
    text_chunks = read_json_dict(storage_dir / "kv_store_text_chunks.json")

    current_doc_status: dict[str, Any] | None = None
    doc_status_refs: list[str] = []
    other_doc_status_files: set[str] = set()
    for doc_status_id, status in doc_statuses.items():
        if not isinstance(status, dict):
            continue
        file_path = status.get("file_path")
        file_name = Path(file_path).name if isinstance(file_path, str) and file_path.strip() else "<missing>"
        doc_status_refs.append(f"{doc_status_id}:{file_name}")
        if same_document_file(file_path, filename):
            current_doc_status = status
        elif isinstance(file_path, str) and file_path.strip():
            other_doc_status_files.add(Path(file_path).name)

    if other_doc_status_files:
        warnings.append(
            "doc_status contains other documents: "
            + ", ".join(sorted(other_doc_status_files))
        )

    if current_doc_status is None:
        warnings.append("current document doc_status is missing")
        logger.warning(
            "Document storage readiness failed expected_document_id=%s safe_document_key=%s storage_dir=%s doc_status_refs=%s polluted_doc_status_files=%s warnings=%s",
            filename,
            safe_document_key(filename),
            storage_dir,
            doc_status_refs,
            sorted(other_doc_status_files),
            warnings,
        )
        return StorageReadiness(False, "partial_success", warnings, storage_dir)

    raw_status = str(current_doc_status.get("status", "")).lower()
    if raw_status not in {"processed", "success", "completed", "complete"}:
        warnings.append(f"current document doc_status is not processed: {raw_status or 'unknown'}")

    chunks_list = [
        str(chunk_id)
        for chunk_id in current_doc_status.get("chunks_list", [])
        if str(chunk_id).strip()
    ]
    chunks_count = int(current_doc_status.get("chunks_count") or len(chunks_list) or 0)
    if chunks_count <= 0:
        warnings.append("current document chunks_count is 0")
    if not chunks_list:
        warnings.append("current document chunks_list is empty")

    text_chunk_files: set[str] = set()
    all_text_chunk_files: set[str] = set()
    current_text_chunk_ids: set[str] = set()
    for chunk_id, chunk in text_chunks.items():
        if not isinstance(chunk, dict):
            continue
        file_path = chunk.get("file_path")
        if isinstance(file_path, str) and file_path.strip():
            all_text_chunk_files.add(Path(file_path).name)
        if same_document_file(file_path, filename):
            current_text_chunk_ids.add(str(chunk_id))
        elif isinstance(file_path, str) and file_path.strip():
            text_chunk_files.add(Path(file_path).name)

    if text_chunk_files:
        warnings.append(
            "text_chunks contains other documents: " + ", ".join(sorted(text_chunk_files))
        )

    missing_text_chunks: list[str] = []
    wrong_file_chunks: list[str] = []
    empty_text_chunks: list[str] = []
    for chunk_id in chunks_list:
        chunk = text_chunks.get(chunk_id)
        if not isinstance(chunk, dict):
            missing_text_chunks.append(chunk_id)
            continue
        if not str(chunk.get("content", "")).strip():
            empty_text_chunks.append(chunk_id)
        if chunk.get("file_path") and not same_document_file(chunk.get("file_path"), filename):
            wrong_file_chunks.append(chunk_id)

    if missing_text_chunks:
        warnings.append(
            "chunks_list references missing text_chunks: "
            + ", ".join(missing_text_chunks[:8])
        )
    if empty_text_chunks:
        warnings.append(
            "chunks_list references empty text_chunks: "
            + ", ".join(empty_text_chunks[:8])
        )
    if wrong_file_chunks:
        warnings.append(
            "chunks_list references chunks from another file: "
            + ", ".join(wrong_file_chunks[:8])
        )

    vdb_records, vdb_warnings = vdb_records_by_file(storage_dir, filename)
    warnings.extend(vdb_warnings)
    vdb_current_ids = {
        str(record.get("__id__") or record.get("id"))
        for record in vdb_records
        if str(record.get("__id__") or record.get("id") or "").strip()
    }
    expected_ids = set(chunks_list)
    if not vdb_current_ids:
        warnings.append("vdb_chunks has no chunks for current document")
    missing_vdb = sorted(expected_ids - vdb_current_ids) if expected_ids else []
    if missing_vdb:
        warnings.append(
            "chunks_list references chunks missing from vdb_chunks: "
            + ", ".join(missing_vdb[:8])
        )
    vdb_without_text = sorted(vdb_current_ids - set(text_chunks.keys()))
    if vdb_without_text:
        warnings.append(
            "vdb_chunks contains ids missing from text_chunks: "
            + ", ".join(vdb_without_text[:8])
        )
    text_without_vdb = sorted(current_text_chunk_ids - vdb_current_ids)
    if text_without_vdb:
        warnings.append(
            "text_chunks contains current-document ids missing from vdb_chunks: "
            + ", ".join(text_without_vdb[:8])
        )

    for tracking_file in ("kv_store_entity_chunks.json", "kv_store_relation_chunks.json"):
        tracking = read_json_dict(storage_dir / tracking_file)
        missing_refs: set[str] = set()
        for value in tracking.values():
            if not isinstance(value, dict):
                continue
            for chunk_id in value.get("chunk_ids", []):
                chunk_id = str(chunk_id)
                if chunk_id and chunk_id not in text_chunks:
                    missing_refs.add(chunk_id)
        if missing_refs:
            warnings.append(
                f"{tracking_file} references ids missing from text_chunks: "
                + ", ".join(sorted(missing_refs)[:8])
            )

    ready = not warnings
    if not ready:
        logger.warning(
            "Document storage readiness failed expected_document_id=%s safe_document_key=%s storage_dir=%s doc_status_refs=%s polluted_doc_status_files=%s text_chunk_files=%s polluted_text_chunk_files=%s chunks_count=%s missing_text_chunks_count=%s wrong_file_chunks_count=%s empty_text_chunks_count=%s missing_vdb_count=%s vdb_without_text_count=%s text_without_vdb_count=%s warnings=%s",
            filename,
            safe_document_key(filename),
            storage_dir,
            doc_status_refs,
            sorted(other_doc_status_files),
            sorted(all_text_chunk_files),
            sorted(text_chunk_files),
            chunks_count,
            len(missing_text_chunks),
            len(wrong_file_chunks),
            len(empty_text_chunks),
            len(missing_vdb),
            len(vdb_without_text),
            len(text_without_vdb),
            warnings,
        )
    return StorageReadiness(
        ready=ready,
        status="ready_for_chat" if ready else "partial_success",
        warnings=warnings,
        storage_dir=storage_dir,
        chunks_count=chunks_count,
    )


def document_storage_ready(document_id: str) -> bool:
    return document_storage_readiness(document_id).ready


def file_name_set_from_doc_statuses(doc_statuses: dict[str, Any]) -> set[str]:
    file_names: set[str] = set()
    for status in doc_statuses.values():
        if not isinstance(status, dict):
            continue
        file_path = status.get("file_path")
        if isinstance(file_path, str) and file_path.strip():
            file_names.add(Path(file_path).name)
    return file_names


def file_name_set_from_text_chunks(text_chunks: dict[str, Any]) -> set[str]:
    file_names: set[str] = set()
    for chunk in text_chunks.values():
        if not isinstance(chunk, dict):
            continue
        file_path = chunk.get("file_path")
        if isinstance(file_path, str) and file_path.strip():
            file_names.add(Path(file_path).name)
    return file_names


def file_name_set_from_vdb_chunks(storage_dir: Path) -> set[str]:
    data = read_json_dict(storage_dir / "vdb_chunks.json")
    records = data.get("data", []) if isinstance(data.get("data"), list) else []
    file_names: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        file_path = record.get("file_path")
        if isinstance(file_path, str) and file_path.strip():
            file_names.add(Path(file_path).name)
    return file_names


def document_storage_debug_payload(document_id: str) -> dict[str, Any]:
    filename = safe_document_id(document_id)
    safe_key = safe_document_key(filename)
    storage_dir = document_storage_dir(filename)
    doc_statuses = read_json_dict(storage_dir / "kv_store_doc_status.json")
    text_chunks = read_json_dict(storage_dir / "kv_store_text_chunks.json")
    doc_status_file_ids = file_name_set_from_doc_statuses(doc_statuses)
    text_chunk_file_ids = file_name_set_from_text_chunks(text_chunks)
    vdb_chunk_file_ids = file_name_set_from_vdb_chunks(storage_dir)
    polluted_doc_status_ids = sorted(doc_status_file_ids - {filename})
    polluted_text_chunk_ids = sorted(text_chunk_file_ids - {filename})
    polluted_vdb_chunk_ids = sorted(vdb_chunk_file_ids - {filename})
    readiness = document_storage_readiness(filename)

    return {
        "document_id": filename,
        "safe_document_key": safe_key,
        "storage_dir": str(storage_dir),
        "doc_status_document_ids": sorted(doc_statuses.keys()),
        "doc_status_file_ids": sorted(doc_status_file_ids),
        "text_chunk_file_ids": sorted(text_chunk_file_ids),
        "vdb_chunk_file_ids": sorted(vdb_chunk_file_ids),
        "polluted_ids": sorted(
            set(polluted_doc_status_ids)
            | set(polluted_text_chunk_ids)
            | set(polluted_vdb_chunk_ids)
        ),
        "polluted_doc_status_ids": polluted_doc_status_ids,
        "polluted_text_chunk_ids": polluted_text_chunk_ids,
        "polluted_vdb_chunk_ids": polluted_vdb_chunk_ids,
        "chunks_count": readiness.chunks_count,
        "warnings": readiness.warnings,
    }


def normalize_answer_level(level: str) -> str:
    return level if level in ANSWER_LEVELS else "undergraduate"


def contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def detect_task_intent(question: str) -> TaskIntent:
    normalized = question.strip().lower()
    if (
        re.search(
            r"\b(quiz|quizzes|exercise|exercises|practice questions?|problem set|test|exam)\b",
            normalized,
        )
        or contains_any(
            normalized,
            (
                "练习题",
                "练习",
                "测验",
                "小测",
                "测试题",
                "出题",
                "考我",
                "生成题目",
                "生成题",
                "习题",
                "试题",
            ),
        )
    ):
        return "quiz"
    if contains_any(
        normalized,
        ("flashcard", "flashcards", "anki", "记忆卡", "抽认卡", "知识卡片", "卡片"),
    ):
        return "flashcards"
    if contains_any(
        normalized,
        ("学习计划", "复习计划", "备考计划", "study plan", "学习路径", "复习安排"),
    ):
        return "study_plan"
    if contains_any(
        normalized,
        ("比较", "对比", "异同", "差异", "共同点", "compare", "comparison", "difference"),
    ):
        return "compare"
    if contains_any(
        normalized,
        ("解题", "求解", "计算", "证明", "推证明", "solve", "calculate", "prove"),
    ):
        return "solve"
    if contains_any(
        normalized,
        ("像老师", "带我学", "循序渐进", "教学", "辅导", "tutor", "teach me"),
    ):
        return "tutoring"
    if contains_any(
        normalized,
        ("解释", "讲解", "推导", "原理", "为什么", "explain", "derive"),
    ):
        return "explain"
    if contains_any(
        normalized,
        (
            "总结",
            "概括",
            "梳理",
            "主要内容",
            "讲了什么",
            "考点",
            "知识点",
            "summary",
            "summarize",
            "overview",
        ),
    ):
        return "summary"
    return "explain"


def is_summary_query(question: str) -> bool:
    return detect_task_intent(question) == "summary"


def prompt_template_name(task_intent: TaskIntent, document_count: int) -> str:
    scope = "multi_document" if document_count > 1 else "single_document"
    return f"{task_intent}_{scope}"


def level_guidance(level_key: str, level_prompt: str, task_intent: TaskIntent) -> str:
    level_label = ANSWER_LEVEL_LABELS.get(level_key, ANSWER_LEVEL_LABELS["undergraduate"])
    guidance = f"当前回答深度：{level_label}。{level_prompt}"
    if task_intent == "quiz":
        guidance += f"\n{QUIZ_LEVEL_GUIDANCE.get(level_key, QUIZ_LEVEL_GUIDANCE['undergraduate'])}"
    return guidance


def build_task_system_prompt(task_intent: TaskIntent) -> str:
    base_prompt = (
        "你是一名个性化中文导师型 RAG 助教。必须优先满足用户当前任务，"
        "只基于检索到的文档内容回答；不要把所有请求默认改写成文档总结。"
        "如果用户要 quiz、练习、讲解、比较、解题、记忆卡或学习计划，就按对应任务输出。"
        "只有用户明确要求总结或询问文档讲了什么时，才输出总结。"
    )
    if task_intent == "quiz":
        return (
            base_prompt
            + " 当前任务是生成 quiz，必须真的出题，包含答案、解析、难度控制和来源标注。"
        )
    return base_prompt


def build_quiz_response_strategy(level_key: str, document_count: int) -> str:
    level_label = ANSWER_LEVEL_LABELS.get(level_key, ANSWER_LEVEL_LABELS["undergraduate"])
    multi_document_rules = ""
    if document_count > 1:
        multi_document_rules = (
            "\n多文档要求：必须覆盖每个选中文档；每题标注来源文档；"
            "至少 2 道题为跨文档综合题，来源标注为“综合两个文档”或“综合多个文档”；"
            "不要只基于其中一个文档出题。"
        )
    return (
        "任务类型：quiz。\n"
        "必须输出 quiz，不要只总结概念，不要只列常见考点。\n"
        f"## 适用对象必须写为：{level_label}，并体现该难度要求。\n"
        "输出结构必须包含：\n"
        "# 个性化 Quiz\n\n"
        "## 适用对象\n"
        "说明当前难度下的题目侧重点。\n\n"
        "## 覆盖知识点\n"
        "列出来自选中文档的 5-8 个知识点。\n\n"
        "## 题目\n\n"
        "### 一、选择题\n"
        "至少 5 题。每题包含：题干、A/B/C/D 选项、正确答案、解析、来源文档。\n\n"
        "### 二、填空题\n"
        "至少 3 题。每题包含：题干、答案、解析、来源文档。\n\n"
        "### 三、简答题\n"
        "至少 3 题。每题包含：题干、参考答案、评分要点、来源文档。\n\n"
        "### 四、计算 / 推导题\n"
        "至少 2 题。每题包含：题干、已知条件、解题步骤、最终答案、常见错误提醒、来源文档。\n\n"
        "## 答案汇总\n"
        "最后给一个简洁答案表。\n\n"
        "## 学习建议\n"
        "根据题目覆盖内容给复习建议。\n"
        "不要编造超出文档范围太远的内容；可以基于文档内容做合理教学化改写。"
        f"{multi_document_rules}"
    )


def build_task_response_strategy(
    task_intent: TaskIntent,
    level_key: str,
    document_count: int,
) -> str:
    if task_intent == "quiz":
        return build_quiz_response_strategy(level_key, document_count)
    if task_intent == "summary":
        return (
            "任务类型：summary。只有在用户明确要求总结时才使用该结构。"
            "请概括核心主题、章节/结构、关键定义、公式或图表、适用条件和常见考点。"
        )
    if task_intent == "explain":
        return (
            "任务类型：explain。请像导师一样分层讲解：先给直觉，再给定义/公式，"
            "再说明推导关系、应用场景和常见误区；不要改成 quiz 或普通总结。"
        )
    if task_intent == "tutoring":
        return (
            "任务类型：tutoring。请采用循序渐进教学：学习目标、前置知识、分步讲解、"
            "检查理解的小问题和下一步建议；不要改成总结清单。"
        )
    if task_intent == "compare":
        return (
            "任务类型：compare。请围绕用户指定角度比较多个文档或多个概念，"
            "说明相同点、差异点、适用条件和各自侧重点；必须标注来源。"
        )
    if task_intent == "solve":
        return (
            "任务类型：solve。请按题意、已知条件、使用公式/定理、解题步骤、最终结论、"
            "常见错误的顺序回答；信息不足时说明缺少什么。"
        )
    if task_intent == "flashcards":
        return (
            "任务类型：flashcards。请生成记忆卡片，每张包含正面问题、背面答案、"
            "易错点和来源文档；覆盖核心概念、公式和适用条件。"
        )
    return (
        "任务类型：study_plan。请生成学习计划，包含目标、阶段安排、每阶段任务、"
        "练习建议、复习节奏和检查点；必须基于文档内容。"
    )


def build_chat_query(question: str, document_id: str, task_intent: TaskIntent) -> str:
    if task_intent == "summary":
        return (
            f"请总结当前文档 {document_id} 的核心主题、章节结构、主要定义、关键公式、"
            "图像、表格、公式内容、适用条件和常见考点。"
            "\nInclude these retrieval hints: definition, formula, image, equation, "
            "summary, key concepts."
            f"\n用户原始问题：{question}"
        )
    retrieval_guidance = {
        "quiz": (
            f"请在当前文档 {document_id} 中检索生成 quiz 所需的核心概念、公式、例题、"
            "图像、应用条件、常见误区和可出题知识点。不要把任务改写成总结。"
        ),
        "explain": (
            f"请在当前文档 {document_id} 中检索讲解所需的定义、核心直觉、关键公式、"
            "推导关系、图像/表格和例子。"
        ),
        "tutoring": (
            f"请在当前文档 {document_id} 中检索循序渐进教学所需的前置概念、关键步骤、"
            "例子、图像和易错点。"
        ),
        "compare": (
            f"请在当前文档 {document_id} 中检索可用于比较的概念、方法、条件、结论、"
            "图像/表格和侧重点。"
        ),
        "solve": (
            f"请在当前文档 {document_id} 中检索解题所需的已知条件、公式、定理、步骤、"
            "例题和边界条件。"
        ),
        "flashcards": (
            f"请在当前文档 {document_id} 中检索生成记忆卡片所需的核心概念、公式、"
            "适用条件和易混点。"
        ),
        "study_plan": (
            f"请在当前文档 {document_id} 中检索制定学习计划所需的章节结构、核心概念、"
            "先修关系、练习主题和复习重点。"
        ),
    }
    return f"{retrieval_guidance[task_intent]}\n用户原始问题：{question}"


def build_document_prompt(
    question: str,
    document_id: str,
    level_key: str,
    level_prompt: str,
    task_intent: TaskIntent,
) -> str:
    scoped_context = (
        f"当前选中文档是：{document_id}。\n"
        "检索范围：document。\n"
        "你只能基于该文档的检索结果回答。\n"
        "如果检索结果不足，请说明“当前文档中没有足够信息”，不要引用其他文档。"
    )
    inline_image_rules = (
        "回答必须使用 Markdown。\n"
        "如果上下文提供了图片引用 ID，且某段解释依赖对应图片，请在该段后插入真实图片 ID，例如 [[image:012345abcdef012345abcdef012345abcdef012345abcdef012345abcdef0123]]。\n"
        "每张图最多引用一次；不要在答案末尾堆全部图片；不要输出本地文件路径；"
        "不要输出字面量 image_id；不要编造 image_id；只能使用上下文明确提供的真实图片 ID；无关图片可以不引用。"
    )
    return (
        f"{scoped_context}\n\n"
        f"任务意图：{task_intent}\n\n"
        f"用户问题与检索需求：{question}\n\n"
        f"回答深度：{level_guidance(level_key, level_prompt, task_intent)}\n\n"
        f"回答策略：{build_task_response_strategy(task_intent, level_key, 1)}\n\n"
        f"图片引用规则：{inline_image_rules}"
    )


def answer_needs_fallback(answer: str) -> bool:
    if not answer.strip():
        return True
    normalized = answer.lower()
    markers = (
        "document chunks 为空",
        "document chunks",
        "document chunks is empty",
        "reference document list",
        "reference document list 为空",
        "context 中不包含任何实际",
        "context 不包含任何实际",
        "不包含任何实际的 pdf 文档内容",
        "无法获取该 pdf 的总体文本",
        "无法获取 pdf 文档内容",
        "无法获取该 pdf 文档内容",
        "no document chunks",
        "empty context",
        "context empty",
    )
    return any(marker in normalized for marker in markers)


def read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read JSON storage file %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def content_from_full_doc(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return ""
    for key in ("content", "text", "full_doc", "raw_content"):
        text = value.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def collect_parser_content_for_document(document_id: str, max_chars: int) -> str:
    try:
        path = content_list_path(document_path(document_id))
    except HTTPException:
        return ""
    if not path:
        return ""

    parts: list[str] = []
    try:
        items = load_content_list(path)
    except Exception as exc:
        logger.warning("Failed to load parser content_list for %s: %s", document_id, exc)
        return ""

    for item in items:
        item_type = str(item.get("type", "")).lower()
        if item_type not in {"text", "equation", "table", "image"}:
            continue
        text = first_non_empty_text(
            item,
            "text",
            "table_body",
            "caption",
            "image_caption",
            "table_caption",
            "equation",
            "latex",
        )
        if text:
            parts.append(text)
        if sum(len(part) for part in parts) >= max_chars:
            break
    return "\n\n".join(parts)[:max_chars]


def collect_direct_summary_context(
    document_id: str, max_chars: int = 24000
) -> tuple[str, str, list[str]]:
    filename = safe_document_id(document_id)
    storage_dir = document_storage_dir(filename)
    warnings: list[str] = []

    if not storage_dir.exists():
        parser_context = collect_parser_content_for_document(filename, max_chars)
        return parser_context, "parser_content_list", warnings

    doc_statuses = read_json_dict(storage_dir / "kv_store_doc_status.json")
    text_chunks = read_json_dict(storage_dir / "kv_store_text_chunks.json")
    current_doc_id = ""
    current_doc_status: dict[str, Any] | None = None
    for doc_id, status in doc_statuses.items():
        if isinstance(status, dict) and same_document_file(status.get("file_path"), filename):
            current_doc_id = str(doc_id)
            current_doc_status = status
            break

    full_docs = read_json_dict(storage_dir / "kv_store_full_docs.json")
    if current_doc_id:
        full_doc_text = content_from_full_doc(full_docs.get(current_doc_id))
        if full_doc_text:
            return full_doc_text[:max_chars], "full_docs", warnings

    if current_doc_status:
        chunk_parts: list[str] = []
        for chunk_id in current_doc_status.get("chunks_list", []):
            chunk = text_chunks.get(str(chunk_id))
            if not isinstance(chunk, dict):
                continue
            if chunk.get("file_path") and not same_document_file(chunk.get("file_path"), filename):
                warnings.append(f"skipped non-current text_chunk {chunk_id}")
                continue
            content = str(chunk.get("content", "")).strip()
            if content:
                chunk_parts.append(content)
            if sum(len(part) for part in chunk_parts) >= max_chars:
                break
        if chunk_parts:
            return "\n\n".join(chunk_parts)[:max_chars], "text_chunks", warnings

    parser_context = collect_parser_content_for_document(filename, max_chars)
    if parser_context.strip():
        return parser_context, "parser_content_list", warnings

    vdb_records, vdb_warnings = vdb_records_by_file(storage_dir, filename)
    warnings.extend(vdb_warnings)
    vdb_parts: list[str] = []
    for record in vdb_records:
        content = str(record.get("content", "")).strip()
        if content:
            vdb_parts.append(content)
        if sum(len(part) for part in vdb_parts) >= max_chars:
            break
    return "\n\n".join(vdb_parts)[:max_chars], "vdb_chunks", warnings


async def direct_summary_fallback(
    rag: RAGAnything,
    question: str,
    document_id: str,
    task_intent: TaskIntent,
    level_key: str,
    level_prompt: str,
) -> tuple[str, str]:
    context, source_path, warnings = collect_direct_summary_context(document_id)
    if not context.strip():
        return "", "none"
    if warnings:
        logger.warning(
            "Direct intent fallback skipped unsafe fragments document_id=%s detected_intent=%s warnings=%s",
            document_id,
            task_intent,
            "; ".join(warnings[:8]),
        )

    fallback_prompt = (
        f"当前选中文档是：{document_id}。\n"
        f"任务意图：{task_intent}\n"
        f"用户问题：{question}\n\n"
        "以下摘录只来自当前文档。请只基于这些摘录回答；"
        "如果摘录不足以回答，请明确说明信息不足，不要引用其他文档或编造来源。\n\n"
        f"回答深度：{level_guidance(level_key, level_prompt, task_intent)}\n\n"
        f"回答策略：{build_task_response_strategy(task_intent, level_key, 1)}\n\n"
        f"当前文档摘录：\n{context}\n\n"
        "请严格按任务意图输出，不要默认改成文档总结。"
    )
    answer = await rag.aquery(
        fallback_prompt,
        mode="bypass",
        system_prompt=build_task_system_prompt(task_intent),
        vlm_enhanced=False,
        enable_rerank=False,
    )
    return response_content_to_text(answer), f"direct_{task_intent}_fallback:{source_path}"


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


def is_relative_to_path(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def normalized_path_key(path: Path | str) -> str:
    try:
        resolved = Path(path).resolve(strict=False)
    except (OSError, RuntimeError):
        resolved = Path(path)
    return os.path.normcase(str(resolved))


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


def document_status_details(pdf_path: Path) -> tuple[DocumentState, list[str]]:
    document_id = safe_document_id(pdf_path.name)
    if is_document_processing(document_id):
        return "indexing", []
    if process_failure_path(document_id).exists():
        return "failed", []

    parsed = content_list_path(pdf_path) is not None
    if not parsed:
        return "uploaded", []

    storage_dir = document_storage_dir(document_id)
    if not storage_dir.exists():
        return "parsed", []

    readiness = document_storage_readiness(document_id)
    return readiness.status, readiness.warnings


def document_status(pdf_path: Path) -> DocumentState:
    status, _warnings = document_status_details(pdf_path)
    return status


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


def build_rerank_model_func():
    if not rerank_requested():
        update_rerank_runtime_state(
            enabled=False,
            model_loaded=False,
            last_error=None,
            reason="disabled_by_config",
        )
        return None

    model_name = rerank_model_name()
    if not model_name:
        update_rerank_runtime_state(
            enabled=False,
            model_loaded=False,
            last_error="missing_model",
            reason="missing_model",
        )
        logger.warning(
            "ENABLE_RERANK=true but RERANK_MODEL is empty; query enable_rerank will be false."
        )
        return None

    binding = rerank_binding_name()
    provider = rerank_provider_name()
    binding_host = rerank_base_url()
    binding_api_key = rerank_binding_api_key(provider)
    configured_top_n = rerank_top_n()

    if provider is None:
        update_rerank_runtime_state(
            enabled=False,
            model_loaded=False,
            last_error=f"unsupported_binding:{binding}",
            reason="unsupported_binding",
        )
        logger.warning(
            "Unsupported RERANK_BINDING=%r; query enable_rerank will be false.",
            binding,
        )
        return None

    if not binding_api_key:
        update_rerank_runtime_state(
            enabled=False,
            model_loaded=False,
            last_error="missing_api_key",
            reason="missing_api_key",
        )
        logger.warning(
            "Rerank requested but no API key is available for provider=%s model=%s; query enable_rerank will be false.",
            provider,
            model_name,
        )
        return None

    try:
        from lightrag.rerank import ali_rerank, cohere_rerank, jina_rerank
    except Exception as exc:
        update_rerank_runtime_state(
            enabled=False,
            model_loaded=False,
            last_error=str(exc),
            reason="provider_init_failed",
        )
        logger.warning("Failed to load LightRAG rerank functions; rerank disabled: %s", exc)
        return None

    rerank_functions = {
        "aliyun": ali_rerank,
        "jina": jina_rerank,
        "cohere": cohere_rerank,
    }
    selected_rerank_func = rerank_functions.get(provider)
    if selected_rerank_func is None:
        update_rerank_runtime_state(
            enabled=False,
            model_loaded=False,
            last_error=f"unsupported_binding:{binding}",
            reason="unsupported_binding",
        )
        logger.warning(
            "Unsupported RERANK_BINDING=%r; rerank disabled for queries.", binding
        )
        return None

    async def rerank_func(
        query: str,
        documents: list,
        top_n: int | None = None,
        extra_body: dict | None = None,
    ):
        effective_top_n = configured_top_n or top_n
        started = time.perf_counter()
        kwargs = {
            "query": query,
            "documents": documents,
            "top_n": effective_top_n,
            "api_key": binding_api_key,
            "model": model_name,
        }
        if binding_host:
            kwargs["base_url"] = binding_host
        if provider == "cohere":
            kwargs["enable_chunking"] = as_bool(
                os.getenv("RERANK_ENABLE_CHUNKING"), False
            )
            kwargs["max_tokens_per_doc"] = as_int(
                os.getenv("RERANK_MAX_TOKENS_PER_DOC"), 4096
            )
        try:
            result = await selected_rerank_func(**kwargs, extra_body=extra_body)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            update_rerank_runtime_state(
                enabled=True,
                model_loaded=True,
                last_error=str(exc),
                reason="provider_call_failed",
            )
            logger.warning(
                "Rerank execution failed; fallback to original retrieval chunks. provider=%s model=%s retrieved_chunks_before_rerank=%s retrieved_chunks_after_rerank=%s rerank_top_n=%s rerank_elapsed=%.3fs rerank_applied=false rerank_failed_fallback=true error=%s",
                provider,
                model_name,
                len(documents),
                len(documents),
                effective_top_n or "QueryParam.chunk_top_k",
                elapsed,
                exc,
            )
            raise

        elapsed = time.perf_counter() - started
        result_count = len(result) if isinstance(result, list) else 0
        update_rerank_runtime_state(
            enabled=True,
            model_loaded=True,
            last_error=None,
            reason=None,
        )
        logger.info(
            "Rerank execution completed provider=%s model=%s retrieved_chunks_before_rerank=%s retrieved_chunks_after_rerank=%s rerank_top_n=%s rerank_elapsed=%.3fs rerank_applied=%s rerank_failed_fallback=false",
            provider,
            model_name,
            len(documents),
            result_count,
            effective_top_n or "QueryParam.chunk_top_k",
            elapsed,
            result_count > 0,
        )
        return result

    previous_status = current_rerank_runtime_status()
    previous_reason = previous_status.get("reason")
    logger.info(
        "Rerank model loaded: provider=%s model=%s base_url=%s top_n=%s rerank_model_func=not None fallback=original retrieval chunks",
        provider,
        model_name,
        binding_host or "provider_default",
        configured_top_n or "QueryParam.chunk_top_k",
    )
    update_rerank_runtime_state(
        enabled=True,
        model_loaded=True,
        last_error=previous_status.get("last_error")
        if previous_reason == "provider_call_failed"
        else None,
        reason=previous_reason if previous_reason == "provider_call_failed" else None,
    )
    return rerank_func


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


def build_rag(
    document_id: str | None = None, working_dir: Path | None = None
) -> RAGAnything:
    llm_func, vision_func = build_model_functions()
    rerank_model_func = build_rerank_model_func()
    normalized_document_id = safe_document_id(document_id) if document_id else None
    if working_dir is not None:
        effective_working_dir = working_dir
    elif normalized_document_id:
        effective_working_dir = document_storage_dir(normalized_document_id)
    else:
        effective_working_dir = RAG_STORAGE_DIR
    workspace = (
        lightrag_workspace(normalized_document_id) if normalized_document_id else None
    )
    enable_multimodal = multimodal_enabled()
    enable_image_processing = enable_multimodal and as_bool(
        os.getenv("ENABLE_IMAGE_PROCESSING"), True
    )
    enable_table_processing = enable_multimodal and as_bool(
        os.getenv("ENABLE_TABLE_PROCESSING"), True
    )
    enable_equation_processing = enable_multimodal and as_bool(
        os.getenv("ENABLE_EQUATION_PROCESSING"), True
    )

    if enable_image_processing and not os.getenv("QWEN_API_KEY"):
        logger.warning(
            "ENABLE_MULTIMODAL=true and ENABLE_IMAGE_PROCESSING=true, but "
            "QWEN_API_KEY is missing; image processing is disabled."
        )
        enable_image_processing = False

    config_kwargs = {
        "working_dir": str(effective_working_dir),
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
    lightrag_kwargs: dict[str, Any] = {}
    if normalized_document_id and workspace:
        lightrag_kwargs["working_dir"] = str(lightrag_working_root())
        lightrag_kwargs["workspace"] = workspace
    if rerank_model_func is not None:
        lightrag_kwargs["rerank_model_func"] = rerank_model_func
    rerank_status = current_rerank_runtime_status()
    logger.info(
        "build_rag rerank wiring document_id=%s rerank_requested=%s rerank_enabled=%s rerank_model_func=%s rerank_provider=%s rerank_model=%s",
        normalized_document_id,
        rerank_status["requested"],
        rerank_status["enabled"],
        "not None" if rerank_model_func is not None else "None",
        rerank_status.get("provider") or "none",
        rerank_status.get("model") or "none",
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_func,
        vision_model_func=vision_func
        if enable_multimodal and enable_image_processing
        else None,
        embedding_func=build_embedding_func(),
        lightrag_kwargs=lightrag_kwargs,
    )
    setattr(rag, "_rerank_model_func_available", rerank_model_func is not None)
    setattr(rag, "_rerank_provider", rerank_status.get("provider"))
    setattr(rag, "_rerank_model", rerank_status.get("model"))
    logger.info(
        "build_rag_created document_id=%s safe_document_key=%s working_dir=%s lightrag_working_dir=%s lightrag_workspace=%s rag_id=%s lightrag_id=%s",
        normalized_document_id,
        safe_document_key(normalized_document_id) if normalized_document_id else None,
        effective_working_dir,
        lightrag_kwargs.get("working_dir", str(effective_working_dir)),
        lightrag_kwargs.get("workspace"),
        id(rag),
        id(rag.lightrag) if rag.lightrag is not None else None,
    )
    return rag


async def get_rag(document_id: str | None = None) -> RAGAnything:
    if not document_id:
        raise ValueError("document_id is required for document-scoped retrieval.")

    normalized_document_id = safe_document_id(document_id)
    storage_dir = document_storage_dir(normalized_document_id)
    cache_key = document_cache_key(normalized_document_id)
    if cache_key not in _rag_instances:
        async with _rag_lock:
            if cache_key not in _rag_instances:
                set_lightrag_default_workspace(lightrag_workspace(normalized_document_id))
                logger.info(
                    "Creating document-scoped RAG instance document_id=%s safe_document_key=%s storage_dir=%s lightrag_working_dir=%s lightrag_workspace=%s",
                    normalized_document_id,
                    cache_key,
                    storage_dir,
                    lightrag_working_root(),
                    lightrag_workspace(normalized_document_id),
                )
                _rag_instances[cache_key] = build_rag(
                    document_id=normalized_document_id,
                    working_dir=storage_dir,
                )
    return _rag_instances[cache_key]


def rag_rerank_model_func_available(rag: RAGAnything) -> bool:
    lightrag = getattr(rag, "lightrag", None)
    if lightrag is not None and callable(getattr(lightrag, "rerank_model_func", None)):
        return True
    lightrag_kwargs = getattr(rag, "lightrag_kwargs", {})
    if isinstance(lightrag_kwargs, dict) and callable(
        lightrag_kwargs.get("rerank_model_func")
    ):
        return True
    return bool(getattr(rag, "_rerank_model_func_available", False))


def build_rerank_query_config(rag: RAGAnything) -> dict[str, Any]:
    status = current_rerank_runtime_status()
    requested = bool(status["requested"])
    model = status.get("model")
    provider = status.get("provider")
    model_func_available = rag_rerank_model_func_available(rag)
    enabled = bool(requested and model and model_func_available)
    if not requested:
        label = "disabled"
    elif enabled:
        label = "enabled"
    else:
        label = "requested_but_unavailable"
    reason = None if enabled else status.get("reason")
    if requested and not enabled and not reason:
        reason = "rerank_model_func_unavailable"
    return {
        "requested": requested,
        "enabled": enabled,
        "label": label,
        "model": model,
        "provider": provider,
        "model_func_available": model_func_available,
        "reason": reason,
        "top_n": status.get("top_n"),
        "last_error": status.get("last_error"),
    }


def knowledge_base_not_ready_response(
    *,
    message: str = "请先解析/更新知识库。",
    error: str = "knowledge_base_not_ready",
    document_status: DocumentState | None = None,
    readiness_warnings: list[str] | None = None,
) -> JSONResponse:
    content: dict[str, Any] = {
        "error": error,
        "message": message,
    }
    if document_status:
        content["document_status"] = document_status
    if readiness_warnings:
        content["readiness_warnings"] = readiness_warnings
    return JSONResponse(
        status_code=409,
        content=content,
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


async def persist_lightrag_storages(rag: RAGAnything) -> None:
    if rag.lightrag is None:
        return
    insert_done = getattr(rag.lightrag, "_insert_done", None)
    if callable(insert_done):
        await insert_done()
        return

    storage_names = (
        "full_docs",
        "doc_status",
        "text_chunks",
        "full_entities",
        "full_relations",
        "entity_chunks",
        "relation_chunks",
        "llm_response_cache",
        "entities_vdb",
        "relationships_vdb",
        "chunks_vdb",
        "chunk_entity_relation_graph",
    )
    await asyncio.gather(
        *[
            getattr(rag.lightrag, name).index_done_callback()
            for name in storage_names
            if getattr(rag.lightrag, name, None) is not None
        ]
    )


class DocumentSummary(BaseModel):
    id: str
    name: str
    size: int
    status: DocumentState
    readiness_warnings: list[str] = Field(default_factory=list)


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
    readiness_warnings: list[str] = Field(default_factory=list)
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
    document_ids: list[str] | None = None
    level: str = "undergraduate"
    mode: str = "hybrid"
    vlm_enhanced: bool | Literal["auto"] = "auto"


class ChatDocumentUsed(BaseModel):
    document_id: str
    name: str
    status: DocumentState


class ChatPartialFailure(BaseModel):
    document_id: str
    name: str | None = None
    error: str


class ImageAssetPublic(BaseModel):
    image_id: str
    document_id: str
    document_name: str
    url: str
    filename: str | None = Field(default=None, exclude=True)
    caption: str | None = None
    page: int | None = None
    bbox: Any | None = None
    source_type: str | None = None
    relevance_reason: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    related_images: list[ImageAssetPublic] = Field(default_factory=list)
    inline_image_refs: list[str] = Field(default_factory=list)
    documents_used: list[ChatDocumentUsed] = Field(default_factory=list)
    partial_failures: list[ChatPartialFailure] = Field(default_factory=list)


class CacheResponse(BaseModel):
    ok: bool


class WarmupResponse(BaseModel):
    ok: bool
    document: DocumentSummary
    storage_dir: str
    warmup_seconds: float


class HealthResponse(BaseModel):
    ok: bool
    has_api_key: bool
    mineru_backend: str
    mineru_device: str


class RAGRetrievalStatus(BaseModel):
    default_mode: str
    rerank_requested: bool
    rerank_enabled: bool
    rerank_model: str | None
    rerank_provider: str | None
    rerank_model_loaded: bool
    rerank_last_error: str | None = None
    reason: str | None = None
    rerank_top_n: int | None = None


class RAGMultimodalStatus(BaseModel):
    enabled: bool
    image_processing: bool
    table_processing: bool
    equation_processing: bool
    formula_processing: bool
    vlm_model: str | None = None


class RAGEmbeddingStatus(BaseModel):
    model: str


class RAGStatusResponse(BaseModel):
    parser: str
    parser_output_dir: str
    retrieval: RAGRetrievalStatus
    multimodal: RAGMultimodalStatus
    embedding: RAGEmbeddingStatus
    embedding_model: str
    vector_store: str
    graph_store: str
    retrieval_mode: str
    rerank_requested: bool
    rerank_enabled: bool
    rerank_model: str | None
    rerank_binding: str | None
    multimodal_enabled: bool
    citation_status: str
    sources_status: str


def summarize_document(pdf_path: Path) -> DocumentSummary:
    status, warnings = document_status_details(pdf_path)
    return DocumentSummary(
        id=pdf_path.name,
        name=pdf_path.name,
        size=pdf_path.stat().st_size,
        status=status,
        readiness_warnings=warnings,
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


def normalize_chat_document_ids(request: ChatRequest) -> list[str]:
    raw_document_ids = request.document_ids or []
    has_document_ids = any(str(document_id).strip() for document_id in raw_document_ids)
    if not has_document_ids:
        raw_document_ids = [request.document_id] if request.document_id else []

    document_ids: list[str] = []
    seen: set[str] = set()
    for raw_document_id in raw_document_ids:
        if not raw_document_id:
            continue
        document_id = safe_document_id(raw_document_id)
        if not document_id or document_id in seen:
            continue
        seen.add(document_id)
        document_ids.append(document_id)
    return document_ids


def validate_chat_document_contexts(
    document_ids: list[str],
) -> list[DocumentChatContext] | JSONResponse:
    not_ready_documents: list[dict[str, Any]] = []
    contexts: list[DocumentChatContext] = []

    for document_id in document_ids:
        pdf_path = document_path(document_id)
        status = document_status(pdf_path)
        storage_dir = document_storage_dir(document_id)
        readiness = document_storage_readiness(document_id)
        logger.info(
            "Chat document validation document_id=%s storage_dir=%s document_status=%s ready_for_chat=%s",
            document_id,
            storage_dir,
            status,
            readiness.ready,
        )
        context = DocumentChatContext(
            document_id=document_id,
            name=pdf_path.name,
            status=status,
            storage_dir=storage_dir,
            readiness=readiness,
        )
        contexts.append(context)
        if is_document_processing(document_id):
            not_ready_documents.append(
                {
                    "document_id": document_id,
                    "name": pdf_path.name,
                    "status": "indexing",
                    "readiness_warnings": ["document is processing"],
                    "storage_dir": str(storage_dir),
                }
            )
            continue
        if status != "ready_for_chat" or not readiness.ready:
            not_ready_documents.append(
                {
                    "document_id": document_id,
                    "name": pdf_path.name,
                    "status": status if status != "ready_for_chat" else readiness.status,
                    "readiness_warnings": readiness.warnings,
                    "storage_dir": str(storage_dir),
                }
            )

    if not_ready_documents:
        names = "、".join(str(item["name"]) for item in not_ready_documents)
        return JSONResponse(
            status_code=409,
            content={
                "error": "knowledge_base_not_ready",
                "message": f"以下文档未就绪：{names}",
                "not_ready_documents": not_ready_documents,
            },
        )

    return contexts


async def query_single_document(
    request: ChatRequest,
    context: DocumentChatContext,
    task_intent: TaskIntent,
    level_key: str,
    level_prompt: str,
) -> DocumentAnswer:
    timings: dict[str, float] = {}
    document_id = context.document_id
    storage_dir = context.storage_dir

    lock = await document_lock(document_id)
    async with lock:
        validate_started = time.perf_counter()
        if is_document_processing(document_id):
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "document_processing",
                    "message": "当前文档正在索引，请稍后再提问。",
                    "document_id": document_id,
                },
            )

        pdf_path = document_path(document_id)
        status = document_status(pdf_path)
        readiness = document_storage_readiness(document_id)
        logger.info(
            "Chat retrieval scope=document document_id=%s storage_dir=%s document_status=%s ready_for_chat=%s",
            document_id,
            storage_dir,
            status,
            readiness.ready,
        )
        if status != "ready_for_chat" or not readiness.ready:
            logger.warning(
                "Document-scoped storage is not ready for chat document_id=%s storage_dir=%s document_status=%s warnings=%s",
                document_id,
                storage_dir,
                status,
                "; ".join(readiness.warnings),
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "knowledge_base_not_ready",
                    "message": "请先解析/更新知识库。",
                    "document_id": document_id,
                    "document_status": status
                    if status != "ready_for_chat"
                    else readiness.status,
                    "readiness_warnings": readiness.warnings,
                },
            )
        timings["validate_document"] = time.perf_counter() - validate_started

        load_started = time.perf_counter()
        rag = await get_rag(document_id)
        configure_vlm_image_registry(rag, [document_id])

        init_result = await rag._ensure_lightrag_initialized()
        if not init_result.get("success") or rag.lightrag is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "knowledge_base_not_ready",
                    "message": "请先解析/更新知识库。",
                    "document_id": document_id,
                    "document_status": "partial_success",
                },
            )
        timings["load_rag"] = time.perf_counter() - load_started

        rewrite_started = time.perf_counter()
        user_question = request.question.strip()
        query = build_chat_query(user_question, document_id, task_intent)
        prompt = build_document_prompt(
            query,
            document_id,
            level_key,
            level_prompt,
            task_intent,
        )
        prompt_template_used = prompt_template_name(task_intent, 1)
        timings["query_rewrite"] = time.perf_counter() - rewrite_started

        rerank_query_config = build_rerank_query_config(rag)
        effective_enable_rerank = bool(rerank_query_config["enabled"])
        requested_vlm_enhanced = request.vlm_enhanced
        logger.info(
            "Chat retrieval config: document_id=%s mode=%s rerank=%s rerank_requested=%s rerank_enabled=%s rerank_model=%s rerank_provider=%s rerank_model_func=%s rerank_top_n=%s rerank_reason=%s vlm_enhanced=%s detected_intent=%s level=%s prompt_template=%s",
            document_id,
            request.mode,
            rerank_query_config["label"],
            rerank_query_config["requested"],
            rerank_query_config["enabled"],
            rerank_query_config["model"] or "none",
            rerank_query_config["provider"] or "none",
            "not None" if rerank_query_config["model_func_available"] else "None",
            rerank_query_config["top_n"] or "QueryParam.chunk_top_k",
            rerank_query_config["reason"] or "none",
            requested_vlm_enhanced,
            task_intent,
            level_key,
            prompt_template_used,
        )
        fallback_used = False
        answer_source_path = "vlm"
        answer = ""
        query_started = time.perf_counter()
        try:
            primary_query_started = time.perf_counter()
            clear_vlm_image_tracking(rag)
            query_kwargs: dict[str, Any] = {
                "mode": request.mode,
                "system_prompt": build_task_system_prompt(task_intent),
                "enable_rerank": effective_enable_rerank,
            }
            if effective_enable_rerank and rerank_query_config["top_n"]:
                query_kwargs["chunk_top_k"] = rerank_query_config["top_n"]
            if requested_vlm_enhanced != "auto":
                query_kwargs["vlm_enhanced"] = requested_vlm_enhanced
            logger.info(
                "Chat query params: document_id=%s mode=%s enable_rerank=%s rerank_model_func_available=%s chunk_top_k=%s vlm_enhanced=%s",
                document_id,
                query_kwargs["mode"],
                query_kwargs["enable_rerank"],
                rerank_query_config["model_func_available"],
                query_kwargs.get("chunk_top_k", "QueryParam.default"),
                query_kwargs.get("vlm_enhanced", "auto"),
            )
            answer = await rag.aquery(
                prompt,
                **query_kwargs,
            )
            answer = response_content_to_text(answer)
            primary_query_elapsed = time.perf_counter() - primary_query_started
            timings["vlm_enhanced_query"] = (
                primary_query_elapsed if requested_vlm_enhanced is not False else 0.0
            )
            if answer_needs_fallback(answer):
                fallback_used = True
                answer_source_path = "text_fallback"
                logger.warning(
                    "VLM enhanced chat query returned empty or context-empty answer; retrying with vlm_enhanced=False. document_id=%s",
                    document_id,
                )
                fallback_started = time.perf_counter()
                try:
                    answer = response_content_to_text(
                        await rag.aquery(
                            prompt,
                            mode=request.mode,
                            system_prompt=build_task_system_prompt(task_intent),
                            vlm_enhanced=False,
                            enable_rerank=effective_enable_rerank,
                            **(
                                {"chunk_top_k": rerank_query_config["top_n"]}
                                if effective_enable_rerank
                                and rerank_query_config["top_n"]
                                else {}
                            ),
                        )
                    )
                except TypeError as error:
                    logger.warning(
                        "vlm_enhanced=False fallback is unsupported document_id=%s error=%s",
                        document_id,
                        error,
                    )
                    answer = ""
                timings["fallback_query"] = time.perf_counter() - fallback_started
            if answer_needs_fallback(answer):
                fallback_used = True
                logger.warning(
                    "Text fallback still returned empty/context-empty answer; trying direct intent fallback. document_id=%s detected_intent=%s",
                    document_id,
                    task_intent,
                )
                fallback_started = time.perf_counter()
                answer, answer_source_path = await direct_summary_fallback(
                    rag,
                    query,
                    document_id,
                    task_intent,
                    level_key,
                    level_prompt,
                )
                timings["fallback_query"] = (
                    timings.get("fallback_query", 0.0)
                    + time.perf_counter()
                    - fallback_started
                )
        except ValueError as error:
            if "No LightRAG instance available" in str(error):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "knowledge_base_not_ready",
                        "message": "请先解析/更新知识库。",
                        "document_id": document_id,
                        "document_status": "partial_success",
                    },
                ) from error
            raise
        timings["aquery_total"] = time.perf_counter() - query_started
        post_query_rerank_status = current_rerank_runtime_status()
        logger.info(
            "Chat rerank verification: document_id=%s enable_rerank_passed=%s rerank_model_func_available=%s rerank_top_n=%s rerank_last_error=%s",
            document_id,
            effective_enable_rerank,
            rerank_query_config["model_func_available"],
            rerank_query_config["top_n"] or "QueryParam.chunk_top_k",
            post_query_rerank_status.get("last_error") or "none",
        )

        if answer_needs_fallback(answer):
            logger.warning(
                "Document-scoped retrieval could not produce a usable answer. document_id=%s storage_dir=%s document_status=%s ready_for_chat=%s fallback_used=%s answer_source_path=%s",
                document_id,
                storage_dir,
                status,
                readiness.ready,
                fallback_used,
                answer_source_path,
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "knowledge_base_not_ready",
                    "message": "请先解析/更新知识库。",
                    "document_id": document_id,
                    "document_status": "partial_success",
                },
            )

    return DocumentAnswer(
        document_id=document_id,
        name=context.name,
        status=status,
        storage_dir=storage_dir,
        answer=answer,
        vlm_image_paths=current_vlm_image_paths(rag),
        fallback_used=fallback_used,
        answer_source_path=answer_source_path,
        prompt_template_used=prompt_template_used,
        timings=timings,
    )


def build_multi_document_synthesis_prompt(
    question: str,
    document_answers: list[DocumentAnswer],
    task_intent: TaskIntent,
    level_key: str,
    level_prompt: str,
) -> str:
    answer_sections = []
    for index, document_answer in enumerate(document_answers, start=1):
        answer = document_answer.answer[:8000]
        answer_sections.append(
            f"[文档 {index}: {document_answer.name} / {document_answer.document_id}]\n{answer}"
        )

    document_count = len(document_answers)
    task_strategy = build_task_response_strategy(task_intent, level_key, document_count)
    shared_rules = (
        "你正在综合多个 PDF 文档的独立检索结果。"
        "下面每一段都已经由对应文档的 document-scoped RAG 单独生成，"
        "不要引入这些段落之外的信息，也不要伪造页码、分数或引用。"
        "最终回答必须优先满足用户任务，不要默认改成文档总结。\n\n"
        f"用户问题：{question.strip()}\n\n"
        f"任务意图：{task_intent}\n\n"
        f"回答深度：{level_guidance(level_key, level_prompt, task_intent)}\n\n"
        f"回答策略：{task_strategy}\n\n"
        "多文档来源规则：覆盖所有已成功检索的文档；每个关键结论、题目或建议都要标注来源文档；"
        "如果某个问题只被部分文档覆盖，请直接说明。\n\n"
    )
    if task_intent == "quiz":
        shared_rules += (
            "请输出一份统一的多文档 quiz，而不是分别总结各文档。"
            "题目应覆盖每个文档，并包含至少 2 道跨文档综合题。"
            "跨文档题可以比较不同文档中相关概念、公式、频域分析方式、采样/频谱/系统响应关系等，"
            "但必须以文档独立回答中出现的信息为依据。\n\n"
        )
    elif task_intent == "compare":
        shared_rules += (
            "请围绕用户问题做对比综合，优先输出可扫描的对比表，再补充差异原因、适用条件和结论。\n\n"
        )
    elif task_intent == "summary":
        shared_rules += (
            "请输出多文档总结，并明确每个文档的贡献、一致点、差异点和覆盖不足。\n\n"
        )
    else:
        shared_rules += (
            "请融合多个文档的信息回答用户问题，避免机械罗列每个文档摘要；必要时指出不同文档的侧重点。\n\n"
        )
    return (
        shared_rules
        + (
        "图片引用规则：回答必须使用 Markdown；如果文档独立回答中已有真实图片 ID 引用，"
        "例如 [[image:012345abcdef012345abcdef012345abcdef012345abcdef012345abcdef0123]]，"
        "且综合回答仍然依赖该图，请把占位符保留在对应段落后；每张图最多引用一次；"
        "不要在答案末尾堆全部图片；不要输出本地文件路径；不要输出字面量 image_id；不要编造 image_id；"
        "只能使用文档独立回答里已经出现的真实图片 ID。\n\n"
        "文档独立回答：\n\n"
        )
        + "\n\n---\n\n".join(answer_sections)
    )


async def synthesize_multi_document_answer(
    question: str,
    document_answers: list[DocumentAnswer],
    task_intent: TaskIntent,
    level_key: str,
    level_prompt: str,
) -> str:
    llm_func, _vision_func = build_model_functions()
    prompt = build_multi_document_synthesis_prompt(
        question,
        document_answers,
        task_intent,
        level_key,
        level_prompt,
    )
    answer = await llm_func(
        prompt,
        system_prompt=build_task_system_prompt(task_intent),
    )
    return response_content_to_text(answer)


load_runtime_config()
log_rerank_startup_status()
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


@app.get("/api/rag/status", response_model=RAGStatusResponse)
async def rag_status() -> RAGStatusResponse:
    rerank_status = current_rerank_runtime_status()
    if (
        rerank_status["requested"]
        and rerank_status["model"]
        and not rerank_status["model_loaded"]
        and rerank_status.get("reason")
        not in {"missing_api_key", "unsupported_binding", "provider_init_failed"}
    ):
        build_rerank_model_func()
        rerank_status = current_rerank_runtime_status()

    default_mode = os.getenv("QUERY_MODE", "hybrid")
    embedding_model = os.getenv("EMBEDDING_LOCAL_MODEL", "BAAI/bge-small-zh-v1.5")
    multimodal_is_enabled = multimodal_enabled()
    return RAGStatusResponse(
        parser=os.getenv("PARSER", "mineru"),
        parser_output_dir=str(OUTPUT_DIR),
        retrieval=RAGRetrievalStatus(
            default_mode=default_mode,
            rerank_requested=bool(rerank_status["requested"]),
            rerank_enabled=bool(rerank_status["enabled"]),
            rerank_model=rerank_status.get("model"),
            rerank_provider=rerank_status.get("provider"),
            rerank_model_loaded=bool(rerank_status["model_loaded"]),
            rerank_last_error=rerank_status.get("last_error"),
            reason=rerank_status.get("reason"),
            rerank_top_n=rerank_status.get("top_n"),
        ),
        multimodal=RAGMultimodalStatus(
            enabled=multimodal_is_enabled,
            image_processing=multimodal_is_enabled
            and as_bool(os.getenv("ENABLE_IMAGE_PROCESSING"), True),
            table_processing=multimodal_is_enabled
            and as_bool(os.getenv("ENABLE_TABLE_PROCESSING"), True),
            equation_processing=multimodal_is_enabled
            and as_bool(os.getenv("ENABLE_EQUATION_PROCESSING"), True),
            formula_processing=formula_processing_enabled(),
            vlm_model=os.getenv("QWEN_VL_MODEL") or None,
        ),
        embedding=RAGEmbeddingStatus(model=embedding_model),
        embedding_model=embedding_model,
        vector_store="LightRAG JSON vector stores in rag_storage/vdb_*.json",
        graph_store="LightRAG GraphML graph in rag_storage/graph_chunk_entity_relation.graphml",
        retrieval_mode=default_mode,
        rerank_requested=bool(rerank_status["requested"]),
        rerank_enabled=bool(rerank_status["enabled"]),
        rerank_model=rerank_status.get("model"),
        rerank_binding=rerank_status.get("provider")
        if rerank_status["enabled"]
        else None,
        multimodal_enabled=multimodal_is_enabled,
        citation_status="not implemented for chat responses",
        sources_status="document preview only; chat sources are not extracted from retrieval data",
    )


@app.get("/api/documents", response_model=list[DocumentSummary])
async def list_documents() -> list[DocumentSummary]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return [summarize_document(path) for path in sorted(UPLOAD_DIR.glob("*.pdf"))]


@app.get("/api/documents/{document_id}/storage/debug")
async def debug_document_storage(document_id: str) -> dict[str, Any]:
    return document_storage_debug_payload(document_id)


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


@app.post("/api/documents/{document_id}/warmup", response_model=WarmupResponse)
async def warmup_document(document_id: str) -> WarmupResponse | JSONResponse:
    started = time.perf_counter()
    normalized_document_id = safe_document_id(document_id)
    pdf_path = document_path(normalized_document_id)
    status = document_status(pdf_path)
    if status != "ready_for_chat":
        return knowledge_base_not_ready_response(document_status=status)

    storage_dir = document_storage_dir(normalized_document_id)
    readiness = document_storage_readiness(normalized_document_id)
    if not readiness.ready:
        logger.warning(
            "Warmup skipped because document-scoped storage is not ready document_id=%s storage_dir=%s warnings=%s",
            normalized_document_id,
            storage_dir,
            "; ".join(readiness.warnings),
        )
        return knowledge_base_not_ready_response(
            document_status=readiness.status,
            readiness_warnings=readiness.warnings,
        )

    rag = await get_rag(normalized_document_id)
    init_result = await rag._ensure_lightrag_initialized()
    if not init_result.get("success") or rag.lightrag is None:
        logger.warning(
            "Warmup failed during LightRAG initialization document_id=%s storage_dir=%s error=%s",
            normalized_document_id,
            storage_dir,
            init_result.get("error"),
        )
        return knowledge_base_not_ready_response(document_status="partial_success")

    elapsed = time.perf_counter() - started
    logger.info(
        "Warmup completed document_id=%s storage_dir=%s warmup=%.3fs",
        normalized_document_id,
        storage_dir,
        elapsed,
    )
    return WarmupResponse(
        ok=True,
        document=summarize_document(pdf_path),
        storage_dir=str(storage_dir),
        warmup_seconds=elapsed,
    )


@app.get("/api/documents/{document_id}/images/{image_id}")
async def get_document_image(document_id: str, image_id: str) -> FileResponse:
    normalized_document_id = safe_document_id(document_id)
    document_path(normalized_document_id)
    asset = get_document_image_asset(normalized_document_id, image_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Image not found.")
    if not image_file_exists(asset):
        raise HTTPException(status_code=404, detail="Image file not found.")

    media_type = IMAGE_MEDIA_TYPES.get(asset.absolute_path.suffix.lower())
    if media_type is None:
        raise HTTPException(status_code=404, detail="Unsupported image type.")
    return FileResponse(
        asset.absolute_path,
        media_type=media_type,
        filename=asset.filename,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse | JSONResponse:
    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    level_key = normalize_answer_level(request.level)
    level_prompt = ANSWER_LEVELS[level_key]
    task_intent = detect_task_intent(request.question)

    validate_started = time.perf_counter()
    document_ids = normalize_chat_document_ids(request)
    prompt_template_used = prompt_template_name(task_intent, len(document_ids))
    logger.info(
        "Chat request document_ids=%s legacy_document_id=%s",
        document_ids,
        request.document_id,
    )
    logger.info(
        "Chat intent detected original_question=%r detected_intent=%s level=%s document_ids=%s prompt_template=%s",
        request.question,
        task_intent,
        level_key,
        document_ids,
        prompt_template_used,
    )
    if not document_ids:
        raise HTTPException(status_code=400, detail="请先选择文档")
    if len(document_ids) > 3:
        raise HTTPException(status_code=400, detail="最多选择 3 个文档")

    contexts_or_response = validate_chat_document_contexts(document_ids)
    if isinstance(contexts_or_response, JSONResponse):
        return contexts_or_response
    contexts = contexts_or_response
    timings["validate_document"] = time.perf_counter() - validate_started

    logger.info(
        "Chat retrieval scope=%s document_ids=%s storage_dirs=%s",
        "multi_document" if len(contexts) > 1 else "document",
        document_ids,
        [str(context.storage_dir) for context in contexts],
    )

    try:
        if len(contexts) == 1:
            document_answer = await query_single_document(
                request,
                contexts[0],
                task_intent,
                level_key,
                level_prompt,
            )
            package_started = time.perf_counter()
            related_images = select_related_images(
                [document_answer.document_id],
                request.question,
                document_answer.vlm_image_paths,
            )
            allowed_images = public_images_for_documents([document_answer.document_id])
            answer, related_images, inline_image_refs = validate_answer_images(
                document_answer.answer,
                related_images,
                allowed_images,
            )
            response = ChatResponse(
                answer=answer,
                sources=[],
                related_images=related_images,
                inline_image_refs=inline_image_refs,
                documents_used=[
                    ChatDocumentUsed(
                        document_id=document_answer.document_id,
                        name=document_answer.name,
                        status=document_answer.status,
                    )
                ],
            )
            timings["response_packaging"] = time.perf_counter() - package_started
            timings["total"] = time.perf_counter() - total_started
            logger.info(
                "Chat answer ready original_question=%r detected_intent=%s level=%s document_ids=%s prompt_template=%s document_id=%s storage_dir=%s document_status=%s ready_for_chat=%s retrieval_scope=document fallback_used=%s answer_source_path=%s",
                request.question,
                task_intent,
                level_key,
                document_ids,
                document_answer.prompt_template_used,
                document_answer.document_id,
                document_answer.storage_dir,
                document_answer.status,
                True,
                document_answer.fallback_used,
                document_answer.answer_source_path,
            )
            logger.info(
                "Chat timing document_id=%s validate_document=%.3fs load_rag=%.3fs query_rewrite=%.3fs vlm_enhanced_query=%.3fs fallback_query=%.3fs aquery_total=%.3fs response_packaging=%.3fs total=%.3fs",
                document_answer.document_id,
                timings.get("validate_document", 0.0),
                document_answer.timings.get("load_rag", 0.0),
                document_answer.timings.get("query_rewrite", 0.0),
                document_answer.timings.get("vlm_enhanced_query", 0.0),
                document_answer.timings.get("fallback_query", 0.0),
                document_answer.timings.get("aquery_total", 0.0),
                timings.get("response_packaging", 0.0),
                timings.get("total", 0.0),
            )
            return response

        document_answers: list[DocumentAnswer] = []
        partial_failures: list[ChatPartialFailure] = []
        for context in contexts:
            document_query_started = time.perf_counter()
            try:
                document_answer = await query_single_document(
                    request,
                    context,
                    task_intent,
                    level_key,
                    level_prompt,
                )
                document_answers.append(document_answer)
                logger.info(
                    "Chat multi-document per-document query document_id=%s storage_dir=%s elapsed=%.3fs answer_source_path=%s",
                    context.document_id,
                    context.storage_dir,
                    time.perf_counter() - document_query_started,
                    document_answer.answer_source_path,
                )
            except HTTPException as exc:
                logger.warning(
                    "Chat multi-document partial failure document_id=%s storage_dir=%s error=%s",
                    context.document_id,
                    context.storage_dir,
                    exc.detail,
                )
                partial_failures.append(
                    ChatPartialFailure(
                        document_id=context.document_id,
                        name=context.name,
                        error=str(exc.detail),
                    )
                )

        if not document_answers:
            return knowledge_base_not_ready_response(
                message="所选文档暂时无法生成回答，请重新解析/更新知识库。",
                document_status="partial_success",
            )

        synthesis_started = time.perf_counter()
        answer = await synthesize_multi_document_answer(
            request.question,
            document_answers,
            task_intent,
            level_key,
            level_prompt,
        )
        timings["synthesis"] = time.perf_counter() - synthesis_started
        if answer_needs_fallback(answer):
            return knowledge_base_not_ready_response(
                message="多文档综合回答为空，请稍后重试。",
                document_status="partial_success",
            )

        package_started = time.perf_counter()
        answer_document_ids = [
            document_answer.document_id for document_answer in document_answers
        ]
        vlm_image_paths = [
            image_path
            for document_answer in document_answers
            for image_path in document_answer.vlm_image_paths
        ]
        related_images = select_related_images(
            answer_document_ids,
            request.question,
            vlm_image_paths,
        )
        allowed_images = public_images_for_documents(answer_document_ids)
        answer, related_images, inline_image_refs = validate_answer_images(
            answer,
            related_images,
            allowed_images,
        )
        response = ChatResponse(
            answer=answer,
            sources=[],
            related_images=related_images,
            inline_image_refs=inline_image_refs,
            documents_used=[
                ChatDocumentUsed(
                    document_id=document_answer.document_id,
                    name=document_answer.name,
                    status=document_answer.status,
                )
                for document_answer in document_answers
            ],
            partial_failures=partial_failures,
        )
        timings["response_packaging"] = time.perf_counter() - package_started
        timings["total"] = time.perf_counter() - total_started
        logger.info(
            "Chat answer ready original_question=%r detected_intent=%s level=%s document_ids=%s prompt_template=%s retrieval_scope=multi_document storage_dirs=%s fallback_used=%s answer_source_path=%s partial_failures=%s",
            request.question,
            task_intent,
            level_key,
            [document_answer.document_id for document_answer in document_answers],
            prompt_template_used,
            [str(document_answer.storage_dir) for document_answer in document_answers],
            any(document_answer.fallback_used for document_answer in document_answers),
            "multi_document_synthesis",
            len(partial_failures),
        )
        logger.info(
            "Chat timing retrieval_scope=multi_document document_ids=%s validate_document=%.3fs synthesis=%.3fs response_packaging=%.3fs total=%.3fs",
            [document_answer.document_id for document_answer in document_answers],
            timings.get("validate_document", 0.0),
            timings.get("synthesis", 0.0),
            timings.get("response_packaging", 0.0),
            timings.get("total", 0.0),
        )
        return response
    except HTTPException as exc:
        if exc.status_code == 409 and isinstance(exc.detail, dict):
            return JSONResponse(status_code=409, content=exc.detail)
        raise


@app.delete("/api/cache", response_model=CacheResponse)
async def clear_runtime_cache() -> CacheResponse:
    async with _rag_lock:
        cached_instances = list(_rag_instances.values())
        _rag_instances.clear()
    for rag in cached_instances:
        try:
            await rag.finalize_storages()
        except Exception as exc:
            logger.warning("Failed to finalize cached RAG during cache clear: %s", exc)
    async with _document_locks_guard:
        _documents_processing.clear()
        _document_locks.clear()
    clear_all_lightrag_shared_storage()
    for path in (RAG_STORAGE_DIR, OUTPUT_DIR):
        if path.exists():
            shutil.rmtree(path)
    return CacheResponse(ok=True)
