from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.config import UPLOAD_DIR
from backend.constants import ALLOWED_EXTENSIONS, DocumentState
from backend.core.ids import safe_document_id
from backend.core.paths import (
    content_list_path,
    document_path,
    document_storage_dir,
    load_content_list,
    process_failure_path,
)
from backend.core.runtime import is_document_processing
from backend.rag.factory import get_rag
from backend.rag.readiness import document_storage_readiness
from backend.schemas import DocumentSummary, SourceItem, UploadResponse, WarmupResponse

logger = logging.getLogger("api_server")

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


async def upload_document(file: UploadFile) -> UploadResponse:
    filename = safe_document_id(file.filename or "")
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / filename
    with target.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return UploadResponse(document=summarize_document(target))


def list_documents() -> list[DocumentSummary]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return [summarize_document(path) for path in sorted(UPLOAD_DIR.glob("*.pdf"))]
