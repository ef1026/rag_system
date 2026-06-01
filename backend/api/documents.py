from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from backend.rag.readiness import document_storage_debug_payload
from backend.schemas import (
    DocumentSummary,
    ProcessResponse,
    UploadResponse,
    WarmupResponse,
)
from backend.services.document_service import list_documents as list_document_summaries
from backend.services.document_service import upload_document as upload_document_service
from backend.services.document_service import warmup_document as warmup_document_service
from backend.services.processing_service import process_document as process_document_service

router = APIRouter()


@router.get("/api/documents", response_model=list[DocumentSummary])
async def list_documents() -> list[DocumentSummary]:
    return list_document_summaries()


@router.get("/api/documents/{document_id}/storage/debug")
async def debug_document_storage(document_id: str) -> dict[str, Any]:
    return document_storage_debug_payload(document_id)


@router.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    return await upload_document_service(file)


@router.post("/api/documents/{document_id}/process", response_model=ProcessResponse)
async def process_document(document_id: str) -> ProcessResponse | JSONResponse:
    return await process_document_service(document_id)


@router.post("/api/documents/{document_id}/warmup", response_model=WarmupResponse)
async def warmup_document(document_id: str) -> WarmupResponse | JSONResponse:
    return await warmup_document_service(document_id)
