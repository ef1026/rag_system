from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import HTTPException

from backend.config import OUTPUT_DIR, RAG_STORAGE_DIR, UPLOAD_DIR
from backend.constants import ALLOWED_EXTENSIONS
from backend.core.ids import safe_document_id, safe_document_key

logger = logging.getLogger("api_server")

def document_storage_dir(document_id: str) -> Path:
    return RAG_STORAGE_DIR / "documents" / safe_document_key(document_id)


def lightrag_working_root() -> Path:
    return RAG_STORAGE_DIR / "documents"


def lightrag_workspace(document_id: str) -> str:
    return safe_document_key(document_id)


def process_failure_path(document_id: str) -> Path:
    return document_storage_dir(document_id) / ".process_failed.json"


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
