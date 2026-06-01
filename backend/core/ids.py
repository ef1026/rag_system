from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

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


def same_document_file(file_path: Any, document_id: str) -> bool:
    if not isinstance(file_path, str) or not file_path.strip():
        return False
    return Path(file_path).name == safe_document_id(document_id)


def document_cache_key(document_id: str) -> str:
    return safe_document_key(safe_document_id(document_id))
