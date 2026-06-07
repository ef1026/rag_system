from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.constants import DocumentState
from backend.schemas import SourceItem

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
    sources: list[SourceItem]
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
