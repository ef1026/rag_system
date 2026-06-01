from __future__ import annotations

import re
from typing import Literal

ALLOWED_EXTENSIONS = {".pdf"}


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


DocumentState = Literal[
    "uploaded",
    "parsed",
    "indexing",
    "ready_for_chat",
    "partial_success",
    "failed",
]


