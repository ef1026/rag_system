from __future__ import annotations

import logging
import shutil

from backend.config import OUTPUT_DIR, RAG_STORAGE_DIR
from backend.core.runtime import (
    _document_locks,
    _document_locks_guard,
    _documents_processing,
    _rag_instances,
    _rag_lock,
)
from backend.rag.storage import clear_all_lightrag_shared_storage
from backend.schemas import CacheResponse

logger = logging.getLogger("api_server")

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
