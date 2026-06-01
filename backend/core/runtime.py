from __future__ import annotations

import asyncio
import logging

from raganything import RAGAnything

from backend.core.ids import document_cache_key, safe_document_id

logger = logging.getLogger("api_server")

_rag_instances: dict[str, RAGAnything] = {}


_rag_lock: asyncio.Lock = asyncio.Lock()


_global_process_lock: asyncio.Lock = asyncio.Lock()


_document_locks: dict[str, asyncio.Lock] = {}


_document_locks_guard: asyncio.Lock = asyncio.Lock()


_documents_processing: set[str] = set()


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
