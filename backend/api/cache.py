from __future__ import annotations

from fastapi import APIRouter

from backend.schemas import CacheResponse
from backend.services.cache_service import clear_runtime_cache as clear_runtime_cache_service

router = APIRouter()


@router.delete("/api/cache", response_model=CacheResponse)
async def clear_runtime_cache() -> CacheResponse:
    return await clear_runtime_cache_service()
