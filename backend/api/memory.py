from __future__ import annotations

from fastapi import APIRouter

from backend.schemas import (
    CacheResponse,
    MemoryCandidateSource,
    MemoryExtractRequest,
    MemoryExtractResponse,
    UserMemory,
    UserMemoryPatch,
)
from backend.services.memory_service import (
    accept_memory,
    delete_memory,
    dismiss_memory,
    extract_memories,
    list_memory_candidate_sources,
    list_memories,
    patch_memory,
)

router = APIRouter()


@router.get("/api/memory", response_model=list[UserMemory])
async def read_memories(status: str | None = None) -> list[UserMemory]:
    return list_memories(status)


@router.get("/api/memory/sources", response_model=list[MemoryCandidateSource])
async def read_memory_candidate_sources() -> list[MemoryCandidateSource]:
    return list_memory_candidate_sources()


@router.post("/api/memory/extract", response_model=MemoryExtractResponse)
async def extract_memory_candidates(
    payload: MemoryExtractRequest,
) -> MemoryExtractResponse:
    return extract_memories(payload)


@router.patch("/api/memory/{memory_id}", response_model=UserMemory)
async def update_memory(memory_id: str, payload: UserMemoryPatch) -> UserMemory:
    return patch_memory(memory_id, payload)


@router.post("/api/memory/{memory_id}/accept", response_model=UserMemory)
async def accept_memory_candidate(memory_id: str) -> UserMemory:
    return accept_memory(memory_id)


@router.post("/api/memory/{memory_id}/dismiss", response_model=UserMemory)
async def dismiss_memory_candidate(memory_id: str) -> UserMemory:
    return dismiss_memory(memory_id)


@router.delete("/api/memory/{memory_id}", response_model=CacheResponse)
async def remove_memory(memory_id: str) -> CacheResponse:
    delete_memory(memory_id)
    return CacheResponse(ok=True)
