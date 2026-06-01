from __future__ import annotations

from fastapi import APIRouter

from backend.schemas import (
    ProfilePromptContextResponse,
    UserProfile,
    UserProfilePatch,
    UserProfilePut,
)
from backend.services.profile_service import (
    get_profile,
    get_prompt_context,
    patch_profile,
    put_profile,
)

router = APIRouter()


@router.get("/api/profile", response_model=UserProfile)
async def read_profile() -> UserProfile:
    return get_profile()


@router.put("/api/profile", response_model=UserProfile)
async def replace_profile(payload: UserProfilePut) -> UserProfile:
    return put_profile(payload)


@router.patch("/api/profile", response_model=UserProfile)
async def update_profile(payload: UserProfilePatch) -> UserProfile:
    return patch_profile(payload)


@router.get("/api/profile/prompt-context", response_model=ProfilePromptContextResponse)
async def read_profile_prompt_context(level: str | None = None) -> ProfilePromptContextResponse:
    return get_prompt_context(level)
