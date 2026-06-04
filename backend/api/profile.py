from __future__ import annotations

from fastapi import APIRouter

from backend.schemas import (
    PersonalizationPreviewRequest,
    PersonalizationPreviewResponse,
    ProfileFeedbackRequest,
    ProfileFeedbackResponse,
    ProfilePromptContextResponse,
    UserProfile,
    UserProfilePatch,
    UserProfilePut,
)
from backend.services.feedback_service import record_profile_feedback
from backend.services.personalization_service import personalization_preview_response
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


@router.post(
    "/api/profile/personalization-preview",
    response_model=PersonalizationPreviewResponse,
)
async def read_personalization_preview(
    payload: PersonalizationPreviewRequest,
) -> PersonalizationPreviewResponse:
    return personalization_preview_response(
        question=payload.question or "",
        document_ids=payload.document_ids,
        level=payload.level,
        conversation_id=payload.conversation_id,
        use_profile=payload.use_profile,
        use_memory=payload.use_memory,
    )


@router.post("/api/profile/feedback", response_model=ProfileFeedbackResponse)
async def create_profile_feedback(
    payload: ProfileFeedbackRequest,
) -> ProfileFeedbackResponse:
    return record_profile_feedback(payload)
