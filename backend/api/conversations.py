from __future__ import annotations

from fastapi import APIRouter

from backend.schemas import (
    CacheResponse,
    Conversation,
    ConversationCreate,
    ConversationImportRequest,
    ConversationImportResponse,
    ConversationMessage,
    ConversationPatch,
)
from backend.services.conversation_service import (
    clear_messages,
    create_conversation,
    delete_conversation,
    import_conversations,
    list_conversations,
    list_messages,
    patch_conversation,
    read_conversation,
)

router = APIRouter()


@router.get("/api/conversations", response_model=list[Conversation])
async def read_conversations() -> list[Conversation]:
    return list_conversations()


@router.post("/api/conversations", response_model=Conversation)
async def add_conversation(payload: ConversationCreate) -> Conversation:
    return create_conversation(payload)


@router.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str) -> Conversation:
    return read_conversation(conversation_id)


@router.patch("/api/conversations/{conversation_id}", response_model=Conversation)
async def update_conversation(
    conversation_id: str,
    payload: ConversationPatch,
) -> Conversation:
    return patch_conversation(conversation_id, payload)


@router.delete("/api/conversations/{conversation_id}", response_model=CacheResponse)
async def remove_conversation(conversation_id: str) -> CacheResponse:
    delete_conversation(conversation_id)
    return CacheResponse(ok=True)


@router.get(
    "/api/conversations/{conversation_id}/messages",
    response_model=list[ConversationMessage],
)
async def read_messages(conversation_id: str) -> list[ConversationMessage]:
    return list_messages(conversation_id)


@router.delete(
    "/api/conversations/{conversation_id}/messages",
    response_model=CacheResponse,
)
async def remove_messages(conversation_id: str) -> CacheResponse:
    clear_messages(conversation_id)
    return CacheResponse(ok=True)


@router.post("/api/conversations/import", response_model=ConversationImportResponse)
async def import_existing_conversations(
    payload: ConversationImportRequest,
) -> ConversationImportResponse:
    return import_conversations(payload)
