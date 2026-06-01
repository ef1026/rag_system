from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.schemas import ChatRequest, ChatResponse
from backend.services.retrieval_service import chat as chat_service

router = APIRouter()


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse | JSONResponse:
    return await chat_service(request)
