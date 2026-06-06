from __future__ import annotations

from fastapi import APIRouter

from backend.schemas import (
    CacheResponse,
    QuizGenerateRequest,
    QuizSession,
    QuizSubmitRequest,
    QuizSubmitResponse,
    WrongQuestion,
)
from backend.services.quiz_service import (
    delete_wrong_question,
    generate_quiz,
    list_wrong_questions,
    mark_wrong_question_reviewed,
    submit_quiz,
)

router = APIRouter()


@router.post("/api/quizzes/generate", response_model=QuizSession)
async def create_quiz(payload: QuizGenerateRequest) -> QuizSession:
    return await generate_quiz(payload)


@router.post("/api/quizzes/{session_id}/submit", response_model=QuizSubmitResponse)
async def submit_quiz_answers(
    session_id: str,
    payload: QuizSubmitRequest,
) -> QuizSubmitResponse:
    return submit_quiz(session_id, payload)


@router.get("/api/wrong-questions", response_model=list[WrongQuestion])
async def read_wrong_questions(reviewed: bool | None = None) -> list[WrongQuestion]:
    return list_wrong_questions(reviewed)


@router.post("/api/wrong-questions/{wrong_question_id}/review", response_model=WrongQuestion)
async def review_wrong_question(wrong_question_id: str) -> WrongQuestion:
    return mark_wrong_question_reviewed(wrong_question_id)


@router.delete("/api/wrong-questions/{wrong_question_id}", response_model=CacheResponse)
async def remove_wrong_question(wrong_question_id: str) -> CacheResponse:
    delete_wrong_question(wrong_question_id)
    return CacheResponse(ok=True)
