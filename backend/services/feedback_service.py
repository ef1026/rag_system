from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from backend.core.ids import safe_document_id
from backend.schemas import ProfileFeedbackRequest, ProfileFeedbackResponse, UserMemory
from backend.services.conversation_service import list_messages, read_conversation
from backend.services.memory_service import create_memory_candidate
from backend.services.profile_service import get_profile
from backend.storage.sqlite import metadata_connection


def record_profile_feedback(payload: ProfileFeedbackRequest) -> ProfileFeedbackResponse:
    conversation = read_conversation(payload.conversation_id)
    messages = list_messages(conversation.id)
    target_index = next(
        (index for index, message in enumerate(messages) if message.id == payload.message_id),
        -1,
    )
    if target_index < 0:
        raise HTTPException(status_code=404, detail="Feedback message not found.")
    target_message = messages[target_index]
    if target_message.role != "assistant":
        raise HTTPException(status_code=400, detail="Feedback must target an assistant message.")

    previous_user = next(
        (
            message
            for message in reversed(messages[:target_index])
            if message.role == "user"
        ),
        None,
    )
    evidence_message_ids = [
        message.id
        for message in (previous_user, target_message)
        if message is not None
    ]
    evidence = _feedback_evidence(payload, previous_user.content if previous_user else "")
    scope_type, scope_id = _scope_from_conversation(conversation)
    candidates = _feedback_candidates(
        payload=payload,
        source_conversation_id=conversation.id,
        evidence=evidence,
        scope_type=scope_type,
        scope_id=scope_id,
        evidence_message_ids=evidence_message_ids,
    )
    _record_feedback(payload, [memory.id for memory in candidates])
    return ProfileFeedbackResponse(ok=True, created_memory_candidates=candidates)


def _feedback_candidates(
    *,
    payload: ProfileFeedbackRequest,
    source_conversation_id: str,
    evidence: str,
    scope_type: str,
    scope_id: str | None,
    evidence_message_ids: list[str],
) -> list[UserMemory]:
    specs: list[dict[str, Any]] = []
    if payload.difficulty == "too_easy":
        specs.append(
            {
                "memory_type": "difficulty_preference",
                "key": "needs_deeper_explanations",
                "value": "User feedback indicates answers can be too simple; when relevant, include deeper reasoning and more technical detail.",
                "confidence": 0.68,
            }
        )
    elif payload.difficulty == "too_hard":
        specs.append(
            {
                "memory_type": "difficulty_preference",
                "key": "needs_simpler_explanations",
                "value": "User feedback indicates answers can be too hard; when relevant, simplify wording and add step-by-step explanations.",
                "confidence": 0.68,
            }
        )

    if payload.style_feedback == "needs_examples":
        specs.append(
            {
                "memory_type": "answer_style",
                "key": "needs_examples",
                "value": "User feedback asks for more concrete examples; include examples when they help answer the current question.",
                "confidence": 0.72,
            }
        )
    elif payload.style_feedback == "needs_derivation":
        specs.append(
            {
                "memory_type": "answer_style",
                "key": "needs_derivation",
                "value": "User feedback asks for derivations; include derivation steps and assumptions when the document supports them.",
                "confidence": 0.72,
            }
        )
    elif payload.style_feedback == "more_concise":
        specs.append(
            {
                "memory_type": "output_format",
                "key": "more_concise",
                "value": "User feedback asks for more concise answers; keep answers compact unless the question asks for detail.",
                "confidence": 0.7,
            }
        )

    candidates: list[UserMemory] = []
    for spec in specs:
        memory = create_memory_candidate(
            **spec,
            source_conversation_id=source_conversation_id,
            evidence=evidence,
            scope_type=scope_type,
            scope_id=scope_id,
            evidence_message_ids=evidence_message_ids,
        )
        if memory:
            candidates.append(memory)
    return candidates


def _record_feedback(
    payload: ProfileFeedbackRequest,
    created_memory_ids: list[str],
) -> None:
    profile_id = get_profile().id
    with metadata_connection() as connection:
        connection.execute(
            """
            INSERT INTO learning_feedback (
                id, profile_id, conversation_id, message_id, rating, difficulty,
                style_feedback, note, created_memory_ids_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _new_id("fb"),
                profile_id,
                payload.conversation_id,
                payload.message_id,
                _optional_text(payload.rating),
                _optional_text(payload.difficulty),
                _optional_text(payload.style_feedback),
                _optional_text(payload.note),
                json.dumps(created_memory_ids, ensure_ascii=False),
                _utc_now(),
            ),
        )


def _feedback_evidence(payload: ProfileFeedbackRequest, question: str) -> str:
    parts = []
    if payload.rating:
        parts.append(f"rating={payload.rating}")
    if payload.difficulty:
        parts.append(f"difficulty={payload.difficulty}")
    if payload.style_feedback:
        parts.append(f"style_feedback={payload.style_feedback}")
    if payload.note:
        parts.append(f"note={payload.note[:180]}")
    if question:
        parts.append(f"question={question[:180]}")
    return " | ".join(parts)


def _scope_from_conversation(conversation: Any) -> tuple[str, str | None]:
    courses = conversation.file_context.get("courses") or []
    if courses:
        return "course", str(courses[0])
    if len(conversation.document_ids) == 1:
        return "document", safe_document_id(conversation.document_ids[0])
    return "conversation", conversation.id


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
