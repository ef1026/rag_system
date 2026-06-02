from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from backend.prompts.memory_context import build_memory_prompt_context
from backend.schemas import (
    MemoryExtractRequest,
    MemoryExtractResponse,
    UserMemory,
    UserMemoryPatch,
)
from backend.services.conversation_service import list_messages, read_conversation
from backend.services.profile_service import get_profile
from backend.storage.sqlite import metadata_connection

MEMORY_COLUMNS = (
    "id",
    "profile_id",
    "memory_type",
    "key",
    "value",
    "confidence",
    "source_conversation_id",
    "evidence",
    "status",
    "sensitivity",
    "last_seen_at",
    "created_at",
    "updated_at",
)


def list_memories(status: str | None = None) -> list[UserMemory]:
    profile_id = _profile_id()
    normalized_status = _optional_text(status)
    if normalized_status:
        with metadata_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM user_memories
                WHERE profile_id = ? AND status = ?
                ORDER BY updated_at DESC
                """,
                (profile_id, normalized_status),
            ).fetchall()
    else:
        with metadata_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM user_memories
                WHERE profile_id = ?
                ORDER BY updated_at DESC
                """,
                (profile_id,),
            ).fetchall()
    return [_memory_from_row(dict(row)) for row in rows]


def patch_memory(memory_id: str, payload: UserMemoryPatch) -> UserMemory:
    memory = read_memory(memory_id)
    data = _model_dump(payload, exclude_unset=True)
    updates: dict[str, Any] = {}
    for field in (
        "memory_type",
        "key",
        "value",
        "confidence",
        "evidence",
        "status",
        "sensitivity",
    ):
        if field in data:
            updates[field] = data[field]
    if "confidence" in updates and updates["confidence"] is not None:
        updates["confidence"] = max(0.0, min(1.0, float(updates["confidence"])))
    if "value" in updates:
        updates["value"] = _required_text(updates["value"], "value")
    if "memory_type" in updates:
        updates["memory_type"] = _required_text(updates["memory_type"], "memory_type")
    if "status" in updates:
        updates["status"] = _normalize_status(updates["status"])

    if updates:
        updates["updated_at"] = _utc_now()
        assignments = ", ".join(f"{name} = ?" for name in updates)
        with metadata_connection() as connection:
            connection.execute(
                f"UPDATE user_memories SET {assignments} WHERE id = ? AND profile_id = ?",
                (*updates.values(), memory.id, memory.profile_id),
            )
        _record_event(memory.id, "updated", memory.source_conversation_id, memory.evidence)
    return read_memory(memory.id)


def accept_memory(memory_id: str) -> UserMemory:
    memory = _set_memory_status(memory_id, "active")
    _record_event(memory.id, "accepted", memory.source_conversation_id, memory.evidence)
    return memory


def dismiss_memory(memory_id: str) -> UserMemory:
    memory = _set_memory_status(memory_id, "dismissed")
    _record_event(memory.id, "dismissed", memory.source_conversation_id, memory.evidence)
    return memory


def delete_memory(memory_id: str) -> None:
    memory = read_memory(memory_id)
    with metadata_connection() as connection:
        connection.execute(
            "DELETE FROM user_memories WHERE id = ? AND profile_id = ?",
            (memory.id, memory.profile_id),
        )
    _record_event(memory.id, "deleted", memory.source_conversation_id, memory.evidence)


def extract_memories(payload: MemoryExtractRequest) -> MemoryExtractResponse:
    limit = max(1, min(int(payload.limit or 10), 25))
    conversation_ids = [_optional_text(payload.conversation_id)] if payload.conversation_id else []
    candidates = _candidate_specs(conversation_ids=[item for item in conversation_ids if item])
    created: list[UserMemory] = []
    for spec in candidates[:limit]:
        memory = _upsert_candidate(**spec)
        if memory:
            created.append(memory)
    return MemoryExtractResponse(created_count=len(created), memories=created)


def build_relevant_memory_context(
    *,
    question: str,
    document_ids: list[str],
    limit: int = 5,
) -> str:
    del document_ids
    memories = _relevant_active_memories(question, limit)
    return build_memory_prompt_context(memories)


def read_memory(memory_id: str) -> UserMemory:
    profile_id = _profile_id()
    normalized_id = _require_id(memory_id)
    with metadata_connection() as connection:
        row = connection.execute(
            "SELECT * FROM user_memories WHERE id = ? AND profile_id = ?",
            (normalized_id, profile_id),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return _memory_from_row(dict(row))


def _candidate_specs(
    *,
    conversation_ids: list[str],
) -> list[dict[str, Any]]:
    conversations = []
    if conversation_ids:
        for conversation_id in conversation_ids:
            conversations.append(read_conversation(conversation_id))
    else:
        conversations = _recent_conversations(limit=20)

    specs: list[dict[str, Any]] = []
    for conversation in conversations:
        messages = list_messages(conversation.id)
        user_messages = [message for message in messages if message.role == "user"]
        if not user_messages:
            continue
        combined = "\n".join(message.content for message in user_messages)
        evidence = _evidence_from_messages(user_messages)

        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", combined))
        if len(user_messages) >= 3 and cjk_count >= 20:
            specs.append(
                {
                    "memory_type": "preference",
                    "key": "preferred_language",
                    "value": "User often asks questions in Chinese and likely prefers Chinese explanations.",
                    "confidence": 0.72,
                    "source_conversation_id": conversation.id,
                    "evidence": evidence,
                }
            )

        quiz_hits = len(
            re.findall(
                r"\bquiz\b|practice|exercise|exam|test|出题|练习|习题|测验|考试",
                combined,
                flags=re.IGNORECASE,
            )
        )
        if quiz_hits >= 2:
            specs.append(
                {
                    "memory_type": "output_format",
                    "key": "quiz_with_answers",
                    "value": "User often asks for quiz or practice-style output; include answers and explanations when appropriate.",
                    "confidence": min(0.9, 0.6 + quiz_hits * 0.08),
                    "source_conversation_id": conversation.id,
                    "evidence": evidence,
                }
            )

        courses = conversation.file_context.get("courses") or []
        if courses and len(user_messages) >= 2:
            course = str(courses[0])
            specs.append(
                {
                    "memory_type": "course_context",
                    "key": _key_slug(course),
                    "value": f"User is working with course context: {course}.",
                    "confidence": 0.65,
                    "source_conversation_id": conversation.id,
                    "evidence": evidence,
                }
            )

        if any(message.level == "custom" for message in user_messages):
            specs.append(
                {
                    "memory_type": "workflow_preference",
                    "key": "custom_depth",
                    "value": "User has used custom answer-depth settings in chat.",
                    "confidence": 0.58,
                    "source_conversation_id": conversation.id,
                    "evidence": evidence,
                }
            )
    return specs


def _upsert_candidate(
    *,
    memory_type: str,
    key: str,
    value: str,
    confidence: float,
    source_conversation_id: str,
    evidence: str,
) -> UserMemory | None:
    profile_id = _profile_id()
    now = _utc_now()
    with metadata_connection() as connection:
        existing = connection.execute(
            """
            SELECT * FROM user_memories
            WHERE profile_id = ?
              AND memory_type = ?
              AND COALESCE(key, '') = ?
              AND status IN ('candidate', 'active')
            """,
            (profile_id, memory_type, key or ""),
        ).fetchone()
        if existing:
            row = dict(existing)
            next_confidence = max(float(row.get("confidence") or 0.5), confidence)
            connection.execute(
                """
                UPDATE user_memories
                SET confidence = ?, source_conversation_id = ?, evidence = ?,
                    last_seen_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    next_confidence,
                    source_conversation_id,
                    evidence,
                    now,
                    now,
                    row["id"],
                ),
            )
            memory_id = str(row["id"])
        else:
            memory_id = _new_id("mem")
            connection.execute(
                f"""
                INSERT INTO user_memories ({", ".join(MEMORY_COLUMNS)})
                VALUES ({", ".join("?" for _name in MEMORY_COLUMNS)})
                """,
                (
                    memory_id,
                    profile_id,
                    memory_type,
                    key,
                    value,
                    max(0.0, min(1.0, confidence)),
                    source_conversation_id,
                    evidence,
                    "candidate",
                    "normal",
                    now,
                    now,
                    now,
                ),
            )
    _record_event(memory_id, "candidate_created", source_conversation_id, evidence)
    return read_memory(memory_id)


def _relevant_active_memories(question: str, limit: int) -> list[UserMemory]:
    active = list_memories("active")
    question_terms = {
        token.lower()
        for token in re.findall(r"[\w\u4e00-\u9fff]+", question)
        if len(token) >= 2
    }

    def score(memory: UserMemory) -> tuple[float, str]:
        haystack = f"{memory.memory_type} {memory.key or ''} {memory.value}".lower()
        overlap = sum(1 for term in question_terms if term in haystack)
        return (overlap + memory.confidence, memory.updated_at)

    return sorted(active, key=score, reverse=True)[: max(1, min(limit, 10))]


def _recent_conversations(limit: int) -> list[Any]:
    profile_id = _profile_id()
    with metadata_connection() as connection:
        rows = connection.execute(
            """
            SELECT id FROM conversations
            WHERE profile_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (profile_id, limit),
        ).fetchall()
    return [read_conversation(str(row["id"])) for row in rows]


def _set_memory_status(memory_id: str, status: str) -> UserMemory:
    memory = read_memory(memory_id)
    now = _utc_now()
    with metadata_connection() as connection:
        connection.execute(
            """
            UPDATE user_memories
            SET status = ?, updated_at = ?, last_seen_at = COALESCE(last_seen_at, ?)
            WHERE id = ? AND profile_id = ?
            """,
            (status, now, now, memory.id, memory.profile_id),
        )
    return read_memory(memory.id)


def _record_event(
    memory_id: str | None,
    event_type: str,
    source_conversation_id: str | None,
    evidence: str | None,
) -> None:
    profile_id = _profile_id()
    with metadata_connection() as connection:
        connection.execute(
            """
            INSERT INTO memory_events (
                id, memory_id, profile_id, event_type, source_conversation_id,
                evidence, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _new_id("mev"),
                memory_id,
                profile_id,
                event_type,
                source_conversation_id,
                evidence,
                _utc_now(),
            ),
        )


def _memory_from_row(row: dict[str, Any]) -> UserMemory:
    return UserMemory(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        memory_type=str(row["memory_type"]),
        key=row.get("key"),
        value=str(row["value"]),
        confidence=float(row.get("confidence") or 0.5),
        source_conversation_id=row.get("source_conversation_id"),
        evidence=row.get("evidence"),
        status=str(row.get("status") or "candidate"),
        sensitivity=str(row.get("sensitivity") or "normal"),
        last_seen_at=row.get("last_seen_at"),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _evidence_from_messages(messages: list[Any]) -> str:
    snippets = []
    for message in messages[:3]:
        text = " ".join(str(message.content).split())
        snippets.append(text[:120])
    return " | ".join(snippets)


def _normalize_status(value: Any) -> str:
    status = _optional_text(value) or "candidate"
    if status not in {"candidate", "active", "dismissed", "expired", "deleted"}:
        raise HTTPException(status_code=400, detail="Invalid memory status.")
    return status


def _key_slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "_", value.strip().lower())
    return text.strip("_")[:80] or "course"


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail=f"{field_name} is required.")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_id(value: str) -> str:
    text = _optional_text(value)
    if not text:
        raise HTTPException(status_code=400, detail="Memory id is required.")
    return text


def _profile_id() -> str:
    return get_profile().id


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _model_dump(model: Any, **kwargs: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)
    return model.dict(**kwargs)
