from __future__ import annotations

import re
import uuid
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from backend.core.ids import safe_document_id
from backend.prompts.memory_context import build_memory_prompt_context
from backend.schemas import (
    ConversationMessage,
    MemoryCandidateSource,
    MemoryExtractRequest,
    MemoryExtractResponse,
    UserMemory,
    UserMemoryPatch,
    WrongQuestion,
)
from backend.services.conversation_service import (
    list_conversations,
    list_messages,
    read_conversation,
)
from backend.services.profile_service import get_profile
from backend.services.quiz_service import list_wrong_questions, read_wrong_question
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
    "scope_type",
    "scope_id",
    "evidence_message_ids_json",
    "last_seen_at",
    "last_confirmed_at",
    "expires_at",
    "auto_apply",
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


def list_memory_candidate_sources() -> list[MemoryCandidateSource]:
    sources: list[MemoryCandidateSource] = []
    for conversation in list_conversations():
        messages = [
            message
            for message in list_messages(conversation.id)
            if message.status == "sent" and message.content.strip()
        ]
        if not messages:
            continue
        sources.append(
            MemoryCandidateSource(
                id=conversation.id,
                source_type="conversation",
                title=conversation.title,
                preview=_conversation_preview(messages),
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
                message_count=len(messages),
                document_ids=conversation.document_ids,
            )
        )

    for wrong_question in list_wrong_questions(None):
        sources.append(
            MemoryCandidateSource(
                id=wrong_question.id,
                source_type="wrong_question",
                title=_clip(wrong_question.prompt, 80),
                preview=_wrong_question_value(wrong_question, limit=180),
                created_at=wrong_question.created_at,
                updated_at=wrong_question.reviewed_at or wrong_question.created_at,
                message_count=1,
                reviewed_at=wrong_question.reviewed_at,
            )
        )

    return sorted(
        sources,
        key=lambda source: source.updated_at or source.created_at,
        reverse=True,
    )


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
        "scope_type",
        "scope_id",
        "evidence_message_ids",
        "last_confirmed_at",
        "expires_at",
        "auto_apply",
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
    if "scope_type" in updates:
        updates["scope_type"] = _normalize_scope_type(updates["scope_type"])
    if "scope_id" in updates:
        updates["scope_id"] = _optional_text(updates["scope_id"])
    if "evidence_message_ids" in updates:
        updates["evidence_message_ids_json"] = _json_dumps(
            _normalize_string_list(updates.pop("evidence_message_ids"), max_items=12)
        )
    if "auto_apply" in updates:
        updates["auto_apply"] = 1 if bool(updates["auto_apply"]) else 0
    if "last_confirmed_at" in updates:
        updates["last_confirmed_at"] = _optional_text(updates["last_confirmed_at"])
    if "expires_at" in updates:
        updates["expires_at"] = _optional_text(updates["expires_at"])

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
    conversation_ids = _normalize_id_list(
        [
            *([payload.conversation_id] if payload.conversation_id else []),
            *payload.conversation_ids,
        ],
        max_items=200,
    )
    wrong_question_ids = _normalize_id_list(payload.wrong_question_ids, max_items=200)
    limit = max(1, min(int(payload.limit or 50), 200))
    candidates = _candidate_specs(
        conversation_ids=conversation_ids,
        wrong_question_ids=wrong_question_ids,
    )
    created: list[UserMemory] = []
    for spec in candidates[:limit]:
        memory = _upsert_candidate(**spec)
        if memory:
            if payload.activate:
                memory = accept_memory(memory.id)
            created.append(memory)
    return MemoryExtractResponse(created_count=len(created), memories=created)


def create_memory_candidate(
    *,
    memory_type: str,
    key: str,
    value: str,
    confidence: float,
    source_conversation_id: str | None = None,
    evidence: str | None = None,
    scope_type: str = "global",
    scope_id: str | None = None,
    evidence_message_ids: list[str] | None = None,
    sensitivity: str = "normal",
    auto_apply: bool = True,
) -> UserMemory | None:
    return _upsert_candidate(
        memory_type=memory_type,
        key=key,
        value=value,
        confidence=confidence,
        source_conversation_id=source_conversation_id,
        evidence=evidence or "",
        scope_type=scope_type,
        scope_id=scope_id,
        evidence_message_ids=evidence_message_ids or [],
        sensitivity=sensitivity,
        auto_apply=auto_apply,
    )


def build_relevant_memory_context(
    *,
    question: str,
    document_ids: list[str],
    conversation_id: str | None = None,
    limit: int = 5,
) -> str:
    memories = select_relevant_memories(
        question=question,
        document_ids=document_ids,
        conversation_id=conversation_id,
        limit=limit,
    )
    return build_memory_prompt_context(memories)


def select_relevant_memories(
    *,
    question: str,
    document_ids: list[str],
    conversation_id: str | None = None,
    limit: int = 5,
) -> list[UserMemory]:
    return _relevant_active_memories(
        question=question,
        document_ids=document_ids,
        conversation_id=conversation_id,
        limit=limit,
    )


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
    wrong_question_ids: list[str],
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
        conversation_spec = _conversation_candidate_spec(conversation, messages)
        if conversation_spec:
            specs.append(conversation_spec)
        if not user_messages:
            continue
        combined = "\n".join(message.content for message in user_messages)
        evidence = _evidence_from_messages(user_messages)
        evidence_message_ids = [message.id for message in user_messages[:3]]
        scoped_type, scoped_id = _scope_from_conversation(conversation)

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
                    "scope_type": "global",
                    "scope_id": None,
                    "evidence_message_ids": evidence_message_ids,
                }
            )

        quiz_hits = len(
            re.findall(
                r"\bquiz\b|practice|exercise|exam|test|出题|练习|习题|测验|考试",
                combined,
                flags=re.IGNORECASE,
            )
        )
        quiz_hits += len(re.findall(r"出题|练习|习题|测验|考试", combined))
        if quiz_hits >= 2:
            specs.append(
                {
                    "memory_type": "output_format",
                    "key": "quiz_with_answers",
                    "value": "User often asks for quiz or practice-style output; include answers and explanations when appropriate.",
                    "confidence": min(0.9, 0.6 + quiz_hits * 0.08),
                    "source_conversation_id": conversation.id,
                    "evidence": evidence,
                    "scope_type": scoped_type,
                    "scope_id": scoped_id,
                    "evidence_message_ids": evidence_message_ids,
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
                    "scope_type": "course",
                    "scope_id": course,
                    "evidence_message_ids": evidence_message_ids,
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
                    "scope_type": "global",
                    "scope_id": None,
                    "evidence_message_ids": evidence_message_ids,
                }
            )
    for wrong_question_id in wrong_question_ids:
        wrong_question_spec = _wrong_question_candidate_spec(
            read_wrong_question(wrong_question_id)
        )
        if wrong_question_spec:
            specs.append(wrong_question_spec)
    return specs


def _conversation_candidate_spec(
    conversation: Any,
    messages: list[ConversationMessage],
) -> dict[str, Any] | None:
    evidence_messages = [
        message
        for message in messages
        if message.status == "sent" and message.content.strip()
    ]
    if not evidence_messages:
        return None
    scoped_type, scoped_id = _scope_from_conversation(conversation)
    evidence = _evidence_from_messages(evidence_messages, max_messages=6, limit=180)
    return {
        "memory_type": "chat_record",
        "key": f"chat_{conversation.id}",
        "value": (
            f"聊天记录「{conversation.title}」包含可复用的学习上下文："
            f"{_conversation_preview(evidence_messages, limit=360)}"
        ),
        "confidence": 0.56,
        "source_conversation_id": conversation.id,
        "evidence": evidence,
        "scope_type": scoped_type,
        "scope_id": scoped_id,
        "evidence_message_ids": [message.id for message in evidence_messages[:6]],
    }


def _wrong_question_candidate_spec(
    wrong_question: WrongQuestion,
) -> dict[str, Any] | None:
    value = _wrong_question_value(wrong_question, limit=900)
    if not value:
        return None
    scope_type, scope_id = _scope_from_conversation_id(wrong_question.conversation_id)
    return {
        "memory_type": "wrong_question",
        "key": f"wrong_{wrong_question.id}",
        "value": value,
        "confidence": 0.84 if not wrong_question.reviewed_at else 0.72,
        "source_conversation_id": wrong_question.conversation_id,
        "evidence": _clip(value, 360),
        "scope_type": scope_type,
        "scope_id": scope_id,
        "evidence_message_ids": _normalize_source_message_ids(
            wrong_question.source_message_ids
        ),
    }


def _wrong_question_value(wrong_question: WrongQuestion, *, limit: int) -> str:
    correct_choice = _choice_text(
        wrong_question.choices,
        wrong_question.correct_choice_id,
    )
    selected_choice = _choice_text(
        wrong_question.choices,
        wrong_question.selected_choice_id,
    )
    value = (
        f"错题：{wrong_question.prompt}\n"
        f"用户错选：{selected_choice}\n"
        f"正确答案：{correct_choice}\n"
        f"解析：{wrong_question.explanation}"
    )
    return _clip(value, limit)


def _choice_text(choices: list[Any], choice_id: str) -> str:
    normalized_id = str(choice_id or "").strip()
    for choice in choices:
        if getattr(choice, "id", "") == normalized_id:
            return str(getattr(choice, "text", "") or normalized_id)
    return normalized_id or "未记录"


def _upsert_candidate(
    *,
    memory_type: str,
    key: str,
    value: str,
    confidence: float,
    source_conversation_id: str | None,
    evidence: str,
    scope_type: str = "global",
    scope_id: str | None = None,
    evidence_message_ids: list[str] | None = None,
    sensitivity: str = "normal",
    auto_apply: bool = True,
) -> UserMemory | None:
    profile_id = _profile_id()
    now = _utc_now()
    normalized_scope_type = _normalize_scope_type(scope_type)
    normalized_scope_id = _optional_text(scope_id)
    normalized_evidence_message_ids = _normalize_string_list(
        evidence_message_ids or [],
        max_items=12,
    )
    with metadata_connection() as connection:
        existing = connection.execute(
            """
            SELECT * FROM user_memories
            WHERE profile_id = ?
              AND memory_type = ?
              AND COALESCE(key, '') = ?
              AND scope_type = ?
              AND COALESCE(scope_id, '') = ?
              AND status IN ('candidate', 'active')
            """,
            (
                profile_id,
                memory_type,
                key or "",
                normalized_scope_type,
                normalized_scope_id or "",
            ),
        ).fetchone()
        if existing:
            row = dict(existing)
            next_confidence = max(float(row.get("confidence") or 0.5), confidence)
            connection.execute(
                """
                UPDATE user_memories
                SET confidence = ?, source_conversation_id = ?, evidence = ?,
                    evidence_message_ids_json = ?, last_seen_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    next_confidence,
                    source_conversation_id,
                    evidence,
                    _json_dumps(normalized_evidence_message_ids),
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
                    _optional_text(sensitivity) or "normal",
                    normalized_scope_type,
                    normalized_scope_id,
                    _json_dumps(normalized_evidence_message_ids),
                    now,
                    None,
                    None,
                    1 if auto_apply else 0,
                    now,
                    now,
                ),
            )
    _record_event(memory_id, "candidate_created", source_conversation_id, evidence)
    return read_memory(memory_id)


def _relevant_active_memories(
    *,
    question: str,
    document_ids: list[str],
    conversation_id: str | None,
    limit: int,
) -> list[UserMemory]:
    active = [
        memory
        for memory in list_memories("active")
        if memory.auto_apply and not _is_expired(memory)
    ]
    allowed_scopes = _allowed_scopes(document_ids, conversation_id)
    question_terms = {
        token.lower()
        for token in re.findall(r"[\w\u4e00-\u9fff]+", question)
        if len(token) >= 2
    }

    def score(memory: UserMemory) -> tuple[float, float, str]:
        haystack = f"{memory.memory_type} {memory.key or ''} {memory.value}".lower()
        overlap = sum(1 for term in question_terms if term in haystack)
        scope_weight = _scope_weight(memory, allowed_scopes)
        return (scope_weight, overlap + memory.confidence, memory.updated_at)

    scoped = [
        memory
        for memory in active
        if (memory.scope_type, memory.scope_id or "") in allowed_scopes
    ]
    return sorted(scoped, key=score, reverse=True)[: max(1, min(limit, 10))]


def _allowed_scopes(
    document_ids: list[str],
    conversation_id: str | None,
) -> set[tuple[str, str]]:
    scopes: set[tuple[str, str]] = {("global", "")}
    normalized_document_ids = [
        safe_document_id(str(document_id or ""))
        for document_id in document_ids
        if str(document_id or "").strip()
    ]
    for document_id in normalized_document_ids:
        if document_id:
            scopes.add(("document", document_id))
    if conversation_id:
        scopes.add(("conversation", conversation_id))
    for course in _course_scope_ids(normalized_document_ids):
        scopes.add(("course", course))
    return scopes


def _course_scope_ids(document_ids: list[str]) -> list[str]:
    normalized_document_ids = [item for item in document_ids if item]
    if not normalized_document_ids:
        return []
    placeholders = ", ".join("?" for _item in normalized_document_ids)
    profile_id = _profile_id()
    with metadata_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT DISTINCT course FROM managed_files
            WHERE profile_id = ? AND document_id IN ({placeholders})
              AND course IS NOT NULL AND TRIM(course) != ''
            """,
            (profile_id, *normalized_document_ids),
        ).fetchall()
    return [str(row["course"]) for row in rows]


def _scope_weight(
    memory: UserMemory,
    allowed_scopes: set[tuple[str, str]],
) -> float:
    scope = (memory.scope_type, memory.scope_id or "")
    if scope not in allowed_scopes:
        return -1.0
    if memory.scope_type in {"document", "course"}:
        return 3.0
    if memory.scope_type == "conversation":
        return 2.0
    return 1.0


def _scope_from_conversation(conversation: Any) -> tuple[str, str | None]:
    courses = conversation.file_context.get("courses") or []
    if courses:
        return "course", str(courses[0])
    if len(conversation.document_ids) == 1:
        return "document", safe_document_id(conversation.document_ids[0])
    return "conversation", conversation.id


def _scope_from_conversation_id(conversation_id: str | None) -> tuple[str, str | None]:
    normalized_id = _optional_text(conversation_id)
    if not normalized_id:
        return "global", None
    try:
        return _scope_from_conversation(read_conversation(normalized_id))
    except HTTPException:
        return "conversation", normalized_id


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
    last_confirmed_at = now if status == "active" else memory.last_confirmed_at
    with metadata_connection() as connection:
        connection.execute(
            """
            UPDATE user_memories
            SET status = ?, updated_at = ?, last_seen_at = COALESCE(last_seen_at, ?),
                last_confirmed_at = ?
            WHERE id = ? AND profile_id = ?
            """,
            (status, now, now, last_confirmed_at, memory.id, memory.profile_id),
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
        scope_type=_normalize_scope_type(row.get("scope_type")),
        scope_id=row.get("scope_id"),
        evidence_message_ids=_load_json_list(row.get("evidence_message_ids_json")),
        last_seen_at=row.get("last_seen_at"),
        last_confirmed_at=row.get("last_confirmed_at"),
        expires_at=row.get("expires_at"),
        auto_apply=bool(row.get("auto_apply", 1)),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _conversation_preview(
    messages: list[ConversationMessage],
    *,
    limit: int = 220,
) -> str:
    snippets = []
    for message in messages[-6:]:
        role = "用户" if message.role == "user" else "助手"
        snippets.append(f"{role}：{_clip(message.content, 120)}")
    return _clip(" / ".join(snippets), limit)


def _evidence_from_messages(
    messages: list[Any],
    *,
    max_messages: int = 3,
    limit: int = 120,
) -> str:
    snippets = []
    for message in messages[:max_messages]:
        text = " ".join(str(message.content).split())
        snippets.append(_clip(text, limit))
    return " | ".join(snippets)


def _is_expired(memory: UserMemory) -> bool:
    return bool(memory.expires_at and memory.expires_at <= _utc_now())


def _normalize_scope_type(value: Any) -> str:
    scope_type = _optional_text(value) or "global"
    if scope_type not in {"global", "course", "document", "conversation"}:
        raise HTTPException(status_code=400, detail="Invalid memory scope_type.")
    return scope_type


def _normalize_status(value: Any) -> str:
    status = _optional_text(value) or "candidate"
    if status not in {"candidate", "active", "dismissed", "expired", "deleted"}:
        raise HTTPException(status_code=400, detail="Invalid memory status.")
    return status


def _normalize_string_list(value: Any, max_items: int = 20) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        items.append(text[:160])
        seen.add(text)
        if len(items) >= max_items:
            break
    return items


def _normalize_id_list(value: Any, max_items: int = 100) -> list[str]:
    return _normalize_string_list(value, max_items=max_items)


def _normalize_source_message_ids(value: Any) -> list[str]:
    normalized = []
    for item in _normalize_string_list(value, max_items=12):
        message_id = item.removeprefix("learn_")
        if message_id.startswith("msg_") or message_id:
            normalized.append(message_id)
    return normalized


def _load_json_list(value: Any) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return _normalize_string_list(parsed, max_items=20)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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


def _clip(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 1)]}..."


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
