from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from backend.core.ids import safe_document_id
from backend.schemas import (
    Conversation,
    ConversationCreate,
    ConversationImportItem,
    ConversationImportRequest,
    ConversationImportResponse,
    ConversationMessage,
    ConversationMessageImport,
    ConversationPatch,
    ImageAssetPublic,
    SourceItem,
)
from backend.services.profile_service import get_profile
from backend.storage.sqlite import metadata_connection

CONVERSATION_COLUMNS = (
    "id",
    "profile_id",
    "title",
    "document_ids_json",
    "level",
    "mode",
    "chat_mode",
    "file_context_json",
    "created_at",
    "updated_at",
)

MESSAGE_COLUMNS = (
    "id",
    "conversation_id",
    "role",
    "content",
    "status",
    "document_ids_json",
    "level",
    "mode",
    "chat_mode",
    "sources_json",
    "related_images_json",
    "inline_image_refs_json",
    "error",
    "created_at",
)


def list_conversations() -> list[Conversation]:
    profile_id = _profile_id()
    with metadata_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM conversations
            WHERE profile_id = ?
            ORDER BY updated_at DESC
            """,
            (profile_id,),
        ).fetchall()
    return [_conversation_from_row(dict(row)) for row in rows]


def create_conversation(payload: ConversationCreate) -> Conversation:
    profile_id = _profile_id()
    now = _utc_now()
    conversation_id = _normalize_client_id(payload.id) or _new_id("conv")
    document_ids = _normalize_document_ids(payload.document_ids)
    values = {
        "id": conversation_id,
        "profile_id": profile_id,
        "title": _optional_text(payload.title) or "New conversation",
        "document_ids_json": _json_dumps(document_ids),
        "level": _optional_text(payload.level) or "undergraduate",
        "mode": _optional_text(payload.mode) or "hybrid",
        "chat_mode": _optional_text(payload.chat_mode) or "multimodal",
        "file_context_json": _json_dumps(
            payload.file_context or build_file_context(document_ids)
        ),
        "created_at": now,
        "updated_at": now,
    }
    _upsert_conversation_values(values)
    return read_conversation(conversation_id)


def read_conversation(conversation_id: str) -> Conversation:
    conversation = _get_conversation(_require_id(conversation_id))
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


def patch_conversation(
    conversation_id: str,
    payload: ConversationPatch,
) -> Conversation:
    profile_id = _profile_id()
    current = read_conversation(conversation_id)
    data = _model_dump(payload, exclude_unset=True)
    updates: dict[str, Any] = {}

    if "title" in data:
        updates["title"] = _optional_text(data["title"]) or current.title
    if "document_ids" in data:
        document_ids = _normalize_document_ids(data["document_ids"])
        updates["document_ids_json"] = _json_dumps(document_ids)
        updates["file_context_json"] = _json_dumps(build_file_context(document_ids))
    if "level" in data:
        updates["level"] = _optional_text(data["level"]) or current.level
    if "mode" in data:
        updates["mode"] = _optional_text(data["mode"]) or current.mode
    if "chat_mode" in data:
        updates["chat_mode"] = _optional_text(data["chat_mode"]) or current.chat_mode
    if "file_context" in data and data["file_context"] is not None:
        updates["file_context_json"] = _json_dumps(data["file_context"])

    if updates:
        updates["updated_at"] = _utc_now()
        assignments = ", ".join(f"{name} = ?" for name in updates)
        with metadata_connection() as connection:
            connection.execute(
                f"UPDATE conversations SET {assignments} WHERE id = ? AND profile_id = ?",
                (*updates.values(), current.id, profile_id),
            )
    return read_conversation(current.id)


def delete_conversation(conversation_id: str) -> None:
    profile_id = _profile_id()
    conversation = read_conversation(conversation_id)
    with metadata_connection() as connection:
        connection.execute(
            "DELETE FROM conversation_messages WHERE conversation_id = ?",
            (conversation.id,),
        )
        connection.execute(
            "DELETE FROM conversations WHERE id = ? AND profile_id = ?",
            (conversation.id, profile_id),
        )


def list_messages(conversation_id: str) -> list[ConversationMessage]:
    conversation = read_conversation(conversation_id)
    with metadata_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM conversation_messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            """,
            (conversation.id,),
        ).fetchall()
    return [_message_from_row(dict(row)) for row in rows]


def clear_messages(conversation_id: str) -> None:
    conversation = read_conversation(conversation_id)
    with metadata_connection() as connection:
        connection.execute(
            "DELETE FROM conversation_messages WHERE conversation_id = ?",
            (conversation.id,),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_utc_now(), conversation.id),
        )


def import_conversations(
    payload: ConversationImportRequest,
) -> ConversationImportResponse:
    imported_count = 0
    message_count = 0
    conversations: list[Conversation] = []
    for item in payload.conversations:
        conversation = upsert_imported_conversation(item)
        imported_count += 1
        conversations.append(conversation)
        for message in item.messages:
            upsert_imported_message(conversation.id, message)
            message_count += 1
    return ConversationImportResponse(
        imported_count=imported_count,
        message_count=message_count,
        conversations=conversations,
    )


def upsert_imported_conversation(item: ConversationImportItem) -> Conversation:
    profile_id = _profile_id()
    now = _utc_now()
    conversation_id = _normalize_client_id(item.id)
    if not conversation_id:
        raise HTTPException(status_code=400, detail="Conversation id is required.")
    document_ids = _normalize_document_ids(item.document_ids)
    values = {
        "id": conversation_id,
        "profile_id": profile_id,
        "title": _optional_text(item.title) or "New conversation",
        "document_ids_json": _json_dumps(document_ids),
        "level": _optional_text(item.level) or "undergraduate",
        "mode": _optional_text(item.mode) or "hybrid",
        "chat_mode": _optional_text(item.chat_mode) or "multimodal",
        "file_context_json": _json_dumps(item.file_context or build_file_context(document_ids)),
        "created_at": _optional_text(item.created_at) or now,
        "updated_at": _optional_text(item.updated_at) or now,
    }
    _upsert_conversation_values(values)
    return read_conversation(conversation_id)


def upsert_imported_message(
    conversation_id: str,
    message: ConversationMessageImport,
) -> ConversationMessage:
    message_id = _normalize_client_id(message.id) or _new_id("msg")
    values = _message_values(
        message_id=message_id,
        conversation_id=conversation_id,
        role=message.role,
        content=message.content,
        status=message.status,
        document_ids=message.document_ids,
        level=message.level,
        mode=message.mode,
        chat_mode=message.chat_mode,
        sources=message.sources,
        related_images=message.related_images,
        inline_image_refs=message.inline_image_refs,
        error=message.error,
        created_at=_optional_text(message.created_at) or _utc_now(),
    )
    _upsert_message_values(values)
    return _read_required_message(message_id)


def ensure_conversation_for_chat(
    *,
    conversation_id: str,
    question: str,
    document_ids: list[str],
    level: str,
    mode: str,
    chat_mode: str | None = None,
) -> Conversation:
    normalized_id = _normalize_client_id(conversation_id)
    if not normalized_id:
        raise HTTPException(status_code=400, detail="conversation_id is invalid.")
    existing = _get_conversation(normalized_id)
    if existing:
        return patch_conversation(
            normalized_id,
            ConversationPatch(
                document_ids=document_ids,
                level=level,
                mode=mode,
                chat_mode=chat_mode or existing.chat_mode,
            ),
        )
    return create_conversation(
        ConversationCreate(
            id=normalized_id,
            title=_title_from_question(question),
            document_ids=document_ids,
            level=level,
            mode=mode,
            chat_mode=chat_mode or "multimodal",
        )
    )


def append_message(
    *,
    conversation_id: str,
    role: str,
    content: str,
    document_ids: list[str],
    message_id: str | None = None,
    level: str | None = None,
    mode: str | None = None,
    chat_mode: str | None = None,
    sources: list[SourceItem] | None = None,
    related_images: list[ImageAssetPublic] | None = None,
    inline_image_refs: list[str] | None = None,
    status: str = "sent",
    error: str | None = None,
) -> ConversationMessage:
    conversation = read_conversation(conversation_id)
    normalized_message_id = _normalize_client_id(message_id) or _new_id("msg")
    now = _utc_now()
    values = _message_values(
        message_id=normalized_message_id,
        conversation_id=conversation.id,
        role=role,
        content=content,
        status=status,
        document_ids=document_ids,
        level=level,
        mode=mode,
        chat_mode=chat_mode,
        sources=sources or [],
        related_images=related_images or [],
        inline_image_refs=inline_image_refs or [],
        error=error,
        created_at=now,
    )
    _upsert_message_values(values)
    _touch_conversation(conversation.id, document_ids, level, mode, chat_mode, now)
    return _read_required_message(normalized_message_id)


def update_message_status(
    message_id: str,
    *,
    status: str,
    error: str | None = None,
) -> ConversationMessage:
    now = _utc_now()
    with metadata_connection() as connection:
        row = connection.execute(
            "SELECT conversation_id FROM conversation_messages WHERE id = ?",
            (message_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Conversation message not found.")
        connection.execute(
            "UPDATE conversation_messages SET status = ?, error = ? WHERE id = ?",
            (_optional_text(status) or "sent", _optional_text(error), message_id),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, row["conversation_id"]),
        )
    return _read_required_message(message_id)


def build_file_context(document_ids: list[str]) -> dict[str, Any]:
    profile_id = _profile_id()
    normalized_ids = _normalize_document_ids(document_ids)
    if not normalized_ids:
        return {}

    placeholders = ", ".join("?" for _item in normalized_ids)
    with metadata_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT document_id, folder_id, tags_json, course, pinned, archived
            FROM managed_files
            WHERE profile_id = ? AND document_id IN ({placeholders})
            """,
            (profile_id, *normalized_ids),
        ).fetchall()

    courses: list[str] = []
    folder_ids: list[str] = []
    tags: list[str] = []
    pinned_ids: list[str] = []
    archived_ids: list[str] = []
    for row in rows:
        data = dict(row)
        document_id = str(data["document_id"])
        course = _optional_text(data.get("course"))
        folder_id = _optional_text(data.get("folder_id"))
        if course and course not in courses:
            courses.append(course)
        if folder_id and folder_id not in folder_ids:
            folder_ids.append(folder_id)
        for tag in _load_json_list(data.get("tags_json")):
            if tag not in tags:
                tags.append(tag)
        if data.get("pinned"):
            pinned_ids.append(document_id)
        if data.get("archived"):
            archived_ids.append(document_id)

    return {
        "document_ids": normalized_ids,
        "courses": courses,
        "folder_ids": folder_ids,
        "tags": tags,
        "pinned_document_ids": pinned_ids,
        "archived_document_ids": archived_ids,
    }


def _upsert_conversation_values(values: dict[str, Any]) -> None:
    placeholders = ", ".join("?" for _name in CONVERSATION_COLUMNS)
    assignments = ", ".join(
        f"{name} = excluded.{name}" for name in CONVERSATION_COLUMNS[1:]
    )
    with metadata_connection() as connection:
        connection.execute(
            f"""
            INSERT INTO conversations ({", ".join(CONVERSATION_COLUMNS)})
            VALUES ({placeholders})
            ON CONFLICT(id) DO UPDATE SET {assignments}
            """,
            tuple(values[name] for name in CONVERSATION_COLUMNS),
        )


def _upsert_message_values(values: dict[str, Any]) -> None:
    placeholders = ", ".join("?" for _name in MESSAGE_COLUMNS)
    assignments = ", ".join(f"{name} = excluded.{name}" for name in MESSAGE_COLUMNS[1:])
    with metadata_connection() as connection:
        connection.execute(
            f"""
            INSERT INTO conversation_messages ({", ".join(MESSAGE_COLUMNS)})
            VALUES ({placeholders})
            ON CONFLICT(id) DO UPDATE SET {assignments}
            """,
            tuple(values[name] for name in MESSAGE_COLUMNS),
        )


def _message_values(
    *,
    message_id: str,
    conversation_id: str,
    role: str,
    content: str,
    status: str,
    document_ids: list[str],
    level: str | None,
    mode: str | None,
    chat_mode: str | None,
    sources: list[SourceItem],
    related_images: list[ImageAssetPublic],
    inline_image_refs: list[str],
    error: str | None,
    created_at: str,
) -> dict[str, Any]:
    if role not in {"user", "assistant"}:
        raise HTTPException(status_code=400, detail="Invalid conversation role.")
    text = str(content or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message content is required.")
    return {
        "id": message_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": text,
        "status": _optional_text(status) or "sent",
        "document_ids_json": _json_dumps(_normalize_document_ids(document_ids)),
        "level": _optional_text(level),
        "mode": _optional_text(mode),
        "chat_mode": _optional_text(chat_mode),
        "sources_json": _json_dumps([_model_dump(item) for item in sources]),
        "related_images_json": _json_dumps(
            [_model_dump(item, exclude_none=True) for item in related_images]
        ),
        "inline_image_refs_json": _json_dumps(
            [str(item) for item in inline_image_refs if str(item).strip()]
        ),
        "error": _optional_text(error),
        "created_at": created_at,
    }


def _touch_conversation(
    conversation_id: str,
    document_ids: list[str],
    level: str | None,
    mode: str | None,
    chat_mode: str | None,
    updated_at: str,
) -> None:
    normalized_ids = _normalize_document_ids(document_ids)
    updates = {
        "document_ids_json": _json_dumps(normalized_ids),
        "file_context_json": _json_dumps(build_file_context(normalized_ids)),
        "updated_at": updated_at,
    }
    if level:
        updates["level"] = level
    if mode:
        updates["mode"] = mode
    if chat_mode:
        updates["chat_mode"] = chat_mode
    assignments = ", ".join(f"{name} = ?" for name in updates)
    with metadata_connection() as connection:
        connection.execute(
            f"UPDATE conversations SET {assignments} WHERE id = ?",
            (*updates.values(), conversation_id),
        )


def _get_conversation(conversation_id: str) -> Conversation | None:
    profile_id = _profile_id()
    with metadata_connection() as connection:
        row = connection.execute(
            "SELECT * FROM conversations WHERE id = ? AND profile_id = ?",
            (conversation_id, profile_id),
        ).fetchone()
    return _conversation_from_row(dict(row)) if row else None


def _read_required_message(message_id: str) -> ConversationMessage:
    with metadata_connection() as connection:
        row = connection.execute(
            "SELECT * FROM conversation_messages WHERE id = ?",
            (message_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=500, detail="Conversation message not found.")
    return _message_from_row(dict(row))


def _conversation_from_row(row: dict[str, Any]) -> Conversation:
    return Conversation(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        title=str(row["title"]),
        document_ids=_load_json_list(row.get("document_ids_json")),
        level=str(row.get("level") or "undergraduate"),
        mode=str(row.get("mode") or "hybrid"),
        chat_mode=str(row.get("chat_mode") or "multimodal"),
        file_context=_load_json_dict(row.get("file_context_json")),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _message_from_row(row: dict[str, Any]) -> ConversationMessage:
    return ConversationMessage(
        id=str(row["id"]),
        conversation_id=str(row["conversation_id"]),
        role=row["role"],
        content=str(row["content"]),
        status=str(row.get("status") or "sent"),
        document_ids=_load_json_list(row.get("document_ids_json")),
        level=row.get("level"),
        mode=row.get("mode"),
        chat_mode=row.get("chat_mode"),
        sources=[
            SourceItem(**item)
            for item in _load_json_records(row.get("sources_json"))
            if isinstance(item, dict)
        ],
        related_images=[
            ImageAssetPublic(**item)
            for item in _load_json_records(row.get("related_images_json"))
            if isinstance(item, dict)
        ],
        inline_image_refs=_load_json_list(row.get("inline_image_refs_json")),
        error=row.get("error"),
        created_at=str(row["created_at"]),
    )


def _normalize_document_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    document_ids: list[str] = []
    seen: set[str] = set()
    for raw_item in value:
        document_id = safe_document_id(str(raw_item or ""))
        if document_id and document_id not in seen:
            document_ids.append(document_id)
            seen.add(document_id)
    return document_ids


def _normalize_client_id(value: str | None) -> str | None:
    text = _optional_text(value)
    if not text:
        return None
    allowed = []
    for char in text[:120]:
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
    normalized = "".join(allowed)
    return normalized or None


def _require_id(value: str) -> str:
    normalized = _normalize_client_id(value)
    if not normalized:
        raise HTTPException(status_code=400, detail="Invalid conversation id.")
    return normalized


def _title_from_question(question: str) -> str:
    normalized = " ".join(str(question or "").strip().split())
    return normalized[:40] or "New conversation"


def _load_json_list(value: Any) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    result: list[str] = []
    for item in parsed:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _load_json_records(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def _load_json_dict(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
