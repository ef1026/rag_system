from __future__ import annotations

import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from backend.core.ids import safe_document_id
from backend.schemas import (
    DocumentSummary,
    Folder,
    FolderCreate,
    FolderPatch,
    ManagedFile,
    ManagedFilePatch,
    ManagedFileRegister,
    RegisterManagedFileResponse,
    SyncExistingDocumentsResponse,
    TagSummary,
)
from backend.services.document_service import list_documents as list_document_summaries
from backend.services.profile_service import get_profile
from backend.storage.sqlite import metadata_connection

FOLDER_COLUMNS = (
    "id",
    "profile_id",
    "parent_id",
    "name",
    "sort_order",
    "created_at",
    "updated_at",
)

FILE_COLUMNS = (
    "id",
    "profile_id",
    "document_id",
    "original_filename",
    "display_name",
    "folder_id",
    "tags_json",
    "course",
    "description",
    "notes",
    "pinned",
    "archived",
    "status",
    "source_type",
    "created_at",
    "updated_at",
)


def list_files(
    *,
    folder_id: str | None = None,
    tag: str | None = None,
    course: str | None = None,
    archived: bool | None = None,
    pinned: bool | None = None,
    q: str | None = None,
) -> list[ManagedFile]:
    profile_id = _profile_id()
    document_statuses = _document_status_map()
    with metadata_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM managed_files
            WHERE profile_id = ?
            ORDER BY pinned DESC, updated_at DESC, display_name COLLATE NOCASE
            """,
            (profile_id,),
        ).fetchall()

    files = [_file_from_row(dict(row), document_statuses) for row in rows]
    return [
        file
        for file in files
        if _matches_file_filters(
            file,
            folder_id=_optional_text(folder_id),
            tag=_optional_text(tag),
            course=_optional_text(course),
            archived=archived,
            pinned=pinned,
            q=_optional_text(q),
        )
    ]


def register_file(payload: ManagedFileRegister) -> RegisterManagedFileResponse:
    profile_id = _profile_id()
    document = _require_document(payload.document_id)
    folder_id = _optional_text(payload.folder_id)
    if folder_id:
        _require_folder(folder_id, profile_id)

    existing = _get_file_by_document(profile_id, document.id)
    if existing:
        return RegisterManagedFileResponse(
            file=_update_registered_file(existing.id, payload, document),
            created=False,
        )

    now = _utc_now()
    file_id = _new_id("file")
    values = {
        "id": file_id,
        "profile_id": profile_id,
        "document_id": document.id,
        "original_filename": _optional_text(payload.original_filename) or document.name,
        "display_name": _display_name(payload.display_name, document),
        "folder_id": folder_id,
        "tags_json": json.dumps(_normalize_tags(payload.tags), ensure_ascii=False),
        "course": _optional_text(payload.course),
        "description": _optional_text(payload.description),
        "notes": _optional_text(payload.notes),
        "pinned": 0,
        "archived": 0,
        "status": str(document.status),
        "source_type": "existing_document",
        "created_at": now,
        "updated_at": now,
    }
    _insert_file(values)
    return RegisterManagedFileResponse(
        file=_read_required_file(file_id),
        created=True,
    )


def patch_file(file_id: str, payload: ManagedFilePatch) -> ManagedFile:
    profile_id = _profile_id()
    current = _get_file_by_id(file_id, profile_id)
    if not current:
        raise HTTPException(status_code=404, detail="Managed file not found.")

    updates: dict[str, Any] = {}
    data = _model_dump(payload, exclude_unset=True)
    if "display_name" in data:
        updates["display_name"] = _required_text(data["display_name"], "display_name")
    if "folder_id" in data:
        folder_id = _optional_text(data["folder_id"])
        if folder_id:
            _require_folder(folder_id, profile_id)
        updates["folder_id"] = folder_id
    if "tags" in data:
        updates["tags_json"] = json.dumps(
            _normalize_tags(data["tags"]),
            ensure_ascii=False,
        )
    if "course" in data:
        updates["course"] = _optional_text(data["course"])
    if "description" in data:
        updates["description"] = _optional_text(data["description"])
    if "notes" in data:
        updates["notes"] = _optional_text(data["notes"])
    if "pinned" in data:
        updates["pinned"] = 1 if bool(data["pinned"]) else 0
    if "archived" in data:
        updates["archived"] = 1 if bool(data["archived"]) else 0

    if updates:
        updates["updated_at"] = _utc_now()
        _update_file_columns(current.id, profile_id, updates)
    return _read_required_file(file_id)


def delete_file(file_id: str) -> None:
    profile_id = _profile_id()
    if not _get_file_by_id(file_id, profile_id):
        raise HTTPException(status_code=404, detail="Managed file not found.")
    with metadata_connection() as connection:
        connection.execute(
            "DELETE FROM managed_files WHERE id = ? AND profile_id = ?",
            (file_id, profile_id),
        )


def list_folders() -> list[Folder]:
    profile_id = _profile_id()
    with metadata_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM folders
            WHERE profile_id = ?
            ORDER BY sort_order ASC, name COLLATE NOCASE ASC
            """,
            (profile_id,),
        ).fetchall()
    return [_folder_from_row(dict(row)) for row in rows]


def create_folder(payload: FolderCreate) -> Folder:
    profile_id = _profile_id()
    parent_id = _optional_text(payload.parent_id)
    if parent_id:
        _require_folder(parent_id, profile_id)

    now = _utc_now()
    folder_id = _new_id("folder")
    values = {
        "id": folder_id,
        "profile_id": profile_id,
        "parent_id": parent_id,
        "name": _required_text(payload.name, "name"),
        "sort_order": payload.sort_order,
        "created_at": now,
        "updated_at": now,
    }
    placeholders = ", ".join("?" for _name in FOLDER_COLUMNS)
    with metadata_connection() as connection:
        connection.execute(
            f"""
            INSERT INTO folders ({", ".join(FOLDER_COLUMNS)})
            VALUES ({placeholders})
            """,
            tuple(values[name] for name in FOLDER_COLUMNS),
        )
    return _read_required_folder(folder_id, profile_id)


def patch_folder(folder_id: str, payload: FolderPatch) -> Folder:
    profile_id = _profile_id()
    current = _read_required_folder(folder_id, profile_id)
    updates: dict[str, Any] = {}
    data = _model_dump(payload, exclude_unset=True)
    if "name" in data:
        updates["name"] = _required_text(data["name"], "name")
    if "parent_id" in data:
        parent_id = _optional_text(data["parent_id"])
        if parent_id == current.id:
            raise HTTPException(status_code=400, detail="Folder cannot parent itself.")
        if parent_id:
            _require_folder(parent_id, profile_id)
        updates["parent_id"] = parent_id
    if "sort_order" in data:
        updates["sort_order"] = int(data["sort_order"] or 0)

    if updates:
        updates["updated_at"] = _utc_now()
        assignments = ", ".join(f"{name} = ?" for name in updates)
        with metadata_connection() as connection:
            connection.execute(
                f"UPDATE folders SET {assignments} WHERE id = ? AND profile_id = ?",
                (*updates.values(), folder_id, profile_id),
            )
    return _read_required_folder(folder_id, profile_id)


def delete_folder(folder_id: str) -> None:
    profile_id = _profile_id()
    _read_required_folder(folder_id, profile_id)
    with metadata_connection() as connection:
        child_count = connection.execute(
            "SELECT COUNT(*) FROM folders WHERE profile_id = ? AND parent_id = ?",
            (profile_id, folder_id),
        ).fetchone()[0]
        file_count = connection.execute(
            "SELECT COUNT(*) FROM managed_files WHERE profile_id = ? AND folder_id = ?",
            (profile_id, folder_id),
        ).fetchone()[0]
        if child_count or file_count:
            raise HTTPException(status_code=409, detail="Folder is not empty.")
        connection.execute(
            "DELETE FROM folders WHERE id = ? AND profile_id = ?",
            (folder_id, profile_id),
        )


def list_tags() -> list[TagSummary]:
    counts: Counter[str] = Counter()
    for file in list_files():
        counts.update(file.tags)
    return [
        TagSummary(name=name, count=count)
        for name, count in sorted(counts.items(), key=lambda item: item[0].lower())
    ]


def sync_existing_documents() -> SyncExistingDocumentsResponse:
    profile_id = _profile_id()
    documents = list_document_summaries()
    created_count = 0
    existing_count = 0

    for document in documents:
        if _get_file_by_document(profile_id, document.id):
            existing_count += 1
            continue
        now = _utc_now()
        _insert_file(
            {
                "id": _new_id("file"),
                "profile_id": profile_id,
                "document_id": document.id,
                "original_filename": document.name,
                "display_name": _fallback_display_name(document.name),
                "folder_id": None,
                "tags_json": "[]",
                "course": None,
                "description": None,
                "notes": None,
                "pinned": 0,
                "archived": 0,
                "status": str(document.status),
                "source_type": "existing_document",
                "created_at": now,
                "updated_at": now,
            }
        )
        created_count += 1

    return SyncExistingDocumentsResponse(
        created_count=created_count,
        existing_count=existing_count,
        files=list_files(),
    )


def _update_registered_file(
    file_id: str,
    payload: ManagedFileRegister,
    document: DocumentSummary,
) -> ManagedFile:
    profile_id = _profile_id()
    fields = _fields_set(payload)
    updates: dict[str, Any] = {"status": str(document.status)}

    if "original_filename" in fields:
        updates["original_filename"] = (
            _optional_text(payload.original_filename) or document.name
        )
    if "display_name" in fields:
        updates["display_name"] = _display_name(payload.display_name, document)
    if "folder_id" in fields:
        folder_id = _optional_text(payload.folder_id)
        if folder_id:
            _require_folder(folder_id, profile_id)
        updates["folder_id"] = folder_id
    if "tags" in fields:
        updates["tags_json"] = json.dumps(
            _normalize_tags(payload.tags),
            ensure_ascii=False,
        )
    if "course" in fields:
        updates["course"] = _optional_text(payload.course)
    if "description" in fields:
        updates["description"] = _optional_text(payload.description)
    if "notes" in fields:
        updates["notes"] = _optional_text(payload.notes)

    updates["updated_at"] = _utc_now()
    _update_file_columns(file_id, profile_id, updates)
    return _read_required_file(file_id)


def _insert_file(values: dict[str, Any]) -> None:
    placeholders = ", ".join("?" for _name in FILE_COLUMNS)
    with metadata_connection() as connection:
        connection.execute(
            f"""
            INSERT INTO managed_files ({", ".join(FILE_COLUMNS)})
            VALUES ({placeholders})
            """,
            tuple(values[name] for name in FILE_COLUMNS),
        )


def _update_file_columns(
    file_id: str,
    profile_id: str,
    updates: dict[str, Any],
) -> None:
    assignments = ", ".join(f"{name} = ?" for name in updates)
    with metadata_connection() as connection:
        connection.execute(
            f"UPDATE managed_files SET {assignments} WHERE id = ? AND profile_id = ?",
            (*updates.values(), file_id, profile_id),
        )


def _read_required_file(file_id: str) -> ManagedFile:
    file = _get_file_by_id(file_id, _profile_id())
    if not file:
        raise HTTPException(status_code=500, detail="Managed file could not be loaded.")
    return file


def _get_file_by_id(file_id: str, profile_id: str) -> ManagedFile | None:
    document_statuses = _document_status_map()
    with metadata_connection() as connection:
        row = connection.execute(
            "SELECT * FROM managed_files WHERE id = ? AND profile_id = ?",
            (file_id, profile_id),
        ).fetchone()
    return _file_from_row(dict(row), document_statuses) if row else None


def _get_file_by_document(profile_id: str, document_id: str) -> ManagedFile | None:
    document_statuses = _document_status_map()
    with metadata_connection() as connection:
        row = connection.execute(
            """
            SELECT * FROM managed_files
            WHERE profile_id = ? AND document_id = ?
            """,
            (profile_id, document_id),
        ).fetchone()
    return _file_from_row(dict(row), document_statuses) if row else None


def _read_required_folder(folder_id: str, profile_id: str) -> Folder:
    folder = _get_folder(folder_id, profile_id)
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found.")
    return folder


def _require_folder(folder_id: str, profile_id: str) -> None:
    _read_required_folder(folder_id, profile_id)


def _get_folder(folder_id: str, profile_id: str) -> Folder | None:
    with metadata_connection() as connection:
        row = connection.execute(
            "SELECT * FROM folders WHERE id = ? AND profile_id = ?",
            (folder_id, profile_id),
        ).fetchone()
    return _folder_from_row(dict(row)) if row else None


def _require_document(document_id: str) -> DocumentSummary:
    normalized_id = safe_document_id(document_id)
    for document in list_document_summaries():
        if document.id == normalized_id:
            return document
    raise HTTPException(status_code=404, detail="Document not found.")


def _document_status_map() -> dict[str, str]:
    return {document.id: str(document.status) for document in list_document_summaries()}


def _file_from_row(
    row: dict[str, Any],
    document_statuses: dict[str, str] | None = None,
) -> ManagedFile:
    document_id = str(row["document_id"])
    status = row.get("status")
    if document_statuses is not None:
        status = document_statuses.get(document_id, status or "missing")
    return ManagedFile(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        document_id=document_id,
        original_filename=row.get("original_filename"),
        display_name=str(row["display_name"]),
        folder_id=row.get("folder_id"),
        tags=_load_tags(row.get("tags_json")),
        course=row.get("course"),
        description=row.get("description"),
        notes=row.get("notes"),
        pinned=bool(row.get("pinned")),
        archived=bool(row.get("archived")),
        status=str(status) if status is not None else None,
        source_type=str(row.get("source_type") or "existing_document"),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _folder_from_row(row: dict[str, Any]) -> Folder:
    return Folder(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        parent_id=row.get("parent_id"),
        name=str(row["name"]),
        sort_order=int(row.get("sort_order") or 0),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _matches_file_filters(
    file: ManagedFile,
    *,
    folder_id: str | None,
    tag: str | None,
    course: str | None,
    archived: bool | None,
    pinned: bool | None,
    q: str | None,
) -> bool:
    if folder_id is not None and file.folder_id != folder_id:
        return False
    if tag is not None and tag.lower() not in {item.lower() for item in file.tags}:
        return False
    if course is not None and (file.course or "").lower() != course.lower():
        return False
    if archived is not None and file.archived != archived:
        return False
    if pinned is not None and file.pinned != pinned:
        return False
    if q is not None:
        haystack = " ".join(
            [
                file.document_id,
                file.original_filename or "",
                file.display_name,
                file.course or "",
                file.description or "",
                file.notes or "",
                " ".join(file.tags),
            ]
        ).lower()
        return q.lower() in haystack
    return True


def _display_name(value: str | None, document: DocumentSummary) -> str:
    return _optional_text(value) or _fallback_display_name(document.name)


def _fallback_display_name(filename: str) -> str:
    return Path(filename).stem or filename


def _normalize_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        key = text.lower()
        if text and key not in seen:
            tags.append(text)
            seen.add(key)
    return tags


def _load_tags(value: Any) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return _normalize_tags(parsed)


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


def _profile_id() -> str:
    return get_profile().id


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _fields_set(model: Any) -> set[str]:
    if hasattr(model, "model_fields_set"):
        return set(model.model_fields_set)
    return set(getattr(model, "__fields_set__", set()))


def _model_dump(model: Any, **kwargs: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)
    return model.dict(**kwargs)
