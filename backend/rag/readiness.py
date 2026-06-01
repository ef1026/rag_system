from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.core.ids import safe_document_id, safe_document_key, same_document_file
from backend.core.models import StorageReadiness
from backend.core.paths import document_storage_dir
from backend.rag.storage import read_json_dict, vdb_records_by_file

logger = logging.getLogger("api_server")

def document_storage_readiness(document_id: str) -> StorageReadiness:
    filename = safe_document_id(document_id)
    storage_dir = document_storage_dir(filename)
    warnings: list[str] = []

    if not storage_dir.exists():
        return StorageReadiness(False, "parsed", ["document-scoped storage is missing"], storage_dir)

    required_files = (
        "kv_store_doc_status.json",
        "kv_store_text_chunks.json",
        "vdb_chunks.json",
        "graph_chunk_entity_relation.graphml",
    )
    for required_file in required_files:
        if not (storage_dir / required_file).exists():
            warnings.append(f"{required_file} is missing")

    doc_statuses = read_json_dict(storage_dir / "kv_store_doc_status.json")
    text_chunks = read_json_dict(storage_dir / "kv_store_text_chunks.json")

    current_doc_status: dict[str, Any] | None = None
    doc_status_refs: list[str] = []
    other_doc_status_files: set[str] = set()
    for doc_status_id, status in doc_statuses.items():
        if not isinstance(status, dict):
            continue
        file_path = status.get("file_path")
        file_name = Path(file_path).name if isinstance(file_path, str) and file_path.strip() else "<missing>"
        doc_status_refs.append(f"{doc_status_id}:{file_name}")
        if same_document_file(file_path, filename):
            current_doc_status = status
        elif isinstance(file_path, str) and file_path.strip():
            other_doc_status_files.add(Path(file_path).name)

    if other_doc_status_files:
        warnings.append(
            "doc_status contains other documents: "
            + ", ".join(sorted(other_doc_status_files))
        )

    if current_doc_status is None:
        warnings.append("current document doc_status is missing")
        logger.warning(
            "Document storage readiness failed expected_document_id=%s safe_document_key=%s storage_dir=%s doc_status_refs=%s polluted_doc_status_files=%s warnings=%s",
            filename,
            safe_document_key(filename),
            storage_dir,
            doc_status_refs,
            sorted(other_doc_status_files),
            warnings,
        )
        return StorageReadiness(False, "partial_success", warnings, storage_dir)

    raw_status = str(current_doc_status.get("status", "")).lower()
    if raw_status not in {"processed", "success", "completed", "complete"}:
        warnings.append(f"current document doc_status is not processed: {raw_status or 'unknown'}")

    chunks_list = [
        str(chunk_id)
        for chunk_id in current_doc_status.get("chunks_list", [])
        if str(chunk_id).strip()
    ]
    chunks_count = int(current_doc_status.get("chunks_count") or len(chunks_list) or 0)
    if chunks_count <= 0:
        warnings.append("current document chunks_count is 0")
    if not chunks_list:
        warnings.append("current document chunks_list is empty")

    text_chunk_files: set[str] = set()
    all_text_chunk_files: set[str] = set()
    current_text_chunk_ids: set[str] = set()
    for chunk_id, chunk in text_chunks.items():
        if not isinstance(chunk, dict):
            continue
        file_path = chunk.get("file_path")
        if isinstance(file_path, str) and file_path.strip():
            all_text_chunk_files.add(Path(file_path).name)
        if same_document_file(file_path, filename):
            current_text_chunk_ids.add(str(chunk_id))
        elif isinstance(file_path, str) and file_path.strip():
            text_chunk_files.add(Path(file_path).name)

    if text_chunk_files:
        warnings.append(
            "text_chunks contains other documents: " + ", ".join(sorted(text_chunk_files))
        )

    missing_text_chunks: list[str] = []
    wrong_file_chunks: list[str] = []
    empty_text_chunks: list[str] = []
    for chunk_id in chunks_list:
        chunk = text_chunks.get(chunk_id)
        if not isinstance(chunk, dict):
            missing_text_chunks.append(chunk_id)
            continue
        if not str(chunk.get("content", "")).strip():
            empty_text_chunks.append(chunk_id)
        if chunk.get("file_path") and not same_document_file(chunk.get("file_path"), filename):
            wrong_file_chunks.append(chunk_id)

    if missing_text_chunks:
        warnings.append(
            "chunks_list references missing text_chunks: "
            + ", ".join(missing_text_chunks[:8])
        )
    if empty_text_chunks:
        warnings.append(
            "chunks_list references empty text_chunks: "
            + ", ".join(empty_text_chunks[:8])
        )
    if wrong_file_chunks:
        warnings.append(
            "chunks_list references chunks from another file: "
            + ", ".join(wrong_file_chunks[:8])
        )

    vdb_records, vdb_warnings = vdb_records_by_file(storage_dir, filename)
    warnings.extend(vdb_warnings)
    vdb_current_ids = {
        str(record.get("__id__") or record.get("id"))
        for record in vdb_records
        if str(record.get("__id__") or record.get("id") or "").strip()
    }
    expected_ids = set(chunks_list)
    if not vdb_current_ids:
        warnings.append("vdb_chunks has no chunks for current document")
    missing_vdb = sorted(expected_ids - vdb_current_ids) if expected_ids else []
    if missing_vdb:
        warnings.append(
            "chunks_list references chunks missing from vdb_chunks: "
            + ", ".join(missing_vdb[:8])
        )
    vdb_without_text = sorted(vdb_current_ids - set(text_chunks.keys()))
    if vdb_without_text:
        warnings.append(
            "vdb_chunks contains ids missing from text_chunks: "
            + ", ".join(vdb_without_text[:8])
        )
    text_without_vdb = sorted(current_text_chunk_ids - vdb_current_ids)
    if text_without_vdb:
        warnings.append(
            "text_chunks contains current-document ids missing from vdb_chunks: "
            + ", ".join(text_without_vdb[:8])
        )

    for tracking_file in ("kv_store_entity_chunks.json", "kv_store_relation_chunks.json"):
        tracking = read_json_dict(storage_dir / tracking_file)
        missing_refs: set[str] = set()
        for value in tracking.values():
            if not isinstance(value, dict):
                continue
            for chunk_id in value.get("chunk_ids", []):
                chunk_id = str(chunk_id)
                if chunk_id and chunk_id not in text_chunks:
                    missing_refs.add(chunk_id)
        if missing_refs:
            warnings.append(
                f"{tracking_file} references ids missing from text_chunks: "
                + ", ".join(sorted(missing_refs)[:8])
            )

    ready = not warnings
    if not ready:
        logger.warning(
            "Document storage readiness failed expected_document_id=%s safe_document_key=%s storage_dir=%s doc_status_refs=%s polluted_doc_status_files=%s text_chunk_files=%s polluted_text_chunk_files=%s chunks_count=%s missing_text_chunks_count=%s wrong_file_chunks_count=%s empty_text_chunks_count=%s missing_vdb_count=%s vdb_without_text_count=%s text_without_vdb_count=%s warnings=%s",
            filename,
            safe_document_key(filename),
            storage_dir,
            doc_status_refs,
            sorted(other_doc_status_files),
            sorted(all_text_chunk_files),
            sorted(text_chunk_files),
            chunks_count,
            len(missing_text_chunks),
            len(wrong_file_chunks),
            len(empty_text_chunks),
            len(missing_vdb),
            len(vdb_without_text),
            len(text_without_vdb),
            warnings,
        )
    return StorageReadiness(
        ready=ready,
        status="ready_for_chat" if ready else "partial_success",
        warnings=warnings,
        storage_dir=storage_dir,
        chunks_count=chunks_count,
    )


def document_storage_ready(document_id: str) -> bool:
    return document_storage_readiness(document_id).ready


def file_name_set_from_doc_statuses(doc_statuses: dict[str, Any]) -> set[str]:
    file_names: set[str] = set()
    for status in doc_statuses.values():
        if not isinstance(status, dict):
            continue
        file_path = status.get("file_path")
        if isinstance(file_path, str) and file_path.strip():
            file_names.add(Path(file_path).name)
    return file_names


def file_name_set_from_text_chunks(text_chunks: dict[str, Any]) -> set[str]:
    file_names: set[str] = set()
    for chunk in text_chunks.values():
        if not isinstance(chunk, dict):
            continue
        file_path = chunk.get("file_path")
        if isinstance(file_path, str) and file_path.strip():
            file_names.add(Path(file_path).name)
    return file_names


def file_name_set_from_vdb_chunks(storage_dir: Path) -> set[str]:
    data = read_json_dict(storage_dir / "vdb_chunks.json")
    records = data.get("data", []) if isinstance(data.get("data"), list) else []
    file_names: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        file_path = record.get("file_path")
        if isinstance(file_path, str) and file_path.strip():
            file_names.add(Path(file_path).name)
    return file_names


def document_storage_debug_payload(document_id: str) -> dict[str, Any]:
    filename = safe_document_id(document_id)
    safe_key = safe_document_key(filename)
    storage_dir = document_storage_dir(filename)
    doc_statuses = read_json_dict(storage_dir / "kv_store_doc_status.json")
    text_chunks = read_json_dict(storage_dir / "kv_store_text_chunks.json")
    doc_status_file_ids = file_name_set_from_doc_statuses(doc_statuses)
    text_chunk_file_ids = file_name_set_from_text_chunks(text_chunks)
    vdb_chunk_file_ids = file_name_set_from_vdb_chunks(storage_dir)
    polluted_doc_status_ids = sorted(doc_status_file_ids - {filename})
    polluted_text_chunk_ids = sorted(text_chunk_file_ids - {filename})
    polluted_vdb_chunk_ids = sorted(vdb_chunk_file_ids - {filename})
    readiness = document_storage_readiness(filename)

    return {
        "document_id": filename,
        "safe_document_key": safe_key,
        "storage_dir": str(storage_dir),
        "doc_status_document_ids": sorted(doc_statuses.keys()),
        "doc_status_file_ids": sorted(doc_status_file_ids),
        "text_chunk_file_ids": sorted(text_chunk_file_ids),
        "vdb_chunk_file_ids": sorted(vdb_chunk_file_ids),
        "polluted_ids": sorted(
            set(polluted_doc_status_ids)
            | set(polluted_text_chunk_ids)
            | set(polluted_vdb_chunk_ids)
        ),
        "polluted_doc_status_ids": polluted_doc_status_ids,
        "polluted_text_chunk_ids": polluted_text_chunk_ids,
        "polluted_vdb_chunk_ids": polluted_vdb_chunk_ids,
        "chunks_count": readiness.chunks_count,
        "warnings": readiness.warnings,
    }
