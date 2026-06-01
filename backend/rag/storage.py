from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from raganything import RAGAnything

from backend.config import RAG_STORAGE_DIR
from backend.core.ids import safe_document_id, safe_document_key, same_document_file
from backend.core.paths import (
    document_storage_dir,
    is_relative_to_path,
    lightrag_workspace,
    process_failure_path,
)

logger = logging.getLogger("api_server")

def clear_lightrag_shared_workspace(workspace: str) -> None:
    try:
        from lightrag.kg import shared_storage
    except Exception as exc:
        logger.warning(
            "process_clear_shared_workspace_failed workspace=%s error=%s",
            workspace,
            exc,
        )
        return

    prefix = f"{workspace}:"
    removed_keys: list[str] = []
    for attr_name in ("_shared_dicts", "_init_flags", "_update_flags"):
        mapping = getattr(shared_storage, attr_name, None)
        if mapping is None:
            continue
        for key in list(mapping.keys()):
            key_text = str(key)
            if key_text == workspace or key_text.startswith(prefix):
                try:
                    del mapping[key]
                except Exception:
                    try:
                        mapping.pop(key, None)
                    except Exception as exc:
                        logger.warning(
                            "process_clear_shared_workspace_key_failed workspace=%s mapping=%s key=%s error=%s",
                            workspace,
                            attr_name,
                            key_text,
                            exc,
                        )
                        continue
                removed_keys.append(f"{attr_name}:{key_text}")

    logger.info(
        "process_clear_shared_workspace_done workspace=%s removed_keys=%s",
        workspace,
        removed_keys,
    )


def clear_all_lightrag_shared_storage() -> None:
    try:
        from lightrag.kg import shared_storage
    except Exception as exc:
        logger.warning("clear_lightrag_shared_storage_failed error=%s", exc)
        return

    for attr_name in ("_shared_dicts", "_init_flags", "_update_flags"):
        mapping = getattr(shared_storage, attr_name, None)
        if mapping is not None:
            try:
                mapping.clear()
            except Exception as exc:
                logger.warning(
                    "clear_lightrag_shared_storage_mapping_failed mapping=%s error=%s",
                    attr_name,
                    exc,
                )
    try:
        shared_storage.set_default_workspace(None)
    except Exception as exc:
        logger.warning("clear_lightrag_default_workspace_failed error=%s", exc)


def set_lightrag_default_workspace(workspace: str) -> None:
    try:
        from lightrag.kg.shared_storage import set_default_workspace

        set_default_workspace(workspace)
    except Exception as exc:
        logger.warning(
            "set_lightrag_default_workspace_failed workspace=%s error=%s",
            workspace,
            exc,
        )


def ensure_document_storage_reset(document_id: str) -> Path:
    normalized_document_id = safe_document_id(document_id)
    storage_root = (RAG_STORAGE_DIR / "documents").resolve(strict=False)
    storage_dir = document_storage_dir(normalized_document_id)
    resolved_storage_dir = storage_dir.resolve(strict=False)
    if not is_relative_to_path(resolved_storage_dir, storage_root):
        raise RuntimeError(f"Unsafe document storage path: {storage_dir}")

    logger.info(
        "process_clean_storage_start document_id=%s safe_document_key=%s storage_dir=%s",
        normalized_document_id,
        safe_document_key(normalized_document_id),
        storage_dir,
    )
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_exists_after_delete = storage_dir.exists()
    logger.info(
        "process_clean_storage_done document_id=%s safe_document_key=%s storage_dir=%s storage_exists_after_delete=%s",
        normalized_document_id,
        safe_document_key(normalized_document_id),
        storage_dir,
        str(storage_exists_after_delete).lower(),
    )
    if storage_exists_after_delete:
        raise RuntimeError(f"Failed to delete document storage: {storage_dir}")

    clear_lightrag_shared_workspace(lightrag_workspace(normalized_document_id))
    storage_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Prepared empty document storage document_id=%s safe_document_key=%s storage_dir=%s",
        normalized_document_id,
        safe_document_key(normalized_document_id),
        storage_dir,
    )
    return storage_dir


def write_process_failure(document_id: str, message: str) -> None:
    storage_dir = document_storage_dir(document_id)
    storage_dir.mkdir(parents=True, exist_ok=True)
    payload = {"message": message, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00")}
    try:
        process_failure_path(document_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to write process failure marker for %s: %s", document_id, exc)


def clear_process_failure(document_id: str) -> None:
    marker = process_failure_path(document_id)
    if marker.exists():
        try:
            marker.unlink()
        except Exception as exc:
            logger.warning("Failed to clear process failure marker for %s: %s", document_id, exc)


def vdb_records_by_file(storage_dir: Path, filename: str) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    data = read_json_dict(storage_dir / "vdb_chunks.json")
    records = data.get("data", []) if isinstance(data.get("data"), list) else []
    current_records: list[dict[str, Any]] = []
    polluted_files: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        file_path = record.get("file_path")
        if same_document_file(file_path, filename):
            current_records.append(record)
        elif isinstance(file_path, str) and file_path.strip():
            polluted_files.add(Path(file_path).name)
    if polluted_files:
        warnings.append(
            "vdb_chunks contains chunks from other documents: "
            + ", ".join(sorted(polluted_files))
        )
    return current_records, warnings


def read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read JSON storage file %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


async def persist_lightrag_storages(rag: RAGAnything) -> None:
    if rag.lightrag is None:
        return
    insert_done = getattr(rag.lightrag, "_insert_done", None)
    if callable(insert_done):
        await insert_done()
        return

    storage_names = (
        "full_docs",
        "doc_status",
        "text_chunks",
        "full_entities",
        "full_relations",
        "entity_chunks",
        "relation_chunks",
        "llm_response_cache",
        "entities_vdb",
        "relationships_vdb",
        "chunks_vdb",
        "chunk_entity_relation_graph",
    )
    await asyncio.gather(
        *[
            getattr(rag.lightrag, name).index_done_callback()
            for name in storage_names
            if getattr(rag.lightrag, name, None) is not None
        ]
    )
