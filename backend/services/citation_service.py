from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from backend.config import as_int
from backend.core.ids import safe_document_id, same_document_file
from backend.core.paths import (
    content_list_path,
    document_path,
    document_storage_dir,
    load_content_list,
)
from backend.core.text import first_non_empty_text
from backend.rag.storage import read_json_dict
from backend.schemas import SourceItem

logger = logging.getLogger("api_server")

SOURCE_MAP_VERSION = 1
SOURCE_MAP_FILENAME = "source_map.v1.json"
SOURCE_MAP_TMP_FILENAME = "source_map.v1.tmp.json"
DEFAULT_SOURCE_LIMIT = 8
CITATION_ORIGIN_KEY = "_citation_origin"
CITATION_ORIGIN_RETRIEVAL = "retrieval_context"
CITATION_ORIGIN_STORAGE_FALLBACK = "storage_fallback"


def source_map_path(document_id: str) -> Path:
    return document_storage_dir(safe_document_id(document_id)) / SOURCE_MAP_FILENAME


def load_source_map(document_id: str) -> dict[str, Any] | None:
    path = source_map_path(document_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load source map document_id=%s path=%s error=%s", document_id, path, exc)
        return None
    if not isinstance(data, dict) or data.get("version") != SOURCE_MAP_VERSION:
        return None
    chunks = data.get("chunks")
    return data if isinstance(chunks, dict) else None


def build_or_load_source_map(document_id: str) -> dict[str, Any]:
    cached = load_source_map(document_id)
    if cached is not None:
        return cached
    return build_source_map(document_id)


def build_source_map(document_id: str) -> dict[str, Any]:
    normalized_document_id = safe_document_id(document_id)
    storage_dir = document_storage_dir(normalized_document_id)
    text_chunks = read_json_dict(storage_dir / "kv_store_text_chunks.json")
    entries = content_entries_for_document(normalized_document_id)
    index = build_content_index(entries)

    mapped_chunks: dict[str, dict[str, Any]] = {}
    for raw_chunk_id, raw_chunk in text_chunks.items():
        if not isinstance(raw_chunk, dict):
            continue
        if raw_chunk.get("file_path") and not same_document_file(
            raw_chunk.get("file_path"),
            normalized_document_id,
        ):
            continue
        chunk_id = str(raw_chunk.get("_id") or raw_chunk.get("id") or raw_chunk_id)
        content = str(raw_chunk.get("content") or "").strip()
        mapped_chunks[chunk_id] = match_chunk_to_content(
            normalized_document_id,
            chunk_id,
            content,
            index,
        )

    payload = {
        "version": SOURCE_MAP_VERSION,
        "document_id": normalized_document_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "chunks_count": len(mapped_chunks),
        "content_items_count": len(entries),
        "chunks": mapped_chunks,
    }
    write_source_map_atomic(normalized_document_id, payload)
    return payload


def write_source_map_atomic(document_id: str, payload: dict[str, Any]) -> None:
    storage_dir = document_storage_dir(document_id)
    storage_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = storage_dir / SOURCE_MAP_TMP_FILENAME
    final_path = storage_dir / SOURCE_MAP_FILENAME
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(final_path)


def content_entries_for_document(document_id: str) -> list[dict[str, Any]]:
    try:
        pdf_path = document_path(document_id)
    except Exception as exc:
        logger.warning("Cannot resolve document for source map document_id=%s error=%s", document_id, exc)
        return []
    path = content_list_path(pdf_path)
    if path is None:
        return []
    try:
        items = load_content_list(path)
    except Exception as exc:
        logger.warning("Cannot load content_list for source map document_id=%s path=%s error=%s", document_id, path, exc)
        return []

    entries: list[dict[str, Any]] = []
    content_root = path.parent
    for index, item in enumerate(items):
        item_type = str(item.get("type") or "text")
        text = content_item_text(item)
        image_path = item.get("img_path")
        if not text and isinstance(image_path, str) and image_path.strip():
            text = image_path.strip()
        if not text:
            continue
        normalized = normalize_text(text)
        if not normalized:
            continue
        entry = {
            "content_index": index,
            "type": canonical_source_type(item_type),
            "page": page_from_content_item(item),
            "text": text,
            "normalized": normalized,
            "hash": text_hash(normalized),
        }
        if isinstance(image_path, str) and image_path.strip():
            entry["image_keys"] = image_keys(image_path, content_root)
        entries.append(entry)
    return entries


def build_content_index(entries: list[dict[str, Any]]) -> dict[str, Any]:
    parts: list[str] = []
    spans: list[dict[str, Any]] = []
    hashes: dict[str, dict[str, Any]] = {}
    image_lookup: dict[str, dict[str, Any]] = {}
    cursor = 0

    for entry in entries:
        normalized = str(entry["normalized"])
        if parts:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(normalized)
        cursor += len(normalized)
        spans.append({"start": start, "end": cursor, "entry": entry})
        hashes.setdefault(str(entry["hash"]), entry)
        for key in entry.get("image_keys", []):
            image_lookup.setdefault(str(key), entry)

    return {
        "entries": entries,
        "document_text": "".join(parts),
        "spans": spans,
        "hashes": hashes,
        "image_lookup": image_lookup,
    }


def match_chunk_to_content(
    document_id: str,
    chunk_id: str,
    content: str,
    index: dict[str, Any],
) -> dict[str, Any]:
    preview = clip_text(content)
    if not content:
        return fallback_mapping(document_id, chunk_id, preview)

    image_entry = match_image_entry(content, index.get("image_lookup", {}))
    if image_entry:
        return mapping_from_entries(
            document_id,
            chunk_id,
            [image_entry],
            content,
            match_method="metadata",
            match_score=1.0,
        )

    normalized = normalize_text(content)
    if not normalized:
        return fallback_mapping(document_id, chunk_id, preview)

    hashed = index.get("hashes", {}).get(text_hash(normalized))
    if hashed:
        return mapping_from_entries(
            document_id,
            chunk_id,
            [hashed],
            content,
            match_method="hash",
            match_score=1.0,
        )

    document_text = str(index.get("document_text") or "")
    if document_text:
        offset = document_text.find(normalized)
        method = "substring"
        score = 1.0
        span_length = len(normalized)
        if offset < 0:
            probe = normalized[: min(600, len(normalized))]
            offset = document_text.find(probe) if len(probe) >= 80 else -1
            method = "substring"
            score = 0.85
            span_length = len(probe)
        if offset >= 0:
            entries = entries_for_span(
                index.get("spans", []),
                offset,
                offset + span_length,
            )
            if entries:
                return mapping_from_entries(
                    document_id,
                    chunk_id,
                    entries,
                    content,
                    match_method=method,
                    match_score=score,
                )

    fuzzy_entry, fuzzy_score = best_fuzzy_entry(normalized, index.get("entries", []))
    if fuzzy_entry and fuzzy_score >= 0.35:
        return mapping_from_entries(
            document_id,
            chunk_id,
            [fuzzy_entry],
            content,
            match_method="fuzzy",
            match_score=fuzzy_score,
        )

    return fallback_mapping(document_id, chunk_id, preview)


def map_retrieved_chunks_to_sources(
    document_id: str,
    document_name: str,
    retrieved_chunks: list[dict[str, Any]],
    limit: int | None = None,
) -> list[SourceItem]:
    normalized_document_id = safe_document_id(document_id)
    try:
        source_map = build_or_load_source_map(normalized_document_id)
    except Exception as exc:
        logger.warning("Citation source map unavailable document_id=%s error=%s", normalized_document_id, exc)
        source_map = {"chunks": {}}

    chunks_by_id = source_map.get("chunks", {}) if isinstance(source_map, dict) else {}
    source_limit = limit or as_int(os.getenv("CHAT_SOURCES_LIMIT"), DEFAULT_SOURCE_LIMIT)
    sources: list[SourceItem] = []
    seen: set[tuple[str | None, str | None, int | None, str]] = set()

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if not isinstance(chunk, dict):
            continue
        chunk_id = chunk_identifier(chunk)
        mapping = chunks_by_id.get(chunk_id) if chunk_id else None
        source = source_from_chunk(
            normalized_document_id,
            document_name,
            chunk,
            mapping if isinstance(mapping, dict) else None,
            rank,
        )
        key = (
            source.document_id,
            source.chunk_id,
            source.page,
            source.text[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        sources.append(source)
        if len(sources) >= source_limit:
            break

    return sources


async def collect_sources_for_query(
    rag: Any,
    document_id: str,
    document_name: str,
    query: str,
    query_kwargs: dict[str, Any],
) -> list[SourceItem]:
    try:
        chunks = await retrieved_chunks_from_lightrag(rag, query, query_kwargs)
        if not chunks:
            chunks = fallback_chunks_from_storage(document_id, query_kwargs)
        return map_retrieved_chunks_to_sources(document_id, document_name, chunks)
    except Exception as exc:
        logger.warning("Citation collection failed document_id=%s error=%s", document_id, exc)
        return []


def fast_text_chunks_from_storage(
    document_id: str,
    query: str,
    query_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    chunk_limit = int((query_kwargs or {}).get("chunk_top_k") or DEFAULT_SOURCE_LIMIT)
    chunk_limit = max(chunk_limit, DEFAULT_SOURCE_LIMIT)
    chunks = storage_chunks_for_document(document_id)
    terms = query_terms_for_storage_rank(query)
    if not chunks or not terms:
        return chunks[:chunk_limit]

    scored_chunks: list[tuple[float, int, int, dict[str, Any]]] = []
    for index, chunk in enumerate(chunks):
        score = storage_chunk_query_score(str(chunk.get("content") or ""), terms)
        if score <= 0:
            continue
        chunk_order = optional_int(chunk.get("chunk_order_index")) or index
        scored_chunks.append((score, chunk_order, index, dict(chunk)))

    if not scored_chunks:
        return chunks[:chunk_limit]

    scored_chunks.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [chunk for _score, _order, _index, chunk in scored_chunks[:chunk_limit]]


def storage_chunks_for_document(document_id: str) -> list[dict[str, Any]]:
    normalized_document_id = safe_document_id(document_id)
    storage_dir = document_storage_dir(normalized_document_id)
    text_chunks = read_json_dict(storage_dir / "kv_store_text_chunks.json")
    chunks: list[dict[str, Any]] = []
    for chunk_id, chunk in text_chunks.items():
        if not isinstance(chunk, dict):
            continue
        if chunk.get("file_path") and not same_document_file(
            chunk.get("file_path"),
            normalized_document_id,
        ):
            continue
        chunks.append(
            {
                "chunk_id": chunk.get("_id") or chunk.get("id") or chunk_id,
                "content": chunk.get("content", ""),
                "file_path": chunk.get("file_path"),
                "chunk_order_index": chunk.get("chunk_order_index", 0),
                CITATION_ORIGIN_KEY: CITATION_ORIGIN_STORAGE_FALLBACK,
            }
        )
    chunks.sort(key=lambda item: optional_int(item.get("chunk_order_index")) or 0)
    return chunks


def query_terms_for_storage_rank(query: str) -> list[str]:
    normalized = normalize_text(query)
    terms: set[str] = set()
    terms.update(re.findall(r"[a-z0-9_]{2,}", normalized))
    for token in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        terms.add(token)
        for gram_size in (2, 3):
            for index in range(0, max(len(token) - gram_size + 1, 0)):
                terms.add(token[index : index + gram_size])
    return sorted(terms, key=lambda term: (-len(term), term))


def storage_chunk_query_score(content: str, terms: list[str]) -> float:
    normalized = normalize_text(content)
    if not normalized:
        return 0.0
    score = 0.0
    for term in terms:
        occurrences = normalized.count(term)
        if occurrences:
            score += occurrences * min(len(term), 8)
    return score


async def retrieved_chunks_from_lightrag(
    rag: Any,
    query: str,
    query_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    lightrag = getattr(rag, "lightrag", None)
    if lightrag is None or not callable(getattr(lightrag, "aquery_data", None)):
        return []

    try:
        from lightrag import QueryParam
    except Exception as exc:
        logger.warning("LightRAG QueryParam unavailable for citation collection: %s", exc)
        return []

    param_kwargs = {
        key: value
        for key, value in query_kwargs.items()
        if key
        in {
            "mode",
            "top_k",
            "chunk_top_k",
            "max_entity_tokens",
            "max_relation_tokens",
            "max_total_tokens",
            "enable_rerank",
        }
    }
    try:
        raw_data = await lightrag.aquery_data(query, QueryParam(**param_kwargs))
    except Exception as exc:
        logger.warning("LightRAG aquery_data citation pass failed: %s", exc)
        return []

    return chunks_from_raw_data(raw_data)


def chunks_from_raw_data(raw_data: Any) -> list[dict[str, Any]]:
    if isinstance(raw_data, list):
        chunks = raw_data
    elif isinstance(raw_data, dict):
        data = raw_data.get("data")
        if isinstance(data, dict) and isinstance(data.get("chunks"), list):
            chunks = data["chunks"]
        elif isinstance(raw_data.get("chunks"), list):
            chunks = raw_data["chunks"]
        elif isinstance(data, list):
            chunks = data
        else:
            chunks = []
    else:
        chunks = []
    return [chunk for chunk in chunks if isinstance(chunk, dict)]


def fallback_chunks_from_storage(
    document_id: str,
    query_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    chunk_top_k = int((query_kwargs or {}).get("chunk_top_k") or DEFAULT_SOURCE_LIMIT)
    chunks = storage_chunks_for_document(document_id)
    return chunks[: max(chunk_top_k, DEFAULT_SOURCE_LIMIT)]


def source_from_chunk(
    document_id: str,
    document_name: str,
    chunk: dict[str, Any],
    mapping: dict[str, Any] | None,
    rank: int,
) -> SourceItem:
    chunk_id = chunk_identifier(chunk)
    text = clip_text(str((mapping or {}).get("text") or chunk.get("content") or ""))
    score, score_type = score_from_chunk(chunk)
    citation_mode = citation_mode_from_chunk(chunk)
    return SourceItem(
        id=f"{document_id}:{chunk_id or rank}:{rank}",
        type=str((mapping or {}).get("type") or "text"),
        page=optional_int((mapping or {}).get("page")),
        page_end=optional_int((mapping or {}).get("page_end")),
        text=text,
        document_id=document_id,
        document_name=document_name,
        chunk_id=chunk_id,
        content_index=optional_int((mapping or {}).get("content_index")),
        rank=rank,
        score=score,
        score_type=score_type,
        match_score=optional_float((mapping or {}).get("match_score")),
        match_method=str((mapping or {}).get("match_method") or "fallback"),
        citation_mode=citation_mode,
    )


def score_from_chunk(chunk: dict[str, Any]) -> tuple[float | None, str]:
    if citation_mode_from_chunk(chunk) == CITATION_ORIGIN_STORAGE_FALLBACK:
        return None, "storage_fallback"
    for key, score_type in (
        ("rerank_score", "rerank"),
        ("relevance_score", "rerank"),
        ("score", "match"),
    ):
        score = optional_float(chunk.get(key))
        if score is not None:
            return score, score_type
    return None, "retrieval_rank"


def citation_mode_from_chunk(chunk: dict[str, Any]) -> str:
    if chunk.get(CITATION_ORIGIN_KEY) == CITATION_ORIGIN_STORAGE_FALLBACK:
        return CITATION_ORIGIN_STORAGE_FALLBACK
    return CITATION_ORIGIN_RETRIEVAL


def chunk_identifier(chunk: dict[str, Any]) -> str | None:
    for key in ("chunk_id", "id", "_id", "__id__"):
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def content_item_text(item: dict[str, Any]) -> str:
    return first_non_empty_text(
        item,
        "text",
        "table_body",
        "latex",
        "equation",
        "content",
        "caption",
        "image_caption",
        "img_caption",
        "table_caption",
        "equation_caption",
    )


def canonical_source_type(item_type: str) -> str:
    normalized = item_type.strip().lower()
    if normalized in {"text", "image", "table", "equation"}:
        return normalized
    if "formula" in normalized:
        return "equation"
    return "text"


def page_from_content_item(item: dict[str, Any]) -> int | None:
    page_idx = item.get("page_idx")
    if isinstance(page_idx, int) and not isinstance(page_idx, bool) and page_idx >= 0:
        return page_idx + 1
    if isinstance(page_idx, str) and page_idx.isdigit():
        return int(page_idx) + 1
    page = item.get("page")
    if isinstance(page, int) and not isinstance(page, bool) and page > 0:
        return page
    if isinstance(page, str) and page.isdigit() and int(page) > 0:
        return int(page)
    return None


def mapping_from_entries(
    document_id: str,
    chunk_id: str,
    entries: list[dict[str, Any]],
    chunk_text: str,
    *,
    match_method: str,
    match_score: float,
) -> dict[str, Any]:
    pages = [
        page
        for page in (optional_int(entry.get("page")) for entry in entries)
        if page is not None
    ]
    first_entry = entries[0]
    return {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "type": first_entry.get("type") or "text",
        "page": min(pages) if pages else None,
        "page_end": max(pages) if len(pages) > 1 else (pages[0] if pages else None),
        "content_index": first_entry.get("content_index"),
        "content_indices": [entry.get("content_index") for entry in entries],
        "text": clip_text(chunk_text or str(first_entry.get("text") or "")),
        "match_score": round(match_score, 4),
        "match_method": match_method,
    }


def fallback_mapping(document_id: str, chunk_id: str, preview: str) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "type": "text",
        "page": None,
        "page_end": None,
        "content_index": None,
        "text": preview,
        "match_score": None,
        "match_method": "fallback",
    }


def entries_for_span(
    spans: list[dict[str, Any]],
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for span in spans:
        if span["end"] <= start:
            continue
        if span["start"] >= end:
            break
        result.append(span["entry"])
    return result


def best_fuzzy_entry(
    normalized: str,
    entries: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, float]:
    if not normalized or not entries:
        return None, 0.0
    probe = normalized[:1200]
    best_entry: dict[str, Any] | None = None
    best_score = 0.0
    for entry in entries:
        candidate = str(entry.get("normalized") or "")
        if not candidate:
            continue
        score = SequenceMatcher(None, probe, candidate[:1200]).ratio()
        if score > best_score:
            best_score = score
            best_entry = entry
    return best_entry, best_score


def match_image_entry(
    content: str,
    image_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not image_lookup:
        return None
    match = re.search(r"Image Path:\s*(.+)", content)
    if not match:
        return None
    raw_path = match.group(1).strip()
    for key in image_keys(raw_path):
        entry = image_lookup.get(key)
        if entry:
            return entry
    return None


def image_keys(raw_path: str, content_root: Path | None = None) -> list[str]:
    keys: list[str] = []
    path = Path(raw_path)
    candidates = [path]
    if content_root is not None and not path.is_absolute():
        candidates.append(content_root / path)
    for candidate in candidates:
        text = os.path.normcase(str(candidate).replace("\\", "/")).lower()
        name = os.path.normcase(candidate.name).lower()
        if text:
            keys.append(text)
        if name:
            keys.append(name)
    return list(dict.fromkeys(keys))


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").replace("\u3000", " ").split()).lower()


def text_hash(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def clip_text(text: str, limit: int = 600) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
