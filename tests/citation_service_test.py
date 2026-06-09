from __future__ import annotations

import json

from backend.schemas import SourceItem
from backend.services import citation_service as citations


def _entry(index: int, text: str, page: int, item_type: str = "text") -> dict:
    normalized = citations.normalize_text(text)
    return {
        "content_index": index,
        "type": item_type,
        "page": page,
        "text": text,
        "normalized": normalized,
        "hash": citations.text_hash(normalized),
    }


def test_substring_match_maps_chunk_to_page_range() -> None:
    entries = [
        _entry(0, "first page concept", 1),
        _entry(1, "second page detail", 2),
    ]
    index = citations.build_content_index(entries)

    mapping = citations.match_chunk_to_content(
        "sample.pdf",
        "chunk-1",
        "first page concept\n\nsecond page detail",
        index,
    )

    assert mapping["page"] == 1
    assert mapping["page_end"] == 2
    assert mapping["match_method"] == "substring"


def test_long_substring_match_uses_full_span_for_page_range() -> None:
    first = "alpha " * 90
    second = "beta " * 90
    third = "gamma " * 90
    entries = [
        _entry(0, first, 1),
        _entry(1, second, 2),
        _entry(2, third, 3),
    ]
    index = citations.build_content_index(entries)

    mapping = citations.match_chunk_to_content(
        "sample.pdf",
        "chunk-long",
        f"{first}\n\n{second}\n\n{third}",
        index,
    )

    assert mapping["page"] == 1
    assert mapping["page_end"] == 3
    assert mapping["match_method"] == "substring"


def test_image_path_metadata_match_maps_to_image_page() -> None:
    entry = _entry(0, "figure caption", 3, "image")
    entry["image_keys"] = citations.image_keys("images/figure-a.png")
    index = citations.build_content_index([entry])

    mapping = citations.match_chunk_to_content(
        "sample.pdf",
        "chunk-image",
        "Image Content Analysis:\nImage Path: C:\\tmp\\figure-a.png\nCaptions: figure caption",
        index,
    )

    assert mapping["page"] == 3
    assert mapping["type"] == "image"
    assert mapping["match_method"] == "metadata"


def test_content_entries_keep_image_without_caption(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        citations,
        "document_path",
        lambda _document_id: tmp_path / "sample.pdf",
    )
    monkeypatch.setattr(
        citations,
        "content_list_path",
        lambda _pdf_path: tmp_path / "sample_content_list.json",
    )
    monkeypatch.setattr(
        citations,
        "load_content_list",
        lambda _path: [
            {
                "type": "image",
                "img_path": "images/figure-a.png",
                "page_idx": 4,
            }
        ],
    )

    entries = citations.content_entries_for_document("sample.pdf")
    index = citations.build_content_index(entries)
    mapping = citations.match_chunk_to_content(
        "sample.pdf",
        "chunk-image",
        "Image Content Analysis:\nImage Path: C:\\tmp\\figure-a.png",
        index,
    )

    assert len(entries) == 1
    assert entries[0]["type"] == "image"
    assert entries[0]["page"] == 5
    assert mapping["page"] == 5
    assert mapping["match_method"] == "metadata"


def test_build_source_map_writes_final_file_atomically(tmp_path, monkeypatch) -> None:
    entries = [_entry(0, "alpha beta", 1)]

    monkeypatch.setattr(citations, "document_storage_dir", lambda _document_id: tmp_path)
    monkeypatch.setattr(citations, "content_entries_for_document", lambda _document_id: entries)
    monkeypatch.setattr(
        citations,
        "read_json_dict",
        lambda _path: {
            "chunk-alpha": {
                "_id": "chunk-alpha",
                "content": "alpha beta",
                "file_path": "sample.pdf",
            }
        },
    )

    payload = citations.build_source_map("sample.pdf")

    assert payload["chunks"]["chunk-alpha"]["page"] == 1
    assert (tmp_path / citations.SOURCE_MAP_FILENAME).exists()
    assert not (tmp_path / citations.SOURCE_MAP_TMP_FILENAME).exists()


def test_build_source_map_overwrites_stale_cached_map(tmp_path, monkeypatch) -> None:
    stale_payload = {
        "version": citations.SOURCE_MAP_VERSION,
        "chunks": {
            "chunk-alpha": {
                "page": 99,
                "text": "stale",
            }
        },
    }
    (tmp_path / citations.SOURCE_MAP_FILENAME).write_text(
        json.dumps(stale_payload),
        encoding="utf-8",
    )

    monkeypatch.setattr(citations, "document_storage_dir", lambda _document_id: tmp_path)
    monkeypatch.setattr(
        citations,
        "content_entries_for_document",
        lambda _document_id: [_entry(0, "fresh content", 2)],
    )
    monkeypatch.setattr(
        citations,
        "read_json_dict",
        lambda _path: {
            "chunk-alpha": {
                "_id": "chunk-alpha",
                "content": "fresh content",
                "file_path": "sample.pdf",
            }
        },
    )

    payload = citations.build_source_map("sample.pdf")
    persisted = json.loads(
        (tmp_path / citations.SOURCE_MAP_FILENAME).read_text(encoding="utf-8")
    )

    assert payload["chunks"]["chunk-alpha"]["page"] == 2
    assert persisted["chunks"]["chunk-alpha"]["page"] == 2


def test_map_retrieved_chunks_falls_back_when_source_map_fails(monkeypatch) -> None:
    def fail_source_map(_document_id: str) -> dict:
        raise RuntimeError("missing source map")

    monkeypatch.setattr(citations, "build_or_load_source_map", fail_source_map)

    sources = citations.map_retrieved_chunks_to_sources(
        "sample.pdf",
        "Sample",
        [{"chunk_id": "chunk-1", "content": "retrieved text"}],
    )

    assert len(sources) == 1
    assert sources[0].page is None
    assert sources[0].chunk_id == "chunk-1"
    assert sources[0].match_method == "fallback"
    assert sources[0].score_type == "retrieval_rank"


def test_source_item_keeps_citation_metadata() -> None:
    source = SourceItem(
        id="doc1:c1:1",
        type="text",
        page=3,
        text="hello",
        document_id="doc1",
        document_name="test.pdf",
        chunk_id="c1",
        rank=1,
        match_method="fallback",
        citation_mode="retrieval_context",
    )

    assert source.document_id == "doc1"
    assert source.chunk_id == "c1"
    assert source.rank == 1
    assert source.match_method == "fallback"
    assert source.citation_mode == "retrieval_context"


def test_retrieval_chunk_without_score_uses_retrieval_rank() -> None:
    source = citations.source_from_chunk(
        "sample.pdf",
        "Sample",
        {"chunk_id": "chunk-1", "content": "retrieved text"},
        None,
        1,
    )

    assert source.citation_mode == "retrieval_context"
    assert source.score is None
    assert source.score_type == "retrieval_rank"


def test_storage_fallback_chunk_uses_storage_fallback_score_type() -> None:
    source = citations.source_from_chunk(
        "sample.pdf",
        "Sample",
        {
            "chunk_id": "chunk-1",
            "content": "fallback text",
            citations.CITATION_ORIGIN_KEY: citations.CITATION_ORIGIN_STORAGE_FALLBACK,
        },
        None,
        1,
    )

    assert source.citation_mode == "storage_fallback"
    assert source.score is None
    assert source.score_type == "storage_fallback"


def test_chunks_from_raw_data_accepts_supported_shapes() -> None:
    chunk = {"chunk_id": "chunk-1", "content": "text"}

    assert citations.chunks_from_raw_data({"data": {"chunks": [chunk]}}) == [chunk]
    assert citations.chunks_from_raw_data({"chunks": [chunk]}) == [chunk]
    assert citations.chunks_from_raw_data({"data": [chunk]}) == [chunk]
    assert citations.chunks_from_raw_data([chunk]) == [chunk]
    assert citations.chunks_from_raw_data({"data": {"chunks": ["bad"]}}) == []


def test_fast_text_chunks_from_storage_ranks_keyword_matches(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "kv_store_text_chunks.json").write_text(
        json.dumps(
            {
                "chunk-a": {
                    "_id": "chunk-a",
                    "content": "unrelated introduction",
                    "chunk_order_index": 0,
                },
                "chunk-b": {
                    "_id": "chunk-b",
                    "content": "alpha beta concept and alpha examples",
                    "chunk_order_index": 1,
                },
                "chunk-c": {
                    "_id": "chunk-c",
                    "content": "beta only",
                    "chunk_order_index": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(citations, "document_storage_dir", lambda _document_id: tmp_path)

    chunks = citations.fast_text_chunks_from_storage(
        "sample.pdf",
        "alpha concept",
        {"chunk_top_k": 2},
    )

    assert [chunk["chunk_id"] for chunk in chunks] == ["chunk-b"]
