from __future__ import annotations

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
