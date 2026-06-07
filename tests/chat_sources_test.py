from __future__ import annotations

from backend.schemas import SourceItem
from backend.services.source_utils import dedupe_sources


def test_dedupe_sources_keeps_distinct_documents() -> None:
    sources = [
        SourceItem(
            id="a-1",
            document_id="a.pdf",
            document_name="A",
            chunk_id="chunk-1",
            page=1,
            text="same text",
        ),
        SourceItem(
            id="a-dup",
            document_id="a.pdf",
            document_name="A",
            chunk_id="chunk-1",
            page=1,
            text="same text",
        ),
        SourceItem(
            id="b-1",
            document_id="b.pdf",
            document_name="B",
            chunk_id="chunk-1",
            page=1,
            text="same text",
        ),
    ]

    deduped = dedupe_sources(sources)

    assert [source.id for source in deduped] == ["a-1", "b-1"]


def test_source_item_serialization_keeps_citation_metadata() -> None:
    source = SourceItem(
        id="doc1:c1:1",
        type="text",
        page=2,
        page_end=3,
        text="evidence text",
        document_id="doc1",
        document_name="Doc 1",
        chunk_id="c1",
        content_index=4,
        rank=1,
        score_type="storage_fallback",
        match_method="fallback",
        citation_mode="storage_fallback",
    )

    dumped = source.model_dump()
    restored = SourceItem(**dumped)

    assert restored.document_id == "doc1"
    assert restored.chunk_id == "c1"
    assert restored.page_end == 3
    assert restored.score_type == "storage_fallback"
    assert restored.citation_mode == "storage_fallback"
