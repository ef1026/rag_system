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
