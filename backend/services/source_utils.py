from __future__ import annotations

from backend.schemas import SourceItem


def dedupe_sources(sources: list[SourceItem]) -> list[SourceItem]:
    seen: set[tuple[str | None, str | None, int | None, str]] = set()
    deduped: list[SourceItem] = []
    for source in sources:
        key = (
            source.document_id,
            source.chunk_id,
            source.page,
            source.text[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped
