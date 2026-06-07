from __future__ import annotations

from collections import Counter
from typing import Any

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


def citation_metrics(sources: list[SourceItem]) -> dict[str, Any]:
    total = len(sources)
    sources_with_page = sum(1 for source in sources if source.page is not None)
    citation_mode_counts = Counter(
        source.citation_mode or "unknown" for source in sources
    )
    match_method_counts = Counter(
        source.match_method or "unknown" for source in sources
    )
    fallback_count = citation_mode_counts.get("storage_fallback", 0)
    documents_covered = sorted(
        {
            source.document_id
            for source in sources
            if source.document_id
        }
    )
    return {
        "sources_total": total,
        "sources_with_page": sources_with_page,
        "page_hit_rate": round(sources_with_page / total, 4) if total else 0.0,
        "citation_mode_counts": dict(citation_mode_counts),
        "match_method_counts": dict(match_method_counts),
        "fallback_ratio": round(fallback_count / total, 4) if total else 0.0,
        "documents_covered": documents_covered,
    }
