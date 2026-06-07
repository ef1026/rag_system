from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import HTTPException

from backend.core.models import DocumentAnswer, DocumentChatContext, StorageReadiness
from backend.schemas import ChatRequest, SourceItem

fake_sentence_transformers = types.ModuleType("sentence_transformers")


class FakeSentenceTransformer:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _text in texts]


fake_sentence_transformers.SentenceTransformer = FakeSentenceTransformer
sys.modules.setdefault("sentence_transformers", fake_sentence_transformers)

from backend.services import retrieval_service as retrieval


class FakeRAG:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.lightrag = object()

    async def _ensure_lightrag_initialized(self) -> dict[str, bool]:
        return {"success": True}

    async def aquery(self, query: str, **kwargs: Any) -> str:
        self.calls.append({"query": query, **kwargs})
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


def make_context(document_id: str, tmp_path: Path) -> DocumentChatContext:
    return DocumentChatContext(
        document_id=document_id,
        name=document_id,
        status="ready_for_chat",
        storage_dir=tmp_path / document_id,
        readiness=StorageReadiness(
            ready=True,
            status="ready_for_chat",
            warnings=[],
            storage_dir=tmp_path / document_id,
        ),
    )


def make_answer(document_id: str, tmp_path: Path) -> DocumentAnswer:
    return DocumentAnswer(
        document_id=document_id,
        name=document_id,
        status="ready_for_chat",
        storage_dir=tmp_path / document_id,
        answer=f"answer {document_id}",
        sources=[
            SourceItem(
                id=f"{document_id}:chunk:1",
                document_id=document_id,
                document_name=document_id,
                chunk_id="chunk",
                page=1,
                text=f"source {document_id}",
            )
        ],
        vlm_image_paths=[],
        fallback_used=False,
        answer_source_path="text",
        prompt_template_used="direct_single_document",
        timings={},
    )


def patch_query_single_document_ready(
    monkeypatch: Any,
    tmp_path: Path,
    fake_rag: FakeRAG,
) -> list[tuple[str, str]]:
    async def fake_document_lock(_document_id: str) -> asyncio.Lock:
        return asyncio.Lock()

    async def fake_get_rag(_document_id: str) -> FakeRAG:
        return fake_rag

    direct_calls: list[tuple[str, str]] = []

    async def fake_direct_context_fallback(
        _rag: FakeRAG,
        question: str,
        document_id: str,
        _level_key: str,
        _level_prompt: str,
        _profile_prompt: str | None = None,
    ) -> tuple[str, str]:
        direct_calls.append((document_id, question))
        return "direct answer", "direct_context_fallback:full_docs"

    async def fake_collect_sources_for_query(**_kwargs: Any) -> list[SourceItem]:
        return []

    monkeypatch.setattr(retrieval, "document_lock", fake_document_lock)
    monkeypatch.setattr(retrieval, "is_document_processing", lambda _document_id: False)
    monkeypatch.setattr(retrieval, "document_path", lambda document_id: tmp_path / document_id)
    monkeypatch.setattr(retrieval, "document_status", lambda _pdf_path: "ready_for_chat")
    monkeypatch.setattr(
        retrieval,
        "document_storage_readiness",
        lambda document_id: StorageReadiness(
            ready=True,
            status="ready_for_chat",
            warnings=[],
            storage_dir=tmp_path / document_id,
        ),
    )
    monkeypatch.setattr(retrieval, "get_rag", fake_get_rag)
    monkeypatch.setattr(retrieval, "configure_vlm_image_registry", lambda *_args: None)
    monkeypatch.setattr(retrieval, "clear_vlm_image_tracking", lambda *_args: None)
    monkeypatch.setattr(retrieval, "current_vlm_image_paths", lambda *_args: [])
    monkeypatch.setattr(
        retrieval,
        "build_rerank_query_config",
        lambda _rag: {
            "enabled": False,
            "label": "disabled",
            "requested": False,
            "model": None,
            "provider": None,
            "model_func_available": False,
            "top_n": None,
            "reason": "test",
        },
    )
    monkeypatch.setattr(
        retrieval,
        "current_rerank_runtime_status",
        lambda: {"last_error": None},
    )
    monkeypatch.setattr(retrieval, "direct_context_fallback", fake_direct_context_fallback)
    monkeypatch.setattr(
        retrieval,
        "collect_sources_for_query",
        fake_collect_sources_for_query,
    )
    return direct_calls


def test_fast_text_empty_answer_skips_duplicate_text_fallback(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    fake_rag = FakeRAG(["empty context"])
    direct_calls = patch_query_single_document_ready(monkeypatch, tmp_path, fake_rag)
    request = ChatRequest(
        question="请总结",
        document_id="a.pdf",
        level="undergraduate",
        mode="hybrid",
        vlm_enhanced=False,
    )

    answer = asyncio.run(
        retrieval.query_single_document(
            request,
            make_context("a.pdf", tmp_path),
            "undergraduate",
            "本科",
            "",
            [],
            False,
        )
    )

    assert len(fake_rag.calls) == 1
    assert fake_rag.calls[0]["vlm_enhanced"] is False
    assert direct_calls == [("a.pdf", "请总结")]
    assert answer.answer_source_path == "direct_context_fallback:full_docs"


def test_auto_empty_answer_allows_one_text_fallback_before_direct_context(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    fake_rag = FakeRAG(["empty context", "empty context"])
    direct_calls = patch_query_single_document_ready(monkeypatch, tmp_path, fake_rag)
    request = ChatRequest(
        question="解释概念",
        document_id="a.pdf",
        level="undergraduate",
        mode="hybrid",
        vlm_enhanced="auto",
    )

    answer = asyncio.run(
        retrieval.query_single_document(
            request,
            make_context("a.pdf", tmp_path),
            "undergraduate",
            "本科",
            "",
            [],
            False,
        )
    )

    assert len(fake_rag.calls) == 2
    assert "vlm_enhanced" not in fake_rag.calls[0]
    assert fake_rag.calls[1]["vlm_enhanced"] is False
    assert direct_calls == [("a.pdf", "解释概念")]
    assert answer.answer_source_path == "direct_context_fallback:full_docs"


def test_summary_multi_document_chat_uses_direct_summary_route(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    contexts = [make_context("a.pdf", tmp_path), make_context("b.pdf", tmp_path)]

    def fail_query_single_document(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("summary route should not call RAG query")

    def fake_map_sources(
        document_id: str,
        document_name: str,
        _chunks: list[dict[str, Any]],
        limit: int | None = None,
    ) -> list[SourceItem]:
        return [
            SourceItem(
                id=f"{document_id}:fallback:1",
                document_id=document_id,
                document_name=document_name,
                chunk_id="fallback",
                page=1,
                text=f"fallback source {document_id}",
                citation_mode="storage_fallback",
                match_method="fallback",
            )
        ][:limit]

    async def fake_synthesize(
        _question: str,
        document_answers: list[DocumentAnswer],
        *_args: Any,
        **_kwargs: Any,
    ) -> str:
        assert [answer.answer_source_path for answer in document_answers] == [
            "direct_summary:full_docs",
            "direct_summary:full_docs",
        ]
        return "summary answer"

    monkeypatch.setattr(retrieval, "validate_chat_document_contexts", lambda _ids: contexts)
    monkeypatch.setattr(
        retrieval,
        "compile_personalization",
        lambda **_kwargs: SimpleNamespace(
            generation_prompt="",
            retrieval_hints=[],
            used_memories=[],
        ),
    )
    monkeypatch.setattr(retrieval, "query_single_document", fail_query_single_document)
    monkeypatch.setattr(
        retrieval,
        "collect_direct_context",
        lambda document_id, max_chars=12000: (f"context {document_id}", "full_docs", []),
    )
    monkeypatch.setattr(
        retrieval,
        "fallback_chunks_from_storage",
        lambda document_id, _kwargs: [
            {"chunk_id": f"{document_id}-chunk", "content": f"context {document_id}"}
        ],
    )
    monkeypatch.setattr(retrieval, "map_retrieved_chunks_to_sources", fake_map_sources)
    monkeypatch.setattr(retrieval, "synthesize_multi_document_answer", fake_synthesize)
    monkeypatch.setattr(retrieval, "select_related_images", lambda *_args: [])
    monkeypatch.setattr(retrieval, "public_images_for_documents", lambda *_args: [])
    monkeypatch.setattr(
        retrieval,
        "validate_answer_images",
        lambda answer, related_images, _allowed_images: (answer, related_images, []),
    )

    response = asyncio.run(
        retrieval.chat(
            ChatRequest(
                question="请给出这两个 PDF 的详细总结",
                document_ids=["a.pdf", "b.pdf"],
                level="undergraduate",
                mode="hybrid",
            )
        )
    )

    assert response.answer == "summary answer"
    assert [document.document_id for document in response.documents_used] == [
        "a.pdf",
        "b.pdf",
    ]
    assert [source.document_id for source in response.sources] == ["a.pdf", "b.pdf"]


def test_multi_document_query_concurrency_preserves_result_order(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    contexts = [
        make_context("a.pdf", tmp_path),
        make_context("b.pdf", tmp_path),
        make_context("c.pdf", tmp_path),
    ]

    async def fake_query_single_document(
        _request: ChatRequest,
        context: DocumentChatContext,
        *_args: Any,
        **_kwargs: Any,
    ) -> DocumentAnswer:
        if context.document_id == "a.pdf":
            await asyncio.sleep(0.02)
        if context.document_id == "b.pdf":
            await asyncio.sleep(0.01)
            raise HTTPException(status_code=409, detail={"error": "not_ready"})
        return make_answer(context.document_id, tmp_path)

    monkeypatch.setattr(retrieval, "query_single_document", fake_query_single_document)
    monkeypatch.setattr(retrieval, "multi_document_query_concurrency", lambda: 2)

    answers, failures = asyncio.run(
        retrieval.query_multi_document_answers(
            ChatRequest(
                question="比较两个文档",
                document_ids=["a.pdf", "b.pdf", "c.pdf"],
                level="undergraduate",
                mode="hybrid",
            ),
            contexts,
            "undergraduate",
            "本科",
            "",
            [],
        )
    )

    assert [answer.document_id for answer in answers] == ["a.pdf", "c.pdf"]
    assert [failure.document_id for failure in failures] == ["b.pdf"]
