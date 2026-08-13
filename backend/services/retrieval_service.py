from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from backend.config import as_bool, as_int
from backend.core.ids import safe_document_id
from backend.core.models import DocumentAnswer, DocumentChatContext
from backend.core.paths import document_path, document_storage_dir
from backend.core.runtime import document_lock, is_document_processing
from backend.core.text import response_content_to_text
from backend.prompts.query_builder import build_chat_query, prompt_template_name
from backend.prompts.templates import (
    ANSWER_LEVELS,
    build_document_prompt,
    build_task_system_prompt,
    normalize_answer_level,
)
from backend.rag.factory import get_rag
from backend.rag.readiness import document_storage_readiness
from backend.rag.rerank import build_rerank_query_config, current_rerank_runtime_status
from backend.schemas import (
    ChatDocumentUsed,
    ChatPartialFailure,
    ChatRequest,
    ChatResponse,
    MemoryExtractRequest,
    SourceItem,
)
from backend.services.personalization_service import compile_personalization
from backend.services.conversation_service import (
    append_message,
    ensure_conversation_for_chat,
    update_message_status,
)
from backend.services.citation_service import (
    collect_sources_for_query,
    fast_text_chunks_from_storage,
    fallback_chunks_from_storage,
    map_retrieved_chunks_to_sources,
)
from backend.services.document_service import (
    document_status,
    knowledge_base_not_ready_response,
)
from backend.services.image_service import (
    clear_vlm_image_tracking,
    configure_vlm_image_registry,
    current_vlm_image_paths,
    public_images_for_documents,
    select_related_images,
    validate_answer_images,
)
from backend.services.memory_service import extract_memories
from backend.services.synthesis_service import (
    answer_needs_fallback,
    collect_direct_context,
    direct_context_fallback,
    synthesize_fast_text_summary_answer,
    synthesize_multi_document_answer,
)
from backend.services.source_utils import citation_metrics, dedupe_sources

logger = logging.getLogger("api_server")

SUMMARY_REQUEST_TERMS = (
    "总结",
    "概述",
    "梳理",
    "归纳",
    "summary",
    "summarize",
    "overview",
)


def broad_query_kwargs(for_synthesis: bool) -> dict[str, int]:
    if for_synthesis:
        default_top_k = 80
        default_chunk_top_k = 28
        default_max_total_tokens = 32000
    else:
        default_top_k = 70
        default_chunk_top_k = 24
        default_max_total_tokens = 28000
    return {
        "top_k": as_int(os.getenv("CHAT_RETRIEVAL_TOP_K"), default_top_k),
        "chunk_top_k": as_int(
            os.getenv("CHAT_RETRIEVAL_CHUNK_TOP_K"),
            default_chunk_top_k,
        ),
        "max_entity_tokens": as_int(
            os.getenv("CHAT_RETRIEVAL_MAX_ENTITY_TOKENS"),
            9000,
        ),
        "max_relation_tokens": as_int(
            os.getenv("CHAT_RETRIEVAL_MAX_RELATION_TOKENS"),
            9000,
        ),
        "max_total_tokens": as_int(
            os.getenv("CHAT_RETRIEVAL_MAX_TOTAL_TOKENS"),
            default_max_total_tokens,
        ),
    }


def fast_text_query_kwargs(for_synthesis: bool) -> dict[str, int]:
    if for_synthesis:
        default_top_k = 36
        default_chunk_top_k = 12
        default_max_total_tokens = 16000
    else:
        default_top_k = 28
        default_chunk_top_k = 8
        default_max_total_tokens = 12000
    return {
        "top_k": as_int(os.getenv("CHAT_FAST_TEXT_RETRIEVAL_TOP_K"), default_top_k),
        "chunk_top_k": as_int(
            os.getenv("CHAT_FAST_TEXT_RETRIEVAL_CHUNK_TOP_K"),
            default_chunk_top_k,
        ),
        "max_entity_tokens": as_int(
            os.getenv("CHAT_FAST_TEXT_RETRIEVAL_MAX_ENTITY_TOKENS"),
            5000,
        ),
        "max_relation_tokens": as_int(
            os.getenv("CHAT_FAST_TEXT_RETRIEVAL_MAX_RELATION_TOKENS"),
            5000,
        ),
        "max_total_tokens": as_int(
            os.getenv("CHAT_FAST_TEXT_RETRIEVAL_MAX_TOTAL_TOKENS"),
            default_max_total_tokens,
        ),
    }


def query_kwargs_for_chat(request: ChatRequest, for_synthesis: bool) -> dict[str, int]:
    if request.vlm_enhanced is False:
        return fast_text_query_kwargs(for_synthesis)
    return broad_query_kwargs(for_synthesis)


def is_fast_text_request(request: ChatRequest) -> bool:
    return request.vlm_enhanced is False


def effective_query_mode_for_chat(request: ChatRequest) -> str:
    if is_fast_text_request(request):
        return (os.getenv("CHAT_FAST_TEXT_QUERY_MODE") or "naive").strip() or "naive"
    return request.mode


def fast_text_rerank_enabled() -> bool:
    return as_bool(os.getenv("CHAT_FAST_TEXT_ENABLE_RERANK"), False)


def effective_rerank_for_chat(request: ChatRequest, rerank_available: bool) -> bool:
    if is_fast_text_request(request):
        return rerank_available and fast_text_rerank_enabled()
    return rerank_available


def multi_document_query_concurrency() -> int:
    configured = as_int(os.getenv("CHAT_MULTI_DOCUMENT_CONCURRENCY"), 2)
    return max(1, min(configured, 3))


def vlm_query_timeout_seconds() -> float | None:
    raw_timeout = os.getenv("CHAT_VLM_QUERY_TIMEOUT_SECONDS")
    if raw_timeout in (None, ""):
        return 45.0
    try:
        timeout = float(raw_timeout)
    except ValueError:
        logger.warning(
            "Invalid CHAT_VLM_QUERY_TIMEOUT_SECONDS=%r; using default 45s.",
            raw_timeout,
        )
        return 45.0
    return timeout if timeout > 0 else None


def is_summary_request(question: str) -> bool:
    normalized = question.strip().lower()
    return any(term in normalized for term in SUMMARY_REQUEST_TERMS)


def log_citation_metrics(scope: str, sources: list[SourceItem]) -> None:
    metrics = citation_metrics(sources)
    logger.info(
        "Chat citation metrics scope=%s sources_total=%s sources_with_page=%s page_hit_rate=%.4f fallback_ratio=%.4f citation_mode_counts=%s match_method_counts=%s documents_covered=%s",
        scope,
        metrics["sources_total"],
        metrics["sources_with_page"],
        metrics["page_hit_rate"],
        metrics["fallback_ratio"],
        metrics["citation_mode_counts"],
        metrics["match_method_counts"],
        metrics["documents_covered"],
    )

def normalize_chat_document_ids(request: ChatRequest) -> list[str]:
    raw_document_ids = request.document_ids or []
    has_document_ids = any(str(document_id).strip() for document_id in raw_document_ids)
    if not has_document_ids:
        raw_document_ids = [request.document_id] if request.document_id else []

    document_ids: list[str] = []
    seen: set[str] = set()
    for raw_document_id in raw_document_ids:
        if not raw_document_id:
            continue
        document_id = safe_document_id(raw_document_id)
        if not document_id or document_id in seen:
            continue
        seen.add(document_id)
        document_ids.append(document_id)
    return document_ids


def chat_mode_from_request(request: ChatRequest) -> str:
    return "fast_text" if request.vlm_enhanced is False else "multimodal"


def mark_user_message_failed(
    message_id: str | None,
    error: str,
) -> None:
    if not message_id:
        return
    try:
        update_message_status(message_id, status="failed", error=error)
    except Exception as exc:
        logger.warning("Failed to mark conversation message failed: %s", exc)


def refresh_memory_candidates(conversation_id: str | None) -> None:
    if not conversation_id:
        return
    try:
        extract_memories(MemoryExtractRequest(conversation_id=conversation_id, limit=5))
    except Exception as exc:
        logger.warning("Memory candidate extraction failed: %s", exc)


async def collect_sources_for_chat_answer(
    rag: Any,
    document_id: str,
    document_name: str,
    query: str,
    query_kwargs: dict[str, Any],
    request: ChatRequest,
) -> list[SourceItem]:
    if request.vlm_enhanced is False and not as_bool(
        os.getenv("CHAT_FAST_TEXT_LIVE_CITATIONS"),
        False,
    ):
        chunks = fast_text_chunks_from_storage(document_id, query, query_kwargs)
        sources = map_retrieved_chunks_to_sources(document_id, document_name, chunks)
        logger.info(
            "Chat fast-text citation sources collected from local storage document_id=%s count=%s live_citations=false",
            document_id,
            len(sources),
        )
        return sources

    return await collect_sources_for_query(
        rag=rag,
        document_id=document_id,
        document_name=document_name,
        query=query,
        query_kwargs=query_kwargs,
    )


def validate_chat_document_contexts(
    document_ids: list[str],
) -> list[DocumentChatContext] | JSONResponse:
    not_ready_documents: list[dict[str, Any]] = []
    contexts: list[DocumentChatContext] = []

    for document_id in document_ids:
        pdf_path = document_path(document_id)
        status = document_status(pdf_path)
        storage_dir = document_storage_dir(document_id)
        readiness = document_storage_readiness(document_id)
        logger.info(
            "Chat document validation document_id=%s storage_dir=%s document_status=%s ready_for_chat=%s",
            document_id,
            storage_dir,
            status,
            readiness.ready,
        )
        context = DocumentChatContext(
            document_id=document_id,
            name=pdf_path.name,
            status=status,
            storage_dir=storage_dir,
            readiness=readiness,
        )
        contexts.append(context)
        if is_document_processing(document_id):
            not_ready_documents.append(
                {
                    "document_id": document_id,
                    "name": pdf_path.name,
                    "status": "indexing",
                    "readiness_warnings": ["document is processing"],
                    "storage_dir": str(storage_dir),
                }
            )
            continue
        if status != "ready_for_chat" or not readiness.ready:
            not_ready_documents.append(
                {
                    "document_id": document_id,
                    "name": pdf_path.name,
                    "status": status if status != "ready_for_chat" else readiness.status,
                    "readiness_warnings": readiness.warnings,
                    "storage_dir": str(storage_dir),
                }
            )

    if not_ready_documents:
        names = "、".join(str(item["name"]) for item in not_ready_documents)
        return JSONResponse(
            status_code=409,
            content={
                "error": "knowledge_base_not_ready",
                "message": f"以下文档未就绪：{names}",
                "not_ready_documents": not_ready_documents,
            },
        )

    return contexts


async def query_single_document(
    request: ChatRequest,
    context: DocumentChatContext,
    level_key: str,
    level_prompt: str,
    profile_prompt: str,
    retrieval_hints: list[str],
    for_synthesis: bool = False,
) -> DocumentAnswer:
    timings: dict[str, float] = {}
    document_id = context.document_id
    storage_dir = context.storage_dir

    lock = await document_lock(document_id)
    async with lock:
        validate_started = time.perf_counter()
        if is_document_processing(document_id):
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "document_processing",
                    "message": "当前文档正在索引，请稍后再提问。",
                    "document_id": document_id,
                },
            )

        pdf_path = document_path(document_id)
        status = document_status(pdf_path)
        readiness = document_storage_readiness(document_id)
        logger.info(
            "Chat retrieval scope=document document_id=%s storage_dir=%s document_status=%s ready_for_chat=%s",
            document_id,
            storage_dir,
            status,
            readiness.ready,
        )
        if status != "ready_for_chat" or not readiness.ready:
            logger.warning(
                "Document-scoped storage is not ready for chat document_id=%s storage_dir=%s document_status=%s warnings=%s",
                document_id,
                storage_dir,
                status,
                "; ".join(readiness.warnings),
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "knowledge_base_not_ready",
                    "message": "请先解析/更新知识库。",
                    "document_id": document_id,
                    "document_status": status
                    if status != "ready_for_chat"
                    else readiness.status,
                    "readiness_warnings": readiness.warnings,
                },
            )
        timings["validate_document"] = time.perf_counter() - validate_started

        load_started = time.perf_counter()
        rag = await get_rag(document_id)
        configure_vlm_image_registry(rag, [document_id])

        init_result = await rag._ensure_lightrag_initialized()
        if not init_result.get("success") or rag.lightrag is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "knowledge_base_not_ready",
                    "message": "请先解析/更新知识库。",
                    "document_id": document_id,
                    "document_status": "partial_success",
                },
            )
        timings["load_rag"] = time.perf_counter() - load_started

        rewrite_started = time.perf_counter()
        user_question = request.question.strip()
        query = build_chat_query(
            user_question,
            document_id,
            for_synthesis,
            retrieval_hints,
        )
        document_system_prompt = build_document_prompt(
            user_question,
            document_id,
            level_key,
            level_prompt,
            profile_prompt,
            for_synthesis,
        )
        prompt_template_used = prompt_template_name(1)
        timings["query_rewrite"] = time.perf_counter() - rewrite_started

        rerank_query_config = build_rerank_query_config(rag)
        effective_enable_rerank = effective_rerank_for_chat(
            request,
            bool(rerank_query_config["enabled"]),
        )
        effective_query_mode = effective_query_mode_for_chat(request)
        requested_vlm_enhanced = request.vlm_enhanced
        logger.info(
            "Chat retrieval config: document_id=%s mode=%s effective_query_mode=%s rerank=%s rerank_requested=%s rerank_enabled=%s effective_enable_rerank=%s rerank_model=%s rerank_provider=%s rerank_model_func=%s configured_rerank_top_n=%s rerank_reason=%s vlm_enhanced=%s routing=direct for_synthesis=%s level=%s prompt_template=%s fast_text_strategy=%s",
            document_id,
            request.mode,
            effective_query_mode,
            rerank_query_config["label"],
            rerank_query_config["requested"],
            rerank_query_config["enabled"],
            effective_enable_rerank,
            rerank_query_config["model"] or "none",
            rerank_query_config["provider"] or "none",
            "not None" if rerank_query_config["model_func_available"] else "None",
            rerank_query_config["top_n"] or "QueryParam.chunk_top_k",
            rerank_query_config["reason"] or "none",
            requested_vlm_enhanced,
            for_synthesis,
            level_key,
            prompt_template_used,
            "lightweight_query" if is_fast_text_request(request) else "default",
        )
        fallback_used = False
        answer_source_path = "text" if requested_vlm_enhanced is False else "vlm"
        answer = ""
        sources: list[SourceItem] = []
        query_started = time.perf_counter()
        try:
            primary_query_started = time.perf_counter()
            clear_vlm_image_tracking(rag)
            system_prompt = (
                build_task_system_prompt(profile_prompt)
                + "\n\n"
                + document_system_prompt
            )
            query_kwargs: dict[str, Any] = {
                "mode": effective_query_mode,
                "system_prompt": system_prompt,
                "enable_rerank": effective_enable_rerank,
                **query_kwargs_for_chat(request, for_synthesis),
            }
            if requested_vlm_enhanced != "auto":
                query_kwargs["vlm_enhanced"] = requested_vlm_enhanced
            logger.info(
                "Chat query params: document_id=%s mode=%s enable_rerank=%s rerank_model_func_available=%s top_k=%s chunk_top_k=%s max_total_tokens=%s vlm_enhanced=%s",
                document_id,
                query_kwargs["mode"],
                query_kwargs["enable_rerank"],
                rerank_query_config["model_func_available"],
                query_kwargs.get("top_k", "QueryParam.default"),
                query_kwargs.get("chunk_top_k", "QueryParam.default"),
                query_kwargs.get("max_total_tokens", "QueryParam.default"),
                query_kwargs.get("vlm_enhanced", "auto"),
            )
            try:
                primary_query = rag.aquery(
                    query,
                    **query_kwargs,
                )
                if requested_vlm_enhanced is not False:
                    timeout = vlm_query_timeout_seconds()
                    if timeout is not None:
                        primary_query = asyncio.wait_for(
                            primary_query,
                            timeout=timeout,
                        )
                answer = response_content_to_text(await primary_query)
            except Exception as error:
                primary_query_elapsed = time.perf_counter() - primary_query_started
                timings["vlm_enhanced_query"] = (
                    primary_query_elapsed
                    if requested_vlm_enhanced is not False
                    else 0.0
                )
                if requested_vlm_enhanced is False:
                    raise
                if isinstance(error, HTTPException):
                    raise
                if (
                    isinstance(error, ValueError)
                    and "No LightRAG instance available" in str(error)
                ):
                    raise
                fallback_used = True
                answer_source_path = "text_fallback"
                logger.warning(
                    "VLM enhanced chat query failed; retrying with vlm_enhanced=False. document_id=%s error=%s",
                    document_id,
                    error,
                )
                fallback_started = time.perf_counter()
                try:
                    answer = response_content_to_text(
                        await rag.aquery(
                            query,
                            mode=effective_query_mode,
                            system_prompt=system_prompt,
                            vlm_enhanced=False,
                            enable_rerank=effective_enable_rerank,
                            **query_kwargs_for_chat(request, for_synthesis),
                        )
                    )
                except TypeError as fallback_error:
                    logger.warning(
                        "vlm_enhanced=False fallback is unsupported document_id=%s error=%s",
                        document_id,
                        fallback_error,
                    )
                    answer = ""
                timings["fallback_query"] = time.perf_counter() - fallback_started
            else:
                primary_query_elapsed = time.perf_counter() - primary_query_started
                timings["vlm_enhanced_query"] = (
                    primary_query_elapsed
                    if requested_vlm_enhanced is not False
                    else 0.0
                )
            if (
                answer_needs_fallback(answer)
                and requested_vlm_enhanced is not False
                and not fallback_used
            ):
                fallback_used = True
                answer_source_path = "text_fallback"
                logger.warning(
                    "VLM enhanced chat query returned empty or context-empty answer; retrying with vlm_enhanced=False. document_id=%s",
                    document_id,
                )
                fallback_started = time.perf_counter()
                try:
                    answer = response_content_to_text(
                        await rag.aquery(
                            query,
                            mode=effective_query_mode,
                            system_prompt=system_prompt,
                            vlm_enhanced=False,
                            enable_rerank=effective_enable_rerank,
                            **query_kwargs_for_chat(request, for_synthesis),
                        )
                    )
                except TypeError as error:
                    logger.warning(
                        "vlm_enhanced=False fallback is unsupported document_id=%s error=%s",
                        document_id,
                        error,
                    )
                    answer = ""
                timings["fallback_query"] = time.perf_counter() - fallback_started
            if answer_needs_fallback(answer):
                fallback_used = True
                logger.warning(
                    "Text chat query returned empty/context-empty answer; trying direct context fallback. document_id=%s routing=direct answer_source_path=%s",
                    document_id,
                    answer_source_path,
                )
                fallback_started = time.perf_counter()
                answer, answer_source_path = await direct_context_fallback(
                    rag,
                    user_question,
                    document_id,
                    level_key,
                    level_prompt,
                    profile_prompt,
                )
                timings["fallback_query"] = (
                    timings.get("fallback_query", 0.0)
                    + time.perf_counter()
                    - fallback_started
                )
        except ValueError as error:
            if "No LightRAG instance available" in str(error):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "knowledge_base_not_ready",
                        "message": "请先解析/更新知识库。",
                        "document_id": document_id,
                        "document_status": "partial_success",
                    },
                ) from error
            raise
        timings["aquery_total"] = time.perf_counter() - query_started
        post_query_rerank_status = current_rerank_runtime_status()
        logger.info(
            "Chat rerank verification: document_id=%s enable_rerank_passed=%s rerank_model_func_available=%s rerank_top_n=%s rerank_last_error=%s",
            document_id,
            effective_enable_rerank,
            rerank_query_config["model_func_available"],
            rerank_query_config["top_n"] or "QueryParam.chunk_top_k",
            post_query_rerank_status.get("last_error") or "none",
        )

        if answer_needs_fallback(answer):
            logger.warning(
                "Document-scoped retrieval could not produce a usable answer. document_id=%s storage_dir=%s document_status=%s ready_for_chat=%s fallback_used=%s answer_source_path=%s",
                document_id,
                storage_dir,
                status,
                readiness.ready,
                fallback_used,
                answer_source_path,
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "knowledge_base_not_ready",
                    "message": "请先解析/更新知识库。",
                    "document_id": document_id,
                    "document_status": "partial_success",
                },
            )

        citation_started = time.perf_counter()
        source_query = user_question if answer_source_path.startswith("direct_context_fallback") else query
        source_query_kwargs: dict[str, Any] = {
            "mode": effective_query_mode,
            "enable_rerank": effective_enable_rerank
            if not answer_source_path.startswith("direct_context_fallback")
            else False,
            **query_kwargs_for_chat(request, for_synthesis),
        }
        sources = await collect_sources_for_chat_answer(
            rag=rag,
            document_id=document_id,
            document_name=context.name,
            query=source_query,
            query_kwargs=source_query_kwargs,
            request=request,
        )
        timings["citation_sources"] = time.perf_counter() - citation_started
        logger.info(
            "Chat citation sources collected document_id=%s count=%s elapsed=%.3fs answer_source_path=%s",
            document_id,
            len(sources),
            timings["citation_sources"],
            answer_source_path,
        )
        log_citation_metrics(f"document:{document_id}", sources)

    return DocumentAnswer(
        document_id=document_id,
        name=context.name,
        status=status,
        storage_dir=storage_dir,
        answer=answer,
        sources=sources,
        vlm_image_paths=current_vlm_image_paths(rag),
        fallback_used=fallback_used,
        answer_source_path=answer_source_path,
        prompt_template_used=prompt_template_used,
        timings=timings,
    )


def build_direct_summary_document_answers(
    contexts: list[DocumentChatContext],
    *,
    max_context_chars: int | None = None,
    answer_source_prefix: str = "direct_summary",
) -> tuple[list[DocumentAnswer], list[ChatPartialFailure]]:
    document_answers: list[DocumentAnswer] = []
    partial_failures: list[ChatPartialFailure] = []
    max_context_chars = max_context_chars or max(
        1000,
        as_int(os.getenv("CHAT_SUMMARY_DIRECT_CONTEXT_CHARS"), 12000),
    )
    source_limit = max(1, as_int(os.getenv("CHAT_SUMMARY_SOURCES_LIMIT"), 8))

    for context in contexts:
        started = time.perf_counter()
        direct_context, source_path, warnings = collect_direct_context(
            context.document_id,
            max_chars=max_context_chars,
        )
        if warnings:
            logger.warning(
                "Chat summary direct context warnings document_id=%s warnings=%s",
                context.document_id,
                "; ".join(warnings[:8]),
            )
        if not direct_context.strip():
            partial_failures.append(
                ChatPartialFailure(
                    document_id=context.document_id,
                    name=context.name,
                    error="direct_summary_context_empty",
                )
            )
            continue

        chunks = fallback_chunks_from_storage(
            context.document_id,
            {"chunk_top_k": source_limit},
        )
        sources = map_retrieved_chunks_to_sources(
            context.document_id,
            context.name,
            chunks,
            limit=source_limit,
        )
        logger.info(
            "Chat summary direct route document_id=%s source_path=%s context_chars=%s sources=%s elapsed=%.3fs",
            context.document_id,
            source_path,
            len(direct_context),
            len(sources),
            time.perf_counter() - started,
        )
        log_citation_metrics(f"summary_direct:{context.document_id}", sources)
        document_answers.append(
            DocumentAnswer(
                document_id=context.document_id,
                name=context.name,
                status=context.status,
                storage_dir=context.storage_dir,
                answer=direct_context[:max_context_chars],
                sources=sources,
                vlm_image_paths=[],
                fallback_used=False,
                answer_source_path=f"{answer_source_prefix}:{source_path}",
                prompt_template_used=prompt_template_name(1),
                timings={"direct_context": time.perf_counter() - started},
            )
        )

    return document_answers, partial_failures


def build_fast_text_summary_document_answers(
    contexts: list[DocumentChatContext],
) -> tuple[list[DocumentAnswer], list[ChatPartialFailure]]:
    max_context_chars = max(
        1000,
        as_int(os.getenv("CHAT_FAST_TEXT_SUMMARY_CHARS_PER_DOC"), 4000),
    )
    return build_direct_summary_document_answers(
        contexts,
        max_context_chars=max_context_chars,
        answer_source_prefix="fast_text_summary",
    )


async def query_multi_document_answers(
    request: ChatRequest,
    contexts: list[DocumentChatContext],
    level_key: str,
    level_prompt: str,
    profile_prompt: str,
    retrieval_hints: list[str],
) -> tuple[list[DocumentAnswer], list[ChatPartialFailure]]:
    semaphore = asyncio.Semaphore(multi_document_query_concurrency())

    async def query_context(
        index: int,
        context: DocumentChatContext,
    ) -> tuple[int, DocumentAnswer | None, ChatPartialFailure | None]:
        document_query_started = time.perf_counter()
        try:
            async with semaphore:
                document_answer = await query_single_document(
                    request,
                    context,
                    level_key,
                    level_prompt,
                    profile_prompt,
                    retrieval_hints,
                    True,
                )
            logger.info(
                "Chat multi-document per-document query document_id=%s storage_dir=%s elapsed=%.3fs answer_source_path=%s",
                context.document_id,
                context.storage_dir,
                time.perf_counter() - document_query_started,
                document_answer.answer_source_path,
            )
            return index, document_answer, None
        except HTTPException as exc:
            logger.warning(
                "Chat multi-document partial failure document_id=%s storage_dir=%s error=%s",
                context.document_id,
                context.storage_dir,
                exc.detail,
            )
            return (
                index,
                None,
                ChatPartialFailure(
                    document_id=context.document_id,
                    name=context.name,
                    error=str(exc.detail),
                ),
            )

    results = await asyncio.gather(
        *(query_context(index, context) for index, context in enumerate(contexts))
    )
    ordered_results = sorted(results, key=lambda item: item[0])
    document_answers = [
        document_answer
        for _index, document_answer, _failure in ordered_results
        if document_answer is not None
    ]
    partial_failures = [
        failure
        for _index, _document_answer, failure in ordered_results
        if failure is not None
    ]
    return document_answers, partial_failures


async def chat(request: ChatRequest) -> ChatResponse | JSONResponse:
    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    level_key = normalize_answer_level(request.level)
    level_prompt = ANSWER_LEVELS[level_key]
    validate_started = time.perf_counter()
    document_ids = normalize_chat_document_ids(request)
    chat_mode = chat_mode_from_request(request)
    summary_requested = is_summary_request(request.question)
    prompt_template_used = prompt_template_name(len(document_ids))
    logger.info(
        "Chat request document_ids=%s legacy_document_id=%s chat_mode=%s is_summary_request=%s effective_query_mode=%s fast_text_strategy=%s",
        document_ids,
        request.document_id,
        chat_mode,
        summary_requested,
        effective_query_mode_for_chat(request) if document_ids else "none",
        "compact_summary" if is_fast_text_request(request) and summary_requested else (
            "lightweight_query" if is_fast_text_request(request) else "default"
        ),
    )
    conversation_id: str | None = None
    user_message_id: str | None = None
    assistant_message_id: str | None = None
    if not document_ids:
        raise HTTPException(status_code=400, detail="请先选择文档")
    if len(document_ids) > 3:
        raise HTTPException(status_code=400, detail="最多选择 3 个文档")

    if request.conversation_id:
        conversation = ensure_conversation_for_chat(
            conversation_id=request.conversation_id,
            question=request.question,
            document_ids=document_ids,
            level=level_key,
            mode=request.mode,
            chat_mode=chat_mode,
        )
        conversation_id = conversation.id
        user_message = append_message(
            conversation_id=conversation.id,
            role="user",
            content=request.question,
            document_ids=document_ids,
            message_id=request.client_user_message_id,
            level=level_key,
            mode=request.mode,
            chat_mode=chat_mode,
            status="sent",
        )
        user_message_id = user_message.id

    personalization = compile_personalization(
        question=request.question,
        document_ids=document_ids,
        level=level_key,
        conversation_id=conversation_id,
        use_profile=request.use_profile,
        use_memory=request.use_memory,
    )
    profile_prompt = personalization.generation_prompt
    retrieval_hints = personalization.retrieval_hints
    logger.info(
        "Chat direct routing original_question=%r level=%s document_ids=%s prompt_template=%s profile_prompt_chars=%s use_profile=%s use_memory=%s used_memories=%s retrieval_hints=%s conversation_id=%s",
        request.question,
        level_key,
        document_ids,
        prompt_template_used,
        len(profile_prompt),
        request.use_profile,
        request.use_memory,
        [memory.id for memory in personalization.used_memories],
        retrieval_hints,
        conversation_id,
    )

    contexts_or_response = validate_chat_document_contexts(document_ids)
    if isinstance(contexts_or_response, JSONResponse):
        mark_user_message_failed(user_message_id, "knowledge_base_not_ready")
        return contexts_or_response
    contexts = contexts_or_response
    timings["validate_document"] = time.perf_counter() - validate_started

    logger.info(
        "Chat retrieval scope=%s document_ids=%s storage_dirs=%s",
        "multi_document" if len(contexts) > 1 else "document",
        document_ids,
        [str(context.storage_dir) for context in contexts],
    )

    try:
        if is_fast_text_request(request) and summary_requested:
            logger.info(
                "Chat fast-text summary routing enabled document_ids=%s question=%r",
                document_ids,
                request.question,
            )
            document_answers, partial_failures = build_fast_text_summary_document_answers(
                contexts
            )
            synthesis_source_path = "fast_text_summary_synthesis"
        elif len(contexts) == 1:
            document_answer = await query_single_document(
                request,
                contexts[0],
                level_key,
                level_prompt,
                profile_prompt,
                retrieval_hints,
                False,
            )
            package_started = time.perf_counter()
            related_images = select_related_images(
                [document_answer.document_id],
                request.question,
                document_answer.vlm_image_paths,
            )
            allowed_images = public_images_for_documents([document_answer.document_id])
            answer, related_images, inline_image_refs = validate_answer_images(
                document_answer.answer,
                related_images,
                allowed_images,
            )
            response = ChatResponse(
                answer=answer,
                sources=document_answer.sources,
                related_images=related_images,
                inline_image_refs=inline_image_refs,
                documents_used=[
                    ChatDocumentUsed(
                        document_id=document_answer.document_id,
                        name=document_answer.name,
                        status=document_answer.status,
                    )
                ],
            )
            timings["response_packaging"] = time.perf_counter() - package_started
            timings["total"] = time.perf_counter() - total_started
            logger.info(
                "Chat answer ready original_question=%r routing=direct level=%s document_ids=%s prompt_template=%s document_id=%s storage_dir=%s document_status=%s ready_for_chat=%s retrieval_scope=document fallback_used=%s answer_source_path=%s",
                request.question,
                level_key,
                document_ids,
                document_answer.prompt_template_used,
                document_answer.document_id,
                document_answer.storage_dir,
                document_answer.status,
                True,
                document_answer.fallback_used,
                document_answer.answer_source_path,
            )
            logger.info(
                "Chat timing document_id=%s validate_document=%.3fs load_rag=%.3fs query_rewrite=%.3fs vlm_enhanced_query=%.3fs fallback_query=%.3fs aquery_total=%.3fs response_packaging=%.3fs total=%.3fs",
                document_answer.document_id,
                timings.get("validate_document", 0.0),
                document_answer.timings.get("load_rag", 0.0),
                document_answer.timings.get("query_rewrite", 0.0),
                document_answer.timings.get("vlm_enhanced_query", 0.0),
                document_answer.timings.get("fallback_query", 0.0),
                document_answer.timings.get("aquery_total", 0.0),
                timings.get("response_packaging", 0.0),
                timings.get("total", 0.0),
            )
            if conversation_id:
                try:
                    assistant_message = append_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=response.answer,
                        document_ids=document_ids,
                        level=level_key,
                        mode=request.mode,
                        chat_mode=chat_mode,
                        sources=response.sources,
                        related_images=response.related_images,
                        inline_image_refs=response.inline_image_refs,
                        status="sent",
                    )
                    assistant_message_id = assistant_message.id
                except HTTPException as exc:
                    if exc.status_code != 404:
                        raise
                    logger.warning(
                        "Skipping assistant message persistence because conversation disappeared conversation_id=%s error=%s",
                        conversation_id,
                        exc.detail,
                    )
                response.conversation_id = conversation_id
                response.user_message_id = user_message_id
                response.assistant_message_id = assistant_message_id
                if assistant_message_id:
                    refresh_memory_candidates(conversation_id)
            return response

        elif summary_requested:
            logger.info(
                "Chat summary direct routing enabled document_ids=%s question=%r",
                document_ids,
                request.question,
            )
            document_answers, partial_failures = build_direct_summary_document_answers(
                contexts
            )
            synthesis_source_path = "summary_direct_synthesis"
        else:
            document_answers, partial_failures = await query_multi_document_answers(
                request,
                contexts,
                level_key,
                level_prompt,
                profile_prompt,
                retrieval_hints,
            )
            synthesis_source_path = "multi_document_synthesis"

        if not document_answers:
            mark_user_message_failed(user_message_id, "knowledge_base_not_ready")
            return knowledge_base_not_ready_response(
                message="所选文档暂时无法生成回答，请重新解析/更新知识库。",
                document_status="partial_success",
            )

        synthesis_started = time.perf_counter()
        if is_fast_text_request(request) and summary_requested:
            answer = await synthesize_fast_text_summary_answer(
                request.question,
                document_answers,
                level_key,
                level_prompt,
                profile_prompt,
            )
        else:
            answer = await synthesize_multi_document_answer(
                request.question,
                document_answers,
                level_key,
                level_prompt,
                profile_prompt,
            )
        timings["synthesis"] = time.perf_counter() - synthesis_started
        if answer_needs_fallback(answer):
            mark_user_message_failed(user_message_id, "empty_multi_document_answer")
            return knowledge_base_not_ready_response(
                message="多文档综合回答为空，请稍后重试。",
                document_status="partial_success",
            )

        package_started = time.perf_counter()
        answer_document_ids = [
            document_answer.document_id for document_answer in document_answers
        ]
        vlm_image_paths = [
            image_path
            for document_answer in document_answers
            for image_path in document_answer.vlm_image_paths
        ]
        related_images = select_related_images(
            answer_document_ids,
            request.question,
            vlm_image_paths,
        )
        allowed_images = public_images_for_documents(answer_document_ids)
        answer, related_images, inline_image_refs = validate_answer_images(
            answer,
            related_images,
            allowed_images,
        )
        sources = dedupe_sources(
            [
                source
                for document_answer in document_answers
                for source in document_answer.sources
            ]
        )
        log_citation_metrics("multi_document", sources)
        response = ChatResponse(
            answer=answer,
            sources=sources,
            related_images=related_images,
            inline_image_refs=inline_image_refs,
            documents_used=[
                ChatDocumentUsed(
                    document_id=document_answer.document_id,
                    name=document_answer.name,
                    status=document_answer.status,
                )
                for document_answer in document_answers
            ],
            partial_failures=partial_failures,
        )
        timings["response_packaging"] = time.perf_counter() - package_started
        timings["total"] = time.perf_counter() - total_started
        logger.info(
            "Chat answer ready original_question=%r routing=direct level=%s document_ids=%s prompt_template=%s retrieval_scope=multi_document storage_dirs=%s fallback_used=%s answer_source_path=%s partial_failures=%s",
            request.question,
            level_key,
            [document_answer.document_id for document_answer in document_answers],
            prompt_template_used,
            [str(document_answer.storage_dir) for document_answer in document_answers],
            any(document_answer.fallback_used for document_answer in document_answers),
            synthesis_source_path,
            len(partial_failures),
        )
        logger.info(
            "Chat timing retrieval_scope=multi_document document_ids=%s validate_document=%.3fs synthesis=%.3fs response_packaging=%.3fs total=%.3fs",
            [document_answer.document_id for document_answer in document_answers],
            timings.get("validate_document", 0.0),
            timings.get("synthesis", 0.0),
            timings.get("response_packaging", 0.0),
            timings.get("total", 0.0),
        )
        if conversation_id:
            try:
                assistant_message = append_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=response.answer,
                    document_ids=document_ids,
                    level=level_key,
                    mode=request.mode,
                    chat_mode=chat_mode,
                    sources=response.sources,
                    related_images=response.related_images,
                    inline_image_refs=response.inline_image_refs,
                    status="sent",
                )
                assistant_message_id = assistant_message.id
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
                logger.warning(
                    "Skipping assistant message persistence because conversation disappeared conversation_id=%s error=%s",
                    conversation_id,
                    exc.detail,
                )
            response.conversation_id = conversation_id
            response.user_message_id = user_message_id
            response.assistant_message_id = assistant_message_id
            if assistant_message_id:
                refresh_memory_candidates(conversation_id)
        return response
    except HTTPException as exc:
        mark_user_message_failed(user_message_id, str(exc.detail))
        if exc.status_code == 409 and isinstance(exc.detail, dict):
            return JSONResponse(status_code=409, content=exc.detail)
        raise
    except Exception as exc:
        mark_user_message_failed(user_message_id, str(exc))
        raise
