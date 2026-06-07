from __future__ import annotations

import os

from fastapi import APIRouter

from backend.config import (
    OUTPUT_DIR,
    as_bool,
    formula_processing_enabled,
    multimodal_enabled,
)
from backend.rag.rerank import build_rerank_model_func, current_rerank_runtime_status
from backend.schemas import (
    HealthResponse,
    RAGEmbeddingStatus,
    RAGMultimodalStatus,
    RAGRetrievalStatus,
    RAGStatusResponse,
)

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        ok=True,
        has_api_key=bool(os.getenv("LLM_BINDING_API_KEY") or os.getenv("QWEN_API_KEY")),
        mineru_backend=os.getenv("MINERU_BACKEND", "pipeline"),
        mineru_device=os.getenv("MINERU_DEVICE", "cuda"),
    )


@router.get("/api/rag/status", response_model=RAGStatusResponse)
async def rag_status() -> RAGStatusResponse:
    rerank_status = current_rerank_runtime_status()
    if (
        rerank_status["requested"]
        and rerank_status["model"]
        and not rerank_status["model_loaded"]
        and rerank_status.get("reason")
        not in {"missing_api_key", "unsupported_binding", "provider_init_failed"}
    ):
        build_rerank_model_func()
        rerank_status = current_rerank_runtime_status()

    default_mode = os.getenv("QUERY_MODE", "hybrid")
    embedding_model = os.getenv("EMBEDDING_LOCAL_MODEL", "BAAI/bge-small-zh-v1.5")
    multimodal_is_enabled = multimodal_enabled()
    return RAGStatusResponse(
        parser=os.getenv("PARSER", "mineru"),
        parser_output_dir=str(OUTPUT_DIR),
        retrieval=RAGRetrievalStatus(
            default_mode=default_mode,
            rerank_requested=bool(rerank_status["requested"]),
            rerank_enabled=bool(rerank_status["enabled"]),
            rerank_model=rerank_status.get("model"),
            rerank_provider=rerank_status.get("provider"),
            rerank_model_loaded=bool(rerank_status["model_loaded"]),
            rerank_last_error=rerank_status.get("last_error"),
            reason=rerank_status.get("reason"),
            rerank_top_n=rerank_status.get("top_n"),
        ),
        multimodal=RAGMultimodalStatus(
            enabled=multimodal_is_enabled,
            image_processing=multimodal_is_enabled
            and as_bool(os.getenv("ENABLE_IMAGE_PROCESSING"), True),
            table_processing=multimodal_is_enabled
            and as_bool(os.getenv("ENABLE_TABLE_PROCESSING"), True),
            equation_processing=multimodal_is_enabled
            and as_bool(os.getenv("ENABLE_EQUATION_PROCESSING"), True),
            formula_processing=formula_processing_enabled(),
            vlm_model=os.getenv("QWEN_VL_MODEL") or None,
        ),
        embedding=RAGEmbeddingStatus(model=embedding_model),
        embedding_model=embedding_model,
        vector_store="LightRAG JSON vector stores in rag_storage/vdb_*.json",
        graph_store="LightRAG GraphML graph in rag_storage/graph_chunk_entity_relation.graphml",
        retrieval_mode=default_mode,
        rerank_requested=bool(rerank_status["requested"]),
        rerank_enabled=bool(rerank_status["enabled"]),
        rerank_model=rerank_status.get("model"),
        rerank_binding=rerank_status.get("provider")
        if rerank_status["enabled"]
        else None,
        multimodal_enabled=multimodal_is_enabled,
        citation_status="retrieval evidence sources enabled for chat responses",
        sources_status="chat sources are mapped from retrieval chunks to MinerU content_list pages when available",
    )
