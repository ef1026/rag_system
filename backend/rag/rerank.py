from __future__ import annotations

import logging
import os
import time
from typing import Any

from raganything import RAGAnything

from backend.config import as_bool, as_int

logger = logging.getLogger("api_server")

def rerank_model_name() -> str:
    return (os.getenv("RERANK_MODEL") or "").strip()


def rerank_requested() -> bool:
    return as_bool(os.getenv("ENABLE_RERANK"), False)


RERANK_BINDING_ALIASES = {
    "aliyun": "aliyun",
    "ali": "aliyun",
    "dashscope": "aliyun",
    "jina": "jina",
    "cohere": "cohere",
}


_rerank_runtime_state: dict[str, Any] = {}


def rerank_binding_name() -> str:
    return (os.getenv("RERANK_BINDING") or "aliyun").strip().lower()


def rerank_provider_name() -> str | None:
    return RERANK_BINDING_ALIASES.get(rerank_binding_name())


def rerank_base_url() -> str | None:
    return (
        (os.getenv("RERANK_BASE_URL") or "").strip()
        or (os.getenv("RERANK_BINDING_HOST") or "").strip()
        or None
    )


def rerank_top_n() -> int | None:
    configured_top_n = as_int(os.getenv("RERANK_TOP_N"), 0)
    return configured_top_n if configured_top_n > 0 else None


def rerank_binding_api_key(provider: str | None = None) -> str | None:
    provider = provider or rerank_provider_name()
    if provider == "aliyun":
        return (
            os.getenv("RERANK_BINDING_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("QWEN_API_KEY")
            or None
        )
    if provider == "jina":
        return os.getenv("RERANK_BINDING_API_KEY") or os.getenv("JINA_API_KEY") or None
    if provider == "cohere":
        return (
            os.getenv("RERANK_BINDING_API_KEY")
            or os.getenv("COHERE_API_KEY")
            or None
        )
    return os.getenv("RERANK_BINDING_API_KEY") or None


def update_rerank_runtime_state(**updates: Any) -> None:
    _rerank_runtime_state.update(updates)


def current_rerank_runtime_status() -> dict[str, Any]:
    requested = rerank_requested()
    model = rerank_model_name() or None
    provider = rerank_provider_name() if requested else None
    status: dict[str, Any] = {
        "requested": requested,
        "enabled": False,
        "model": model,
        "provider": provider,
        "model_loaded": False,
        "last_error": None,
        "reason": "disabled_by_config" if not requested else None,
        "top_n": rerank_top_n(),
        "base_url": rerank_base_url(),
    }
    status.update(_rerank_runtime_state)
    status["requested"] = requested
    status["model"] = model
    status["provider"] = provider
    status["top_n"] = rerank_top_n()
    status["base_url"] = rerank_base_url()
    if not requested:
        status.update(
            {
                "enabled": False,
                "model_loaded": False,
                "last_error": None,
                "reason": "disabled_by_config",
            }
        )
    elif not model:
        status.update(
            {
                "enabled": False,
                "model_loaded": False,
                "last_error": "missing_model",
                "reason": "missing_model",
            }
        )
    return status


def rerank_enabled_for_query() -> bool:
    status = current_rerank_runtime_status()
    return bool(
        status["requested"] and status["model"] and status["enabled"] and status["model_loaded"]
    )


def log_rerank_startup_status() -> None:
    if rerank_requested() and not rerank_model_name():
        logger.warning(
            "ENABLE_RERANK=true but RERANK_MODEL is empty; rerank will be disabled for queries."
        )
    elif rerank_requested():
        provider = rerank_provider_name()
        if provider is None:
            logger.warning(
                "Unsupported RERANK_BINDING=%r; rerank will be disabled for queries.",
                rerank_binding_name(),
            )
        elif not rerank_binding_api_key(provider):
            logger.warning(
                "ENABLE_RERANK=true and RERANK_MODEL=%s, but no rerank API key is available for provider=%s; rerank will be disabled for queries.",
                rerank_model_name(),
                provider,
            )
        else:
            logger.info(
                "Rerank requested: provider=%s model=%s base_url=%s top_n=%s",
                provider,
                rerank_model_name(),
                rerank_base_url() or "provider_default",
                rerank_top_n() or "QueryParam.chunk_top_k",
            )
    else:
        logger.info("Rerank disabled.")


def build_rerank_model_func():
    if not rerank_requested():
        update_rerank_runtime_state(
            enabled=False,
            model_loaded=False,
            last_error=None,
            reason="disabled_by_config",
        )
        return None

    model_name = rerank_model_name()
    if not model_name:
        update_rerank_runtime_state(
            enabled=False,
            model_loaded=False,
            last_error="missing_model",
            reason="missing_model",
        )
        logger.warning(
            "ENABLE_RERANK=true but RERANK_MODEL is empty; query enable_rerank will be false."
        )
        return None

    binding = rerank_binding_name()
    provider = rerank_provider_name()
    binding_host = rerank_base_url()
    binding_api_key = rerank_binding_api_key(provider)
    configured_top_n = rerank_top_n()

    if provider is None:
        update_rerank_runtime_state(
            enabled=False,
            model_loaded=False,
            last_error=f"unsupported_binding:{binding}",
            reason="unsupported_binding",
        )
        logger.warning(
            "Unsupported RERANK_BINDING=%r; query enable_rerank will be false.",
            binding,
        )
        return None

    if not binding_api_key:
        update_rerank_runtime_state(
            enabled=False,
            model_loaded=False,
            last_error="missing_api_key",
            reason="missing_api_key",
        )
        logger.warning(
            "Rerank requested but no API key is available for provider=%s model=%s; query enable_rerank will be false.",
            provider,
            model_name,
        )
        return None

    try:
        from lightrag.rerank import ali_rerank, cohere_rerank, jina_rerank
    except Exception as exc:
        update_rerank_runtime_state(
            enabled=False,
            model_loaded=False,
            last_error=str(exc),
            reason="provider_init_failed",
        )
        logger.warning("Failed to load LightRAG rerank functions; rerank disabled: %s", exc)
        return None

    rerank_functions = {
        "aliyun": ali_rerank,
        "jina": jina_rerank,
        "cohere": cohere_rerank,
    }
    selected_rerank_func = rerank_functions.get(provider)
    if selected_rerank_func is None:
        update_rerank_runtime_state(
            enabled=False,
            model_loaded=False,
            last_error=f"unsupported_binding:{binding}",
            reason="unsupported_binding",
        )
        logger.warning(
            "Unsupported RERANK_BINDING=%r; rerank disabled for queries.", binding
        )
        return None

    async def rerank_func(
        query: str,
        documents: list,
        top_n: int | None = None,
        extra_body: dict | None = None,
    ):
        effective_top_n = top_n or configured_top_n
        started = time.perf_counter()
        kwargs = {
            "query": query,
            "documents": documents,
            "top_n": effective_top_n,
            "api_key": binding_api_key,
            "model": model_name,
        }
        if binding_host:
            kwargs["base_url"] = binding_host
        if provider == "cohere":
            kwargs["enable_chunking"] = as_bool(
                os.getenv("RERANK_ENABLE_CHUNKING"), False
            )
            kwargs["max_tokens_per_doc"] = as_int(
                os.getenv("RERANK_MAX_TOKENS_PER_DOC"), 4096
            )
        try:
            result = await selected_rerank_func(**kwargs, extra_body=extra_body)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            update_rerank_runtime_state(
                enabled=True,
                model_loaded=True,
                last_error=str(exc),
                reason="provider_call_failed",
            )
            logger.warning(
                "Rerank execution failed; fallback to original retrieval chunks. provider=%s model=%s retrieved_chunks_before_rerank=%s retrieved_chunks_after_rerank=%s rerank_top_n=%s rerank_elapsed=%.3fs rerank_applied=false rerank_failed_fallback=true error=%s",
                provider,
                model_name,
                len(documents),
                len(documents),
                effective_top_n or "QueryParam.chunk_top_k",
                elapsed,
                exc,
            )
            raise

        elapsed = time.perf_counter() - started
        result_count = len(result) if isinstance(result, list) else 0
        update_rerank_runtime_state(
            enabled=True,
            model_loaded=True,
            last_error=None,
            reason=None,
        )
        logger.info(
            "Rerank execution completed provider=%s model=%s retrieved_chunks_before_rerank=%s retrieved_chunks_after_rerank=%s rerank_top_n=%s rerank_elapsed=%.3fs rerank_applied=%s rerank_failed_fallback=false",
            provider,
            model_name,
            len(documents),
            result_count,
            effective_top_n or "QueryParam.chunk_top_k",
            elapsed,
            result_count > 0,
        )
        return result

    previous_status = current_rerank_runtime_status()
    previous_reason = previous_status.get("reason")
    logger.info(
        "Rerank model loaded: provider=%s model=%s base_url=%s top_n=%s rerank_model_func=not None fallback=original retrieval chunks",
        provider,
        model_name,
        binding_host or "provider_default",
        configured_top_n or "QueryParam.chunk_top_k",
    )
    update_rerank_runtime_state(
        enabled=True,
        model_loaded=True,
        last_error=previous_status.get("last_error")
        if previous_reason == "provider_call_failed"
        else None,
        reason=previous_reason if previous_reason == "provider_call_failed" else None,
    )
    return rerank_func


def rag_rerank_model_func_available(rag: RAGAnything) -> bool:
    lightrag = getattr(rag, "lightrag", None)
    if lightrag is not None and callable(getattr(lightrag, "rerank_model_func", None)):
        return True
    lightrag_kwargs = getattr(rag, "lightrag_kwargs", {})
    if isinstance(lightrag_kwargs, dict) and callable(
        lightrag_kwargs.get("rerank_model_func")
    ):
        return True
    return bool(getattr(rag, "_rerank_model_func_available", False))


def build_rerank_query_config(rag: RAGAnything) -> dict[str, Any]:
    status = current_rerank_runtime_status()
    requested = bool(status["requested"])
    model = status.get("model")
    provider = status.get("provider")
    model_func_available = rag_rerank_model_func_available(rag)
    enabled = bool(requested and model and model_func_available)
    if not requested:
        label = "disabled"
    elif enabled:
        label = "enabled"
    else:
        label = "requested_but_unavailable"
    reason = None if enabled else status.get("reason")
    if requested and not enabled and not reason:
        reason = "rerank_model_func_unavailable"
    return {
        "requested": requested,
        "enabled": enabled,
        "label": label,
        "model": model,
        "provider": provider,
        "model_func_available": model_func_available,
        "reason": reason,
        "top_n": status.get("top_n"),
        "last_error": status.get("last_error"),
    }
