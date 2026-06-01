from __future__ import annotations

import logging
import os
from dataclasses import fields
from pathlib import Path
from typing import Any

from lightrag.llm.openai import openai_complete_if_cache
from openai import AsyncOpenAI
from raganything import RAGAnything, RAGAnythingConfig

from backend.config import (
    OUTPUT_DIR,
    RAG_STORAGE_DIR,
    as_bool,
    as_int,
    multimodal_enabled,
)
from backend.core.ids import document_cache_key, safe_document_id, safe_document_key
from backend.core.paths import (
    document_storage_dir,
    lightrag_working_root,
    lightrag_workspace,
)
from backend.core.runtime import _rag_instances, _rag_lock
from backend.core.text import response_content_to_text
from backend.rag.embedding import build_embedding_func
from backend.rag.rerank import build_rerank_model_func, current_rerank_runtime_status
from backend.rag.storage import set_lightrag_default_workspace

logger = logging.getLogger("api_server")

def filter_rag_config_kwargs(config_kwargs: dict[str, Any]) -> dict[str, Any]:
    supported_fields = {field.name for field in fields(RAGAnythingConfig)}
    filtered = {
        key: value for key, value in config_kwargs.items() if key in supported_fields
    }
    unsupported_fields = sorted(set(config_kwargs) - supported_fields)
    if unsupported_fields:
        logger.warning(
            "Ignoring unsupported RAGAnythingConfig fields: %s",
            ", ".join(unsupported_fields),
        )
    return filtered


def validate_vision_messages(messages: Any) -> list[dict[str, Any]] | None:
    if not isinstance(messages, list) or not messages:
        logger.warning("Invalid vision messages: expected a non-empty list.")
        return None

    validated: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            logger.warning("Invalid vision message %s: expected object.", message_index)
            return None

        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            logger.warning("Invalid vision message %s: missing role.", message_index)
            return None

        if "content" not in message or message.get("content") is None:
            logger.warning("Invalid vision message %s: missing content.", message_index)
            return None

        content = message["content"]
        if isinstance(content, str):
            validated.append(dict(message))
            continue

        if not isinstance(content, list) or not content:
            logger.warning(
                "Invalid vision message %s: content must be string or non-empty list.",
                message_index,
            )
            return None

        for part_index, part in enumerate(content):
            if not isinstance(part, dict):
                logger.warning(
                    "Invalid vision message %s content part %s: expected object.",
                    message_index,
                    part_index,
                )
                return None

            part_type = part.get("type")
            if part_type == "text":
                if not isinstance(part.get("text"), str):
                    logger.warning(
                        "Invalid vision message %s content part %s: text must be string.",
                        message_index,
                        part_index,
                    )
                    return None
            elif part_type == "image_url":
                image_url = part.get("image_url")
                if (
                    not isinstance(image_url, dict)
                    or not isinstance(image_url.get("url"), str)
                    or not image_url.get("url")
                ):
                    logger.warning(
                        "Invalid vision message %s content part %s: image_url.url is required.",
                        message_index,
                        part_index,
                    )
                    return None
            else:
                logger.warning(
                    "Invalid vision message %s content part %s: unsupported type %r.",
                    message_index,
                    part_index,
                    part_type,
                )
                return None

        validated.append(dict(message))

    return validated


def build_fresh_document_rag(document_id: str, storage_dir: Path) -> RAGAnything:
    normalized_document_id = safe_document_id(document_id)
    logger.info(
        "Building fresh document RAG document_id=%s safe_document_key=%s working_dir=%s",
        normalized_document_id,
        safe_document_key(normalized_document_id),
        storage_dir,
    )
    rag = build_rag(document_id=normalized_document_id, working_dir=storage_dir)
    logger.info(
        "build_fresh_document_rag_created document_id=%s safe_document_key=%s working_dir=%s rag_id=%s lightrag_id=%s",
        normalized_document_id,
        safe_document_key(normalized_document_id),
        storage_dir,
        id(rag),
        id(rag.lightrag) if rag.lightrag is not None else None,
    )
    return rag


def build_model_functions():
    qwen_api_key = os.getenv("QWEN_API_KEY")
    llm_model = os.getenv("LLM_MODEL", "qwen-plus")
    llm_api_key = os.getenv("LLM_BINDING_API_KEY") or qwen_api_key
    llm_base_url = os.getenv(
        "LLM_BINDING_HOST", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    qwen_model = os.getenv("QWEN_VL_MODEL") or os.getenv("QWEN_MODEL", "qwen-vl-max")
    qwen_base_url = os.getenv("QWEN_BASE_URL", llm_base_url)

    if not llm_api_key:
        raise RuntimeError(
            "Missing LLM API key. Set LLM_BINDING_API_KEY or QWEN_API_KEY in .env."
        )

    async def llm_func(
        prompt,
        system_prompt="你是一名严谨的中文 AI 助教。回答必须基于已检索到的文档内容。",
        history_messages=None,
        **kwargs,
    ):
        return await openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=llm_api_key,
            base_url=llm_base_url,
            **clean_lightrag_kwargs(kwargs),
        )

    async def vision_func(
        prompt,
        system_prompt="你是一名擅长理解图像、表格、公式和文档截图的中文 AI 助教。",
        image_data=None,
        **kwargs,
    ):
        if not qwen_api_key:
            logger.warning("QWEN_API_KEY is missing; skipping vision model call.")
            return ""

        call_kwargs = dict(kwargs)
        raw_messages = call_kwargs.pop("messages", None)
        for key in ("response_format", "keyword_extraction", "image_data"):
            call_kwargs.pop(key, None)

        if raw_messages is not None:
            messages = validate_vision_messages(raw_messages)
            if messages is None:
                return ""
        else:
            if not image_data:
                logger.warning("image_data is empty; skipping vision model call.")
                return ""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or ""},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ],
                },
            ]

        client = None
        try:
            client = AsyncOpenAI(api_key=qwen_api_key, base_url=qwen_base_url)
            response = await client.chat.completions.create(
                model=qwen_model,
                messages=messages,
                **call_kwargs,
            )
            content = (
                response_content_to_text(response.choices[0].message.content)
                if response.choices
                else ""
            )
            if not content:
                logger.warning("Vision model returned empty content.")
            return content
        except Exception as exc:
            logger.warning("Vision model call failed: %s", exc)
            return ""
        finally:
            if client is not None:
                try:
                    await client.close()
                except Exception as exc:
                    logger.warning("Failed to close vision model client: %s", exc)

    return llm_func, vision_func


def clean_lightrag_kwargs(kwargs: dict) -> dict:
    cleaned = dict(kwargs)
    for key in ("response_format", "keyword_extraction", "image_data"):
        cleaned.pop(key, None)

    messages = cleaned.get("messages")
    if messages:
        normalized = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            normalized.append({"role": message.get("role", "user"), "content": content})
        cleaned["messages"] = normalized
    return cleaned


def build_rag(
    document_id: str | None = None, working_dir: Path | None = None
) -> RAGAnything:
    llm_func, vision_func = build_model_functions()
    rerank_model_func = build_rerank_model_func()
    normalized_document_id = safe_document_id(document_id) if document_id else None
    if working_dir is not None:
        effective_working_dir = working_dir
    elif normalized_document_id:
        effective_working_dir = document_storage_dir(normalized_document_id)
    else:
        effective_working_dir = RAG_STORAGE_DIR
    workspace = (
        lightrag_workspace(normalized_document_id) if normalized_document_id else None
    )
    enable_multimodal = multimodal_enabled()
    enable_image_processing = enable_multimodal and as_bool(
        os.getenv("ENABLE_IMAGE_PROCESSING"), True
    )
    enable_table_processing = enable_multimodal and as_bool(
        os.getenv("ENABLE_TABLE_PROCESSING"), True
    )
    enable_equation_processing = enable_multimodal and as_bool(
        os.getenv("ENABLE_EQUATION_PROCESSING"), True
    )

    if enable_image_processing and not os.getenv("QWEN_API_KEY"):
        logger.warning(
            "ENABLE_MULTIMODAL=true and ENABLE_IMAGE_PROCESSING=true, but "
            "QWEN_API_KEY is missing; image processing is disabled."
        )
        enable_image_processing = False

    config_kwargs = {
        "working_dir": str(effective_working_dir),
        "parser_output_dir": str(OUTPUT_DIR),
        "parser": "mineru",
        "parse_method": os.getenv("PARSE_METHOD", "auto"),
        "mineru_backend": os.getenv("MINERU_BACKEND", "pipeline"),
        "mineru_device": os.getenv("MINERU_DEVICE", "cuda"),
        "mineru_source": os.getenv("MINERU_SOURCE", "modelscope"),
        "mineru_vram": as_int(os.getenv("MINERU_VRAM"), 0),
        "enable_image_processing": enable_image_processing,
        "enable_table_processing": enable_table_processing,
        "enable_equation_processing": enable_equation_processing,
    }
    config = RAGAnythingConfig(**filter_rag_config_kwargs(config_kwargs))
    lightrag_kwargs: dict[str, Any] = {}
    if normalized_document_id and workspace:
        lightrag_kwargs["working_dir"] = str(lightrag_working_root())
        lightrag_kwargs["workspace"] = workspace
    if rerank_model_func is not None:
        lightrag_kwargs["rerank_model_func"] = rerank_model_func
    rerank_status = current_rerank_runtime_status()
    logger.info(
        "build_rag rerank wiring document_id=%s rerank_requested=%s rerank_enabled=%s rerank_model_func=%s rerank_provider=%s rerank_model=%s",
        normalized_document_id,
        rerank_status["requested"],
        rerank_status["enabled"],
        "not None" if rerank_model_func is not None else "None",
        rerank_status.get("provider") or "none",
        rerank_status.get("model") or "none",
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_func,
        vision_model_func=vision_func
        if enable_multimodal and enable_image_processing
        else None,
        embedding_func=build_embedding_func(),
        lightrag_kwargs=lightrag_kwargs,
    )
    setattr(rag, "_rerank_model_func_available", rerank_model_func is not None)
    setattr(rag, "_rerank_provider", rerank_status.get("provider"))
    setattr(rag, "_rerank_model", rerank_status.get("model"))
    logger.info(
        "build_rag_created document_id=%s safe_document_key=%s working_dir=%s lightrag_working_dir=%s lightrag_workspace=%s rag_id=%s lightrag_id=%s",
        normalized_document_id,
        safe_document_key(normalized_document_id) if normalized_document_id else None,
        effective_working_dir,
        lightrag_kwargs.get("working_dir", str(effective_working_dir)),
        lightrag_kwargs.get("workspace"),
        id(rag),
        id(rag.lightrag) if rag.lightrag is not None else None,
    )
    return rag


async def get_rag(document_id: str | None = None) -> RAGAnything:
    if not document_id:
        raise ValueError("document_id is required for document-scoped retrieval.")

    normalized_document_id = safe_document_id(document_id)
    storage_dir = document_storage_dir(normalized_document_id)
    cache_key = document_cache_key(normalized_document_id)
    if cache_key not in _rag_instances:
        async with _rag_lock:
            if cache_key not in _rag_instances:
                set_lightrag_default_workspace(lightrag_workspace(normalized_document_id))
                logger.info(
                    "Creating document-scoped RAG instance document_id=%s safe_document_key=%s storage_dir=%s lightrag_working_dir=%s lightrag_workspace=%s",
                    normalized_document_id,
                    cache_key,
                    storage_dir,
                    lightrag_working_root(),
                    lightrag_workspace(normalized_document_id),
                )
                _rag_instances[cache_key] = build_rag(
                    document_id=normalized_document_id,
                    working_dir=storage_dir,
                )
    return _rag_instances[cache_key]
