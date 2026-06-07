from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException
from raganything import RAGAnything

from backend.core.ids import safe_document_id, same_document_file
from backend.core.models import DocumentAnswer
from backend.core.paths import (
    content_list_path,
    document_path,
    document_storage_dir,
    load_content_list,
)
from backend.core.text import (
    content_from_full_doc,
    first_non_empty_text,
    response_content_to_text,
)
from backend.prompts.templates import (
    build_task_system_prompt,
    level_guidance,
    profile_prompt_prefix,
)
from backend.rag.factory import build_model_functions
from backend.rag.storage import read_json_dict, vdb_records_by_file

logger = logging.getLogger("api_server")

def answer_needs_fallback(answer: str) -> bool:
    if not answer.strip():
        return True
    normalized = answer.lower()
    markers = (
        "document chunks 为空",
        "document chunks",
        "document chunks is empty",
        "reference document list",
        "reference document list 为空",
        "context 中不包含任何实际",
        "context 不包含任何实际",
        "当前文档中没有足够信息",
        "没有足够信息",
        "未提供足够信息",
        "未提取出可用于",
        "无法基于",
        "无法生成",
        "不包含任何实际的 pdf 文档内容",
        "无法获取该 pdf 的总体文本",
        "无法获取 pdf 文档内容",
        "无法获取该 pdf 文档内容",
        "no document chunks",
        "empty context",
        "context empty",
    )
    return any(marker in normalized for marker in markers)


def collect_parser_content_for_document(document_id: str, max_chars: int) -> str:
    try:
        path = content_list_path(document_path(document_id))
    except HTTPException:
        return ""
    if not path:
        return ""

    parts: list[str] = []
    try:
        items = load_content_list(path)
    except Exception as exc:
        logger.warning("Failed to load parser content_list for %s: %s", document_id, exc)
        return ""

    for item in items:
        item_type = str(item.get("type", "")).lower()
        if item_type not in {"text", "equation", "table", "image"}:
            continue
        text = first_non_empty_text(
            item,
            "text",
            "table_body",
            "caption",
            "image_caption",
            "table_caption",
            "equation",
            "latex",
        )
        if text:
            parts.append(text)
        if sum(len(part) for part in parts) >= max_chars:
            break
    return "\n\n".join(parts)[:max_chars]


def collect_direct_context(
    document_id: str, max_chars: int = 24000
) -> tuple[str, str, list[str]]:
    filename = safe_document_id(document_id)
    storage_dir = document_storage_dir(filename)
    warnings: list[str] = []

    if not storage_dir.exists():
        parser_context = collect_parser_content_for_document(filename, max_chars)
        return parser_context, "parser_content_list", warnings

    doc_statuses = read_json_dict(storage_dir / "kv_store_doc_status.json")
    text_chunks = read_json_dict(storage_dir / "kv_store_text_chunks.json")
    current_doc_id = ""
    current_doc_status: dict[str, Any] | None = None
    for doc_id, status in doc_statuses.items():
        if isinstance(status, dict) and same_document_file(status.get("file_path"), filename):
            current_doc_id = str(doc_id)
            current_doc_status = status
            break

    full_docs = read_json_dict(storage_dir / "kv_store_full_docs.json")
    if current_doc_id:
        full_doc_text = content_from_full_doc(full_docs.get(current_doc_id))
        if full_doc_text:
            return full_doc_text[:max_chars], "full_docs", warnings

    if current_doc_status:
        chunk_parts: list[str] = []
        for chunk_id in current_doc_status.get("chunks_list", []):
            chunk = text_chunks.get(str(chunk_id))
            if not isinstance(chunk, dict):
                continue
            if chunk.get("file_path") and not same_document_file(chunk.get("file_path"), filename):
                warnings.append(f"skipped non-current text_chunk {chunk_id}")
                continue
            content = str(chunk.get("content", "")).strip()
            if content:
                chunk_parts.append(content)
            if sum(len(part) for part in chunk_parts) >= max_chars:
                break
        if chunk_parts:
            return "\n\n".join(chunk_parts)[:max_chars], "text_chunks", warnings

    parser_context = collect_parser_content_for_document(filename, max_chars)
    if parser_context.strip():
        return parser_context, "parser_content_list", warnings

    vdb_records, vdb_warnings = vdb_records_by_file(storage_dir, filename)
    warnings.extend(vdb_warnings)
    vdb_parts: list[str] = []
    for record in vdb_records:
        content = str(record.get("content", "")).strip()
        if content:
            vdb_parts.append(content)
        if sum(len(part) for part in vdb_parts) >= max_chars:
            break
    return "\n\n".join(vdb_parts)[:max_chars], "vdb_chunks", warnings


async def direct_context_fallback(
    rag: RAGAnything,
    question: str,
    document_id: str,
    level_key: str,
    level_prompt: str,
    profile_prompt: str | None = None,
) -> tuple[str, str]:
    context, source_path, warnings = collect_direct_context(document_id)
    if not context.strip():
        return "", "none"
    if warnings:
        logger.warning(
            "Direct context fallback skipped unsafe fragments document_id=%s warnings=%s",
            document_id,
            "; ".join(warnings[:8]),
        )

    fallback_prompt = (
        profile_prompt_prefix(profile_prompt)
        + f"当前选中文档是：{document_id}。\n"
        "用户原始问题如下。不要分类、不要改写、不要替换成内置任务模板：\n"
        f"{question}\n\n"
        "以下摘录只来自当前文档。请只基于这些摘录回答；"
        "如果摘录不足以回答，请明确说明信息不足，不要引用其他文档或编造来源。\n\n"
        f"语言难度：{level_guidance(level_key, level_prompt)}\n\n"
        f"当前文档摘录：\n{context}\n\n"
        "请按用户原始问题直接输出。"
    )
    answer = await rag.aquery(
        fallback_prompt,
        mode="bypass",
        system_prompt=build_task_system_prompt(profile_prompt),
        vlm_enhanced=False,
        enable_rerank=False,
    )
    return response_content_to_text(answer), f"direct_context_fallback:{source_path}"


def build_multi_document_synthesis_prompt(
    question: str,
    document_answers: list[DocumentAnswer],
    level_key: str,
    level_prompt: str,
    profile_prompt: str | None = None,
) -> str:
    answer_sections = []
    for index, document_answer in enumerate(document_answers, start=1):
        is_direct_summary = document_answer.answer_source_path.startswith("direct_summary")
        answer = document_answer.answer[:12000 if is_direct_summary else 8000]
        direct_context_note = synthesis_context_note(document_answer)
        section_label = "文档摘要原文摘录" if is_direct_summary else "单文档检索回答"
        answer_sections.append(
            f"[文档 {index}: {document_answer.name} / {document_answer.document_id}]"
            f"\n[{section_label}]\n{answer}"
            f"{direct_context_note}"
        )

    shared_rules = (
        profile_prompt_prefix(profile_prompt)
        + "你正在综合多个 PDF 文档的独立检索结果。"
        "下面每一段都已经由对应文档的 document-scoped RAG 单独生成，"
        "不要引入这些段落之外的信息，也不要伪造页码、分数或引用。"
        "不要判断用户问题属于哪一类任务，不要套用任何专用模板；"
        "最终回答必须直接按用户原始问题执行。\n\n"
        f"用户问题：{question.strip()}\n\n"
        f"语言难度：{level_guidance(level_key, level_prompt)}\n\n"
        "回答规则：用户要求什么就输出什么；如果用户指定数量、格式或范围，严格按原文执行。"
        "不要新增用户没有要求的任务结构。\n\n"
        "宽召回规则：如果单文档检索回答声称信息不足，但同一文档的“宽召回原文摘录”包含可用内容，"
        "应优先使用原文摘录，不要把该文档判定为缺失。"
        "先从所有文档摘录中筛选与用户问题相关的材料，再生成最终答案。\n\n"
        "多文档来源规则：覆盖所有已成功检索的文档；每个关键结论、题目或建议都要标注来源文档；"
        "如果某个问题只被部分文档覆盖，请直接说明。\n\n"
    )
    return (
        shared_rules
        + (
        "图片引用规则：回答必须使用 Markdown；如果文档独立回答中已有真实图片 ID 引用，"
        "例如 [[image:012345abcdef012345abcdef012345abcdef012345abcdef012345abcdef0123]]，"
        "且综合回答仍然依赖该图，请把占位符保留在对应段落后；每张图最多引用一次；"
        "不要在答案末尾堆全部图片；不要输出本地文件路径；不要输出字面量 image_id；不要编造 image_id；"
        "只能使用文档独立回答里已经出现的真实图片 ID。\n\n"
        "文档独立回答：\n\n"
        )
        + "\n\n---\n\n".join(answer_sections)
    )


def synthesis_context_note(document_answer: DocumentAnswer) -> str:
    if document_answer.answer_source_path.startswith("direct_summary"):
        return ""
    needs_wide_context = (
        document_answer.fallback_used
        or document_answer.answer_source_path.startswith("direct_context_fallback")
        or answer_needs_fallback(document_answer.answer)
    )
    if needs_wide_context:
        direct_context, context_source, context_warnings = collect_direct_context(
            document_answer.document_id,
            max_chars=10000,
        )
        if context_warnings:
            logger.warning(
                "Multi-document synthesis direct context warnings document_id=%s warnings=%s",
                document_answer.document_id,
                "; ".join(context_warnings[:8]),
            )
        if direct_context.strip():
            return (
                f"\n\n[宽召回原文摘录 source={context_source}]"
                f"\n{direct_context[:10000]}"
            )
        return ""

    evidence_parts: list[str] = []
    for source in document_answer.sources[:6]:
        text = source.text.strip()
        if text:
            evidence_parts.append(text)
        if sum(len(part) for part in evidence_parts) >= 2400:
            break
    if not evidence_parts:
        return ""
    excerpt = "\n\n".join(evidence_parts)[:2400]
    return f"\n\n[检索证据摘录]\n{excerpt}"


async def synthesize_multi_document_answer(
    question: str,
    document_answers: list[DocumentAnswer],
    level_key: str,
    level_prompt: str,
    profile_prompt: str | None = None,
) -> str:
    llm_func, _vision_func = build_model_functions()
    prompt = build_multi_document_synthesis_prompt(
        question,
        document_answers,
        level_key,
        level_prompt,
        profile_prompt,
    )
    answer = await llm_func(
        prompt,
        system_prompt=build_task_system_prompt(profile_prompt),
    )
    return response_content_to_text(answer)
