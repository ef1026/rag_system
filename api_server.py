"""FastAPI entry point for the RAG teaching assistant.

The frontend must only call this API. RAG parsing, indexing, embedding, LLM
calls, and local file access all stay on the backend.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from raganything import RAGAnything, RAGAnythingConfig
from raganything.utils import insert_text_content, separate_content


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
RAG_STORAGE_DIR = BASE_DIR / "rag_storage"

ALLOWED_EXTENSIONS = {".pdf"}
ANSWER_LEVELS = {
    "beginner": "请用直观类比解释概念，避免堆砌术语，并补充必要背景。",
    "undergraduate": "请按定义、公式、物理含义、常见考点的顺序回答。",
    "expert": "请直接进入机制、边界条件、推导要点和局限性。",
}
TEXT_ONLY_INDEXED_MESSAGE = "Document parsed and text-only indexed. MVP text-only mode is active."

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

_rag_instance: Optional[RAGAnything] = None
_rag_lock: asyncio.Lock = asyncio.Lock()


def load_runtime_config() -> None:
    load_dotenv(BASE_DIR / ".env", override=False)
    local_cache = BASE_DIR / ".cache"
    (local_cache / "ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(local_cache / "ultralytics"))
    os.environ.setdefault("MINERU_BACKEND", "pipeline")
    os.environ.setdefault("MINERU_DEVICE", "cuda")
    os.environ.setdefault("MINERU_SOURCE", "modelscope")
    os.environ.setdefault("LLM_MODEL", "qwen-plus")
    os.environ.setdefault(
        "LLM_BINDING_HOST", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    os.environ.setdefault("QWEN_MODEL", "qwen-vl-max")
    os.environ.setdefault(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def as_int(value: str | None, default: int = 0) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


def safe_document_id(filename: str) -> str:
    return Path(filename).name


def document_path(document_id: str) -> Path:
    candidate = UPLOAD_DIR / safe_document_id(document_id)
    if candidate.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Document not found.")
    return candidate


def content_list_path(pdf_path: Path) -> Path | None:
    candidates = sorted(
        OUTPUT_DIR.glob(f"**/{pdf_path.stem}_content_list.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_content_list(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Unexpected content_list format: {path}")
    return [item for item in data if isinstance(item, dict)]


def document_status(pdf_path: Path) -> Literal["uploaded", "processed"]:
    return "processed" if content_list_path(pdf_path) else "uploaded"


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def build_embedding_func() -> EmbeddingFunc:
    model_name = os.getenv("EMBEDDING_LOCAL_MODEL", "BAAI/bge-small-zh-v1.5")
    model = load_embedding_model(model_name)

    async def embed(texts: Iterable[str]):
        return await asyncio.to_thread(lambda: model.encode(list(texts)))

    return EmbeddingFunc(embedding_dim=512, max_token_size=512, func=embed)


def build_model_functions():
    qwen_api_key = os.getenv("QWEN_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
    llm_model = os.getenv("LLM_MODEL", "qwen-plus")
    llm_api_key = os.getenv("LLM_BINDING_API_KEY") or qwen_api_key
    llm_base_url = os.getenv(
        "LLM_BINDING_HOST", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    qwen_model = os.getenv("QWEN_MODEL", "qwen-vl-max")
    qwen_base_url = os.getenv("QWEN_BASE_URL", llm_base_url)

    if not llm_api_key:
        raise RuntimeError("Missing Qwen API key. Set QWEN_API_KEY in .env.")

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
        if not qwen_api_key or not image_data:
            return await llm_func(prompt, system_prompt=system_prompt, **kwargs)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            },
        ]
        return await openai_complete_if_cache(
            qwen_model,
            None,
            system_prompt=None,
            history_messages=messages,
            api_key=qwen_api_key,
            base_url=qwen_base_url,
            **clean_lightrag_kwargs(kwargs),
        )

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


def build_rag() -> RAGAnything:
    llm_func, _vision_func = build_model_functions()
    config = RAGAnythingConfig(
        working_dir=str(RAG_STORAGE_DIR),
        parser_output_dir=str(OUTPUT_DIR),
        parser="mineru",
        parse_method=os.getenv("PARSE_METHOD", "auto"),
        mineru_backend=os.getenv("MINERU_BACKEND", "pipeline"),
        mineru_device=os.getenv("MINERU_DEVICE", "cuda"),
        mineru_source=os.getenv("MINERU_SOURCE", "modelscope"),
        mineru_vram=as_int(os.getenv("MINERU_VRAM"), 0),
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )
    return RAGAnything(
        config=config,
        llm_model_func=llm_func,
        vision_model_func=None,
        embedding_func=build_embedding_func(),
    )


async def get_rag() -> RAGAnything:
    global _rag_instance
    if _rag_instance is None:
        async with _rag_lock:
            if _rag_instance is None:
                _rag_instance = build_rag()
    return _rag_instance


def knowledge_base_not_ready_response() -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "error": "knowledge_base_not_ready",
            "message": "知识库尚未就绪，请先解析文档后再提问。",
        },
    )


async def process_document_text_only(rag: RAGAnything, pdf_path: Path) -> None:
    content_list, doc_id = await rag.parse_document(
        file_path=str(pdf_path),
        output_dir=str(OUTPUT_DIR),
        parse_method=os.getenv("PARSE_METHOD", "auto"),
        display_stats=rag.config.display_content_stats,
    )
    text_content, _multimodal_items = separate_content(content_list)
    if not text_content.strip():
        raise HTTPException(
            status_code=422,
            detail="No text content was extracted from the document.",
        )

    await insert_text_content(
        rag.lightrag,
        input=text_content,
        file_paths=rag._get_file_reference(str(pdf_path)),
        ids=doc_id,
    )


class DocumentSummary(BaseModel):
    id: str
    name: str
    size: int
    status: Literal["uploaded", "processed"]


class SourceItem(BaseModel):
    id: str
    type: str
    page: int | None = None
    text: str


class UploadResponse(BaseModel):
    document: DocumentSummary


class ProcessResponse(BaseModel):
    document: DocumentSummary
    message: str
    sources: list[SourceItem]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    document_id: str | None = None
    level: str = "undergraduate"
    mode: str = "hybrid"


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class CacheResponse(BaseModel):
    ok: bool


class HealthResponse(BaseModel):
    ok: bool
    has_api_key: bool
    mineru_backend: str
    mineru_device: str


def summarize_document(pdf_path: Path) -> DocumentSummary:
    return DocumentSummary(
        id=pdf_path.name,
        name=pdf_path.name,
        size=pdf_path.stat().st_size,
        status=document_status(pdf_path),
    )


def extract_sources(pdf_path: Path, limit: int = 12) -> list[SourceItem]:
    path = content_list_path(pdf_path)
    if not path:
        return []

    sources: list[SourceItem] = []
    for index, item in enumerate(load_content_list(path)):
        item_type = item.get("type")
        text = " ".join(
            (item.get("text") or item.get("table_body") or "").split()
        )
        if item_type not in {"text", "equation", "table"} or not text:
            continue
        page_idx = item.get("page_idx")
        page = int(page_idx) + 1 if isinstance(page_idx, int) else None
        sources.append(
            SourceItem(
                id=f"{pdf_path.stem}-{index}",
                type=str(item_type),
                page=page,
                text=text[:600],
            )
        )
        if len(sources) >= limit:
            break
    return sources


load_runtime_config()
app = FastAPI(title="RAG Teaching Assistant API")

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        ok=True,
        has_api_key=bool(os.getenv("LLM_BINDING_API_KEY") or os.getenv("QWEN_API_KEY")),
        mineru_backend=os.getenv("MINERU_BACKEND", "pipeline"),
        mineru_device=os.getenv("MINERU_DEVICE", "cuda"),
    )


@app.get("/api/documents", response_model=list[DocumentSummary])
async def list_documents() -> list[DocumentSummary]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return [summarize_document(path) for path in sorted(UPLOAD_DIR.glob("*.pdf"))]


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    filename = safe_document_id(file.filename or "")
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / filename
    with target.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return UploadResponse(document=summarize_document(target))


@app.post("/api/documents/{document_id}/process", response_model=ProcessResponse)
async def process_document(document_id: str) -> ProcessResponse:
    pdf_path = document_path(document_id)
    rag = await get_rag()
    init_result = await rag._ensure_lightrag_initialized()
    if not init_result.get("success"):
        raise HTTPException(
            status_code=500,
            detail=init_result.get("error", "LightRAG initialization failed"),
        )

    await process_document_text_only(rag, pdf_path)
    return ProcessResponse(
        document=summarize_document(pdf_path),
        message=TEXT_ONLY_INDEXED_MESSAGE,
        sources=[],
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse | JSONResponse:
    level_prompt = ANSWER_LEVELS.get(request.level, ANSWER_LEVELS["undergraduate"])
    rag = await get_rag()
    pdf_path: Path | None = None

    if request.document_id:
        pdf_path = document_path(request.document_id)
        if document_status(pdf_path) != "processed":
            return knowledge_base_not_ready_response()
    else:
        pdfs = sorted(UPLOAD_DIR.glob("*.pdf")) if UPLOAD_DIR.exists() else []
        pdf_path = pdfs[0] if pdfs else None
        if pdf_path and document_status(pdf_path) != "processed":
            return knowledge_base_not_ready_response()

    init_result = await rag._ensure_lightrag_initialized()
    if not init_result.get("success") or rag.lightrag is None:
        return knowledge_base_not_ready_response()

    prompt = f"{request.question.strip()}\n\n回答风格：{level_prompt}"
    try:
        answer = await rag.aquery(prompt, mode=request.mode)
    except ValueError as error:
        if "No LightRAG instance available" in str(error):
            return knowledge_base_not_ready_response()
        raise
    return ChatResponse(
        answer=answer,
        sources=[],
    )


@app.delete("/api/cache", response_model=CacheResponse)
async def clear_runtime_cache() -> CacheResponse:
    for path in (RAG_STORAGE_DIR, OUTPUT_DIR):
        if path.exists():
            shutil.rmtree(path)
    return CacheResponse(ok=True)
