from __future__ import annotations

import asyncio
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import cors_allowed_origins, load_runtime_config
from backend.rag.rerank import log_rerank_startup_status

if sys.platform.startswith("win"):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_runtime_config()
log_rerank_startup_status()

app = FastAPI(title="RAG Teaching Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.api import cache, chat, documents, file_manager, health, images, profile  # noqa: E402

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(images.router)
app.include_router(chat.router)
app.include_router(cache.router)
app.include_router(profile.router)
app.include_router(file_manager.router)
