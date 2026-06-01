from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
RAG_STORAGE_DIR = BASE_DIR / "rag_storage"
APP_DATA_DIR = BASE_DIR / "app_data"
METADATA_DB_PATH = APP_DATA_DIR / "metadata.sqlite3"
logger = logging.getLogger("api_server")

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
    os.environ.setdefault("ENABLE_MULTIMODAL", "true")
    os.environ.setdefault("ENABLE_IMAGE_PROCESSING", "true")
    os.environ.setdefault("ENABLE_TABLE_PROCESSING", "true")
    os.environ.setdefault("ENABLE_EQUATION_PROCESSING", "true")
    os.environ.setdefault("ENABLE_FORMULA_PROCESSING", "true")
    os.environ.setdefault("ENABLE_GENERIC_PROCESSING", "false")
    os.environ.setdefault("QWEN_VL_MODEL", "qwen-vl-max")
    os.environ.setdefault(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    os.environ.setdefault("ENABLE_RERANK", "false")
    os.environ.setdefault("RERANK_MODEL", "")


def as_int(value: str | None, default: int = 0) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


def as_bool(value: str | None, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid boolean value %r; using default %s.", value, default)
    return default


def multimodal_enabled() -> bool:
    return as_bool(os.getenv("ENABLE_MULTIMODAL"), True)


def formula_processing_enabled() -> bool:
    return multimodal_enabled() and as_bool(
        os.getenv("ENABLE_FORMULA_PROCESSING"), True
    )


def generic_processing_enabled() -> bool:
    return as_bool(os.getenv("ENABLE_GENERIC_PROCESSING"), False)


def mineru_runtime_kwargs() -> dict[str, str]:
    kwargs: dict[str, str] = {}
    for env_name, kwarg_name, default in (
        ("MINERU_BACKEND", "backend", "pipeline"),
        ("MINERU_DEVICE", "device", "cuda"),
        ("MINERU_SOURCE", "source", "modelscope"),
    ):
        value = os.getenv(env_name, default)
        if value:
            kwargs[kwarg_name] = value
    return kwargs


def cors_allowed_origins() -> list[str]:
    return [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
        if origin.strip()
    ]
