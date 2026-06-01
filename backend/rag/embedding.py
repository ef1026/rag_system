from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import Iterable

from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def build_embedding_func() -> EmbeddingFunc:
    model_name = os.getenv("EMBEDDING_LOCAL_MODEL", "BAAI/bge-small-zh-v1.5")
    model = load_embedding_model(model_name)

    async def embed(texts: Iterable[str]):
        return await asyncio.to_thread(lambda: model.encode(list(texts)))

    return EmbeddingFunc(embedding_dim=512, max_token_size=512, func=embed)
