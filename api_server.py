import os
import asyncio
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

app = FastAPI(title="RAG-Anything API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rag-frontend-wine.vercel.app", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("LLM_BINDING_API_KEY")
base_url = os.getenv("LLM_BINDING_HOST")

llm = lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
    "gpt-4o-mini",
    prompt,
    system_prompt=system_prompt,
    history_messages=history_messages,
    api_key=api_key,
    base_url=base_url,
    **kwargs,
)

embedding = EmbeddingFunc(
    embedding_dim=3072,
    max_token_size=8192,
    func=lambda texts: openai_embed(
        texts,
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        api_key=api_key,
        base_url=base_url,
    ),
)

rag = RAGAnything(
    config=RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        parse_method="auto",
    ),
    llm_model_func=llm,
    embedding_func=embedding,
)


class ChatRequest(BaseModel):
    message: str
    persona: str = "undergraduate"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    persona_prompts = {
        "beginner": "用直觉化、类比化的方式讲解，避免专业术语",
        "undergraduate": "包含定义、公式、物理意义和考点结构",
        "expert": "跳过基础定义，聚焦机制与边界条件",
    }
    system_hint = persona_prompts.get(req.persona, persona_prompts["undergraduate"])
    query = f"[系统提示：{system_hint}]\n\n{req.message}"
    answer = await rag.aquery(query, mode="hybrid")
    return {"answer": answer, "sources": []}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    os.makedirs("./uploads", exist_ok=True)
    upload_path = f"./uploads/{file.filename}"
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)
    await rag.process_document_complete(file_path=upload_path, output_dir="./output")
    return {"status": "ok", "filename": file.filename}


@app.delete("/api/cache")
async def clear_cache():
    for d in ["./rag_storage", "./output"]:
        if os.path.exists(d):
            shutil.rmtree(d)
            os.makedirs(d)
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)