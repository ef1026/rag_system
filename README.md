# rag_system

这是一个面向教学场景的本地 RAG 应用。当前架构已经改为 FastAPI 后端 + Next.js 前端：知识库使用 LightRAG，文档解析使用 RAG-Anything 的处理链，MinerU 2.7.6 默认使用 GPU 解析。模型接口统一为 Qwen/DashScope。`app.py` 仅作为 legacy Streamlit 参考入口保留。

## 主流程

1. 用户在 `frontend/` Next.js 页面上传 PDF。
2. PDF 保存到 `uploads/`。
3. `api_server.py` 调用 `raganything.parser.MineruParser` 和 RAG-Anything 解析文档。
4. 解析结果写入 `output/`，向量、图谱和缓存写入 `rag_storage/`。
5. 默认启用多模态处理；如需 text-only，可设置 `ENABLE_MULTIMODAL=false`。
6. 用户提问时，前端调用 FastAPI，后端通过 `RAGAnything.aquery()` 检索知识库并调用 Qwen 生成回答。

## 目录

```text
app.py                 legacy Streamlit 入口，仅保留作参考
api_server.py          FastAPI 后端入口
raganything/           RAG-Anything 核心处理代码
raganything/parser.py  MinerU/Docling 文档解析适配层
raganything/config.py  运行配置和环境变量默认值
uploads/               本地上传和测试 PDF
output/                MinerU 解析输出
rag_storage/           LightRAG 知识库、向量库、图谱和缓存
frontend/              Next.js 前端，可部署到 Vercel
env.example            环境变量模板
requirements.txt       Python 运行依赖
```

## 环境

项目当前按 `advdoc` conda 环境维护：

```powershell
conda activate advdoc
pip install -r requirements.txt
```

确认 MinerU：

```powershell
mineru --version
```

期望版本：

```text
mineru, version 2.7.6
```

## 配置

本项目只读取 `.env`。不要再维护 `rag.env` 或 `RAG.env`，否则容易出现“我改了 key 但程序没用上”的情况。

复制模板：

```powershell
Copy-Item env.example .env
```

Qwen/DashScope 配置：

```env
ENABLE_MULTIMODAL=true
ENABLE_IMAGE_PROCESSING=true
ENABLE_TABLE_PROCESSING=true
ENABLE_EQUATION_PROCESSING=true
ENABLE_FORMULA_PROCESSING=true
ENABLE_GENERIC_PROCESSING=false

MAX_IMAGE_ITEMS=0
MAX_TABLE_ITEMS=0
MAX_EQUATION_ITEMS=0
MAX_FORMULA_ITEMS=0
MAX_GENERIC_ITEMS=0

QWEN_API_KEY=
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_VL_MODEL=qwen-vl-max

LLM_MODEL=qwen-plus
LLM_BINDING_HOST=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_BINDING_API_KEY=
```

说明：

- `LLM_*` 用于普通文本问答、RAG 总结和知识库写入阶段的语言模型调用。
- `ENABLE_MULTIMODAL=true` 是推荐默认值；如需回滚到 text-only，可设为 `false`。
- `ENABLE_IMAGE_PROCESSING`、`ENABLE_TABLE_PROCESSING`、`ENABLE_EQUATION_PROCESSING`、`ENABLE_FORMULA_PROCESSING` 分别控制图片、表格、公式和 formula item 处理。
- `ENABLE_GENERIC_PROCESSING=false` 默认关闭 generic item；只有显式开启后才会处理 generic。
- `MAX_IMAGE_ITEMS`、`MAX_TABLE_ITEMS`、`MAX_EQUATION_ITEMS`、`MAX_FORMULA_ITEMS`、`MAX_GENERIC_ITEMS` 控制每类多模态 item 的处理上限。`0`、`-1`、`unlimited`、`none` 表示无限制；如担心成本，可设置正整数，例如 `MAX_IMAGE_ITEMS=5` 或 `MAX_EQUATION_ITEMS=10`。
- `QWEN_*` 用于 qwen-vl-max 图片理解；如果启用图片处理但缺少 `QWEN_API_KEY`，后端会记录 warning 并自动禁用 image processing，不影响服务启动。
- `QWEN_API_KEY` 和 `LLM_BINDING_API_KEY` 可以填同一把 DashScope API key。
- 如果只填了 `QWEN_API_KEY`，代码也会自动把它用于文本模型调用。

启用 image-only 多模态：

```env
ENABLE_MULTIMODAL=true
ENABLE_IMAGE_PROCESSING=true
ENABLE_TABLE_PROCESSING=false
ENABLE_EQUATION_PROCESSING=false
ENABLE_FORMULA_PROCESSING=false
QWEN_API_KEY=your_qwen_dashscope_api_key
```

回滚到 text-only：

```env
ENABLE_MULTIMODAL=false
```

MinerU 默认 GPU 模式：

```env
PARSER=mineru
PARSE_METHOD=auto
MINERU_BACKEND=pipeline
MINERU_DEVICE=cuda
MINERU_SOURCE=modelscope
```

MinerU 2.7.6 的 `-d/--device` 参数只对 `pipeline` backend 生效，所以这里固定推荐 `pipeline + cuda`。指定显卡：
`MINERU_SOURCE=modelscope` 会从 ModelScope 查找/下载模型，避免默认 Hugging Face 源在国内网络下失败。模型已经完整下载到本机后，也可以改成 `MINERU_SOURCE=local` 并按 MinerU 的 `mineru.json` 配置本地模型目录。

```env
MINERU_DEVICE=cuda:0
```

限制单进程显存：

```env
MINERU_VRAM=8
```

## 启动

后端：

```powershell
conda activate advdoc
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

前端：

```powershell
cd frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

打开 `http://localhost:3000`。前端通过 `NEXT_PUBLIC_API_BASE_URL` 连接后端；不要把 Qwen、embedding、向量库或数据库密钥放进前端环境变量。

legacy Streamlit 参考入口：

```powershell
conda activate advdoc
streamlit run app.py
```

## 开发者检查

```powershell
conda activate advdoc
python -m py_compile app.py raganything\parser.py raganything\processor.py raganything\raganything.py
python -m raganything.parser --check
```

## Vision smoke test

Use this helper to test the qwen-vl-max `vision_func` path with one local image. It does not use the frontend and does not process a PDF.

```powershell
conda activate advdoc
python scripts/test_vision_func.py path\to\image.jpg
```

The script reads `QWEN_API_KEY` from `.env` or the current environment and exits with a clear message if the key or image file is missing.

## RAG pipeline status

The current FastAPI pipeline is:

- Upload: `/api/upload` saves PDFs in `uploads/`.
- Parser: `api_server.py` calls RAG-Anything with MinerU. MinerU output is written under `output/<document>/<method>/`, including `<stem>_content_list.json`, markdown, intermediate JSON files, rendered PDFs, and extracted images.
- Content split: `raganything.utils.separate_content()` separates `content_list` into text content and multimodal items. The backend further normalizes image, table, equation, formula, and generic items before multimodal indexing.
- Text chunking/indexing: text is inserted through `insert_text_content()` into a document-scoped LightRAG instance. Each PDF writes to `rag_storage/documents/<safe-document-key>/`, including its own `kv_store_text_chunks.json` and `vdb_chunks.json`.
- Multimodal indexing: when `ENABLE_MULTIMODAL=true`, enabled multimodal processors generate descriptions/chunks for image, table, equation, and formula items, then write them into that same document-scoped LightRAG chunk/vector/graph store.
- Embedding: `EMBEDDING_LOCAL_MODEL`, default `BAAI/bge-small-zh-v1.5`, is loaded server-side with `SentenceTransformer`. No embedding key is exposed to the frontend.
- Vector store: LightRAG local JSON vector stores in `rag_storage/documents/<safe-document-key>/vdb_*.json`.
- Graph store: LightRAG graph data in `rag_storage/documents/<safe-document-key>/graph_chunk_entity_relation.graphml` plus entity/relation KV stores.
- Retrieval mode: `/api/chat` defaults to `hybrid`; the frontend currently sends `mode: "hybrid"`. Chat requires `document_id` and loads only that document's scoped storage.
- VLM enhanced query: enabled automatically when `ENABLE_MULTIMODAL=true`, image processing is enabled, and a vision model function is available. If no valid images are found in retrieved context, RAG-Anything falls back to normal text query.
- Rerank: controlled by `ENABLE_RERANK`, `RERANK_BINDING`, `RERANK_MODEL`, `RERANK_BINDING_API_KEY`, optional `RERANK_BASE_URL`, and optional `RERANK_TOP_N`. Default is disabled. If rerank is requested but model/provider/key initialization is unavailable, chat automatically passes `enable_rerank=false` and continues without rerank.
- Sources/citations: document preview sources are extracted from MinerU `content_list`; chat responses currently return `sources: []` because retrieval raw data is not mapped back to source/page/score metadata yet.

Document-scoped retrieval:

- New processing runs no longer use the legacy global `rag_storage/` index for chat retrieval.
- Existing documents parsed before document-scoped storage must be processed again so their indexes are created under `rag_storage/documents/<safe-document-key>/`.
- The backend does not migrate or fall back to legacy global indexes; if a selected document has no scoped storage, `/api/chat` returns knowledge-base-not-ready.

Development status endpoint:

```powershell
curl http://127.0.0.1:8000/api/rag/status
```

Stable no-rerank mode:

```env
ENABLE_RERANK=false
RERANK_MODEL=
```

To enable remote rerank, configure a supported LightRAG rerank provider on the backend only:

```env
ENABLE_RERANK=true
RERANK_BINDING=aliyun
RERANK_MODEL=gte-rerank-v2
RERANK_BINDING_API_KEY=your_backend_only_key
# RERANK_BASE_URL=
# RERANK_TOP_N=20
```

For DashScope/Aliyun, `RERANK_BINDING_API_KEY` may reuse the same DashScope key as `QWEN_API_KEY` if that key has access to the rerank model. Keeping `RERANK_BINDING_API_KEY` separate is recommended so the frontend never receives any LLM, embedding, vector DB, database, or rerank key.

`GET /api/rag/status` includes the effective retrieval status:

```json
{
  "retrieval": {
    "default_mode": "hybrid",
    "rerank_requested": true,
    "rerank_enabled": true,
    "rerank_model": "gte-rerank-v2",
    "rerank_provider": "aliyun",
    "rerank_model_loaded": true,
    "rerank_last_error": null,
    "reason": null,
    "rerank_top_n": 20
  }
}
```

If rerank is requested but unavailable, `rerank_requested` remains `true`, `rerank_enabled` is `false`, and `reason` is set to values such as `missing_model`, `missing_api_key`, or `provider_init_failed`.

查看 MinerU 实际命令：

```powershell
python -c "from raganything.parser import MineruParser; print(' '.join(MineruParser._build_mineru_command('RAG_test1.pdf','output')))"
```

期望包含：

```text
-b pipeline -d cuda
```

## 运行资产

保留的测试和运行资产：

- `RAG_test1.pdf`
- `uploads/`
- `output/`
- `rag_storage/`

这些文件用于复现实验状态和测试知识库。如果要从零开始，可以在应用里点击“清空知识库缓存”，或手动删除 `output/` 与 `rag_storage/` 后重新解析 PDF。

## 常见问题

如果 Qwen 报认证错误，检查 `.env` 中的 key 是否真实有效：

```env
QWEN_API_KEY=...
LLM_BINDING_API_KEY=...
```

如果 MinerU 没有使用 GPU，检查：

```powershell
echo $env:MINERU_BACKEND
echo $env:MINERU_DEVICE
```

应为：

```text
pipeline
cuda
```
