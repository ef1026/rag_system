# rag_system

这是一个面向教学场景的本地 RAG 应用。当前架构已经改为 FastAPI 后端 + Next.js 前端：知识库使用 LightRAG，文档解析使用 RAG-Anything 的处理链，MinerU 2.7.6 默认使用 GPU 解析。模型接口统一为 Qwen/DashScope。`app.py` 仅作为 legacy Streamlit 参考入口保留。

## 主流程

1. 用户在 `frontend/` Next.js 页面上传 PDF。
2. PDF 保存到 `uploads/`。
3. `api_server.py` 调用 `raganything.parser.MineruParser` 和 RAG-Anything 解析文档。
4. 解析结果写入 `output/`，向量、图谱和缓存写入 `rag_storage/`。
5. `RAGAnything.process_document_complete()` 将文本、图片、表格、公式等内容写入 LightRAG。
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
QWEN_API_KEY=your_qwen_dashscope_api_key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-vl-max

LLM_MODEL=qwen-plus
LLM_BINDING_HOST=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_BINDING_API_KEY=your_qwen_dashscope_api_key
```

说明：

- `LLM_*` 用于普通文本问答、RAG 总结和知识库写入阶段的语言模型调用。
- `QWEN_*` 用于图片、表格截图、公式截图等视觉理解。
- `QWEN_API_KEY` 和 `LLM_BINDING_API_KEY` 可以填同一把 DashScope API key。
- 如果只填了 `QWEN_API_KEY`，代码也会自动把它用于文本模型调用。

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
