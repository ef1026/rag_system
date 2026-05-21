# RAG-Anything（代码实测版 README）

本 README 基于当前仓库代码重建，覆盖了系统功能、启动方式、配置项、目录语义和已知限制，适合直接用于开发与部署。

## 1. 这套代码到底做什么

RAG-Anything 是一个多模态文档 RAG 框架，核心目标是把“文档解析 + 文本入库 + 多模态理解 + 知识图谱增强 + 问答检索”串成一条完整链路。

它支持的内容类型包括：
- 文本
- 图片
- 表格
- 公式
- 其他自定义类型（走通用处理器）

核心入口类是 [raganything/raganything.py](raganything/raganything.py) 里的 `RAGAnything`。

---

## 2. 我通读代码后的功能结论

### 2.1 核心能力

1. 文档解析（支持多解析器）
- 代码位置：[raganything/parser.py](raganything/parser.py)
- `MineruParser`：默认解析器（PDF、图片，Office 通过 LibreOffice 转 PDF）
- `DoclingParser`：可选解析器（PDF、Office、HTML）

2. 文本与多模态分离入库
- 代码位置：[raganything/processor.py](raganything/processor.py)、[raganything/utils.py](raganything/utils.py)
- 先抽出纯文本送入 LightRAG，再对图片/表格/公式分别做分析与实体关系抽取

3. 多模态专用处理器
- 代码位置：[raganything/modalprocessors.py](raganything/modalprocessors.py)
- `ImageModalProcessor`、`TableModalProcessor`、`EquationModalProcessor`、`GenericModalProcessor`

4. 上下文感知多模态分析
- 代码位置：[raganything/modalprocessors.py](raganything/modalprocessors.py)、[raganything/config.py](raganything/config.py)
- 会按页或按块抽取周边文本作为上下文，增强图/表/公式描述质量

5. 检索与查询
- 代码位置：[raganything/query.py](raganything/query.py)
- `aquery`：纯文本查询
- `aquery_with_multimodal`：查询时额外传入图片/表格/公式内容
- `aquery_vlm_enhanced`：把检索上下文中的图片路径替换为 base64，直接给 VLM 做图文联合回答

6. 批处理
- 代码位置：[raganything/batch.py](raganything/batch.py)、[raganything/batch_parser.py](raganything/batch_parser.py)
- 支持目录递归、并行 worker、dry-run 预检查、批量解析后入 RAG

7. 直接插入内容列表（跳过解析）
- 代码位置：[raganything/processor.py](raganything/processor.py)
- `insert_content_list` 允许你把外部系统产出的结构化内容直接接入

8. 增强 Markdown 转 PDF
- 代码位置：[raganything/enhanced_markdown.py](raganything/enhanced_markdown.py)
- 支持 WeasyPrint / Pandoc 后端、代码高亮、表格样式、目录等

### 2.2 提供的应用与脚本

1. Web UI（Streamlit）
- 入口：[app.py](app.py)
- 功能：上传 PDF、问答、测验、学习画像（增强模块存在时）
- 已修复：`streamlit` 别名导入问题，当前可用 `st` 正常启动

2. 示例脚本
- 基础流程示例：[examples/raganything_example.py](examples/raganything_example.py)
- 批处理示例：[examples/batch_processing_example.py](examples/batch_processing_example.py)
- Dry-run 示例：[examples/batch_dry_run_example.py](examples/batch_dry_run_example.py)
- LM Studio 集成示例：[examples/lmstudio_integration_example.py](examples/lmstudio_integration_example.py)
- 直接内容列表插入示例：[examples/insert_content_list_example.py](examples/insert_content_list_example.py)
- 多模态处理器示例：[examples/modalprocessors_example.py](examples/modalprocessors_example.py)
- 格式测试脚本：[examples/image_format_test.py](examples/image_format_test.py)、[examples/office_document_test.py](examples/office_document_test.py)、[examples/text_format_test.py](examples/text_format_test.py)
- 增强 Markdown 示例：[examples/enhanced_markdown_example.py](examples/enhanced_markdown_example.py)

---

## 3. 代码目录说明（重点）

- [raganything](raganything): 真实核心源码
- [examples](examples): 用法示例
- [docs](docs): 专项功能文档
- [app.py](app.py): Streamlit 界面入口
- [scripts/create_tiktoken_cache.py](scripts/create_tiktoken_cache.py): 离线缓存 tiktoken
- [rag_storage](rag_storage): LightRAG 存储与缓存目录（向量、图谱、KV）
- [output](output): 解析输出目录（中间结果/markdown/images）
- [uploads](uploads): Web UI 上传文件目录

下面这些通常是构建产物或缓存，不是主源码：
- [build](build)
- [raganything.egg-info](raganything.egg-info)

---

## 4. 安装与依赖

## 4.1 Python 版本

- 推荐 Python >= 3.10（见 [pyproject.toml](pyproject.toml)）

## 4.2 安装方式

### 方式 A：开发模式安装（推荐）

```bash
cd /Users/yuechen/Downloads/RAG-Anything
pip install -e .
```

### 方式 B：带可选能力安装

```bash
pip install -e '.[all]'
```

### 可选依赖分组（来自 [pyproject.toml](pyproject.toml)）
- `image`: Pillow（处理 BMP/TIFF/GIF/WebP 转换）
- `text`: reportlab（TXT/MD 转 PDF）
- `markdown`: markdown + weasyprint + pygments
- `all`: 全部可选项

### 额外建议安装（运行 UI / 部分示例时）

```bash
pip install streamlit sentence-transformers python-dotenv openai
```

---

## 5. 环境变量配置

先复制模板：

```bash
cp env.example .env
```

参考配置文件：
- [env.example](env.example)
- [rag.env](rag.env)

建议最小必填（跑示例和问答）：
- `LLM_BINDING_API_KEY`
- `LLM_BINDING_HOST`

常用可选：
- `PARSER`（`mineru` 或 `docling`）
- `PARSE_METHOD`（`auto` / `ocr` / `txt`）
- `OUTPUT_DIR`（默认 `./output`）
- `WORKING_DIR`（默认 `./rag_storage`）
- `QWEN_API_KEY`、`QWEN_BASE_URL`（视觉模型）
- `CONTEXT_WINDOW`、`CONTEXT_MODE`、`MAX_CONTEXT_TOKENS`（上下文感知处理）

安全建议：
- 不要把真实密钥写入可提交文件。
- 若密钥已写入 [rag.env](rag.env) 或 `.env` 并被提交，请立即轮换。

---

## 6. 如何启动（按场景）

## 6.1 场景 A：作为 Python 库运行（最常用）

先准备 `.env` 和输入文档，然后运行官方示例：

```bash
python examples/raganything_example.py ./RAG_test1.pdf --working_dir ./rag_storage --output ./output --parser mineru
```

如果你不想用环境变量，也可显式传 API：

```bash
python examples/raganything_example.py ./RAG_test1.pdf --api-key YOUR_KEY --base-url https://api.deepseek.com --working_dir ./rag_storage --output ./output
```

## 6.2 场景 B：启动 Web UI（Streamlit）

```bash
streamlit run app.py
```

默认访问：
- http://localhost:8501

说明：
- [app.py](app.py) 会尝试导入增强模块（如数据库、画像、测验引擎等）。
- 若这些模块不在仓库中，会自动降级为基础模式，不影响基础问答流程。

### 6.2.1 Web UI 详细功能说明（前端重构参考）

本节专门描述当前 Web UI 的业务行为，适合作为你交给前端重做时的功能规格基线。

#### 1) 运行模式

- 增强模式（ENHANCED_MODE=True）：可用登录、用户画像、Elo、诊断测验、错题本、元认知追踪、知识图谱入库等能力。
- 基础模式（ENHANCED_MODE=False）：若增强模块导入失败，自动退化为“上传 PDF + 智能问答”的最小可用版本。
- 增强模式依赖的模块在 [app.py](app.py) 中通过 try/except 导入，缺失时不会阻塞应用启动。

#### 2) 页面结构（信息架构）

- 侧边栏（全局设置区）
    - 学习角色选择（初学者/本科生/领域专家）
    - 迷你画像卡（增强模式）
    - PDF 上传
    - 清空缓存（rag_storage/output）
    - 用户信息区（增强模式）
- 主区标签页（增强模式）
    - 智能问答
    - 诊断测验
    - 学习画像
- 主区（基础模式）
    - 仅保留智能问答容器

#### 3) 侧边栏与全局设置

- 学习角色会改变问答系统提示词风格：
    - 初学者：偏直觉化、类比化讲解
    - 本科生：定义/公式/物理意义/考点结构
    - 专家：跳过基础定义，聚焦机制与边界
- 上传 PDF 后文件写入 [uploads](uploads) 目录。
- 点击“清空知识库缓存”会删除 [rag_storage](rag_storage) 与 [output](output)（用于强制重建索引与中间产物）。

#### 4) 上传后的后台流程

- 记录当前文档名到 session state。
- 检查数据库里该文档是否已有知识图谱节点。
    - 若无：触发后台知识图谱抽取。
    - 若有：标记为已处理，避免重复抽取。
- 抽取时会尝试多种 PDF 文本读取链路（PyMuPDF / pdfplumber / pypdf 兜底），再调用 LLM 提取知识点并落库。

#### 5) 智能问答页（核心学习入口）

- 展示 AI 教师反馈卡（增强模式）。
- 展示用户当前等级（由 Elo 映射得到等级名称/图标）。
- 多文档场景下可切换“当前提问文档”；单文档自动选中。
- 支持会话消息历史展示（前端聊天气泡）。
- 发送问题时，增强模式会额外做：
    - 主题抽取与知识访问记录
    - 元认知活动记录（提问行为）
- 回答生成逻辑由 RAG 驱动，并附加以下个性化信息：
    - 学习角色对应提示词策略
    - 薄弱知识点补充提示（最多前 3 个）
    - 元认知个性化系统补充提示
- 问答结果会写入聊天历史存储（增强模式）。

#### 6) 诊断测验页（Elo + 错题本核心页）

- 测验题基于布鲁姆认知层级（L1/L2/L3）生成。
- 可选择测验范围：
    - 全部文档
    - 指定文档
- 题目生成是自适应的，综合输入包括：
    - 用户总 Elo
    - 薄弱知识点
    - 学习历史关键词
    - 错题主题（mistake_topics）
- 默认生成 3 题，并维护完整答题状态（当前题号、用户作答、已提交结果）。
- 提交答案后会发生：
    - 评估正误与错误类型
    - 按知识点更新 Elo（含新 Elo、对局数、胜场、峰值、连胜/连败）
    - 记录题目结果到测验结果存储
    - 若答错，写入错题本（question/user_answer/correct_answer/analysis/topic/error_type）
    - 若等级晋升，触发晋级提示（含可视化庆祝）
- 每题可查看解析；全部完成后展示诊断反馈摘要。

#### 7) 学习画像页

- 由画像模块渲染完整学习画像页面。
- 输入包含当前用户与当前文档名，用于展示与文档相关的掌握情况。

#### 8) 与前端重构直接相关的状态模型

关键 session state（前端重做建议保持同等语义）：

- `elo_system`
- `metacognition_tracker`
- `quiz_engine`
- `current_kg`
- `current_doc_name`
- `meta_session_id`
- `selected_chat_doc`
- `messages`
- `need_extract_kg`
- `kg_extract_file_path`
- `kg_extract_doc_name`
- `kg_processed_docs`
- `current_quiz_session`
- `quiz_current_idx`
- `quiz_answers`
- `quiz_submitted_questions`

#### 9) 数据落点（前后端联调要点）

- 文档上传目录：[uploads](uploads)
- RAG 索引与 KV 缓存：[rag_storage](rag_storage)
- 解析输出与中间文件：[output](output)
- 用户画像、Elo、测验结果、错题本等：通过数据库接口函数读写（在 [app.py](app.py) 中调用）

#### 10) 当前版本的行为备注

- 冷启动预评估函数 `show_preassessment` 已保留，但当前主流程中已取消“自动触发”；用户主要通过“诊断测验”页手动进入测验流程。
- 前端重做时，建议优先保留：
    - 多文档选择与上下文切换
    - 诊断测验整套闭环（生成 -> 作答 -> Elo 更新 -> 错题沉淀 -> 反馈）
    - 画像与元认知追踪的可视化入口

## 6.3 场景 C：只做文档解析（命令行）

```bash
python -m raganything.parser ./RAG_test1.pdf --output ./output --parser mineru --method auto
```

检查解析器安装：

```bash
python -m raganything.parser --check --parser mineru ./RAG_test1.pdf
```

## 6.4 场景 D：批处理目录文档

```bash
python -m raganything.batch_parser ./uploads --output ./output --parser mineru --workers 4 --method auto
```

仅预览将处理哪些文件（不执行解析）：

```bash
python examples/batch_dry_run_example.py ./uploads --parser mineru --recursive
```

## 6.5 场景 E：增强 Markdown 转 PDF

```bash
python -m raganything.enhanced_markdown ./docs/enhanced_markdown.md --output ./output/enhanced_markdown.pdf --method auto
```

查看可用后端：

```bash
python -m raganything.enhanced_markdown --info
```

---

## 7. 最小代码接入示例

```python
import os
import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

async def main():
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
        config=RAGAnythingConfig(working_dir="./rag_storage", parser="mineru", parse_method="auto"),
        llm_model_func=llm,
        embedding_func=embedding,
    )

    await rag.process_document_complete(file_path="./RAG_test1.pdf", output_dir="./output")
    answer = await rag.aquery("这份文档讲了什么？", mode="hybrid")
    print(answer)

asyncio.run(main())
```

更多示例见：
- [examples/raganything_example.py](examples/raganything_example.py)
- [examples/insert_content_list_example.py](examples/insert_content_list_example.py)
- [examples/lmstudio_integration_example.py](examples/lmstudio_integration_example.py)

---

## 8. 关键 API 速览

- 主类：`RAGAnything`（[raganything/raganything.py](raganything/raganything.py)）
- 文档全流程：`process_document_complete`（[raganything/processor.py](raganything/processor.py)）
- 内容列表直插：`insert_content_list`（[raganything/processor.py](raganything/processor.py)）
- 文本查询：`aquery`（[raganything/query.py](raganything/query.py)）
- 多模态查询：`aquery_with_multimodal`（[raganything/query.py](raganything/query.py)）
- VLM 增强查询：`aquery_vlm_enhanced`（[raganything/query.py](raganything/query.py)）
- 批处理：`BatchParser.process_batch`（[raganything/batch_parser.py](raganything/batch_parser.py)）

---

## 9. 输出与缓存目录语义

- [output](output): 解析器输出目录
  - 通常包含 `*_content_list.json`、`*.md`、中间 JSON、图片文件夹
- [rag_storage](rag_storage): LightRAG 持久化目录
  - 包含向量库（`vdb_*.json`）、KV 存储（`kv_store_*.json`）、图谱（`graph_chunk_entity_relation.graphml`）
- [uploads](uploads): Web UI 上传目录

---

## 10. 离线部署（必须看）

离线场景要解决 `tiktoken` 首次联网下载问题：

1. 先缓存 tokenizer 文件

```bash
python scripts/create_tiktoken_cache.py
```

2. 在 `.env` 中设置：

```bash
TIKTOKEN_CACHE_DIR=./tiktoken_cache
```

详细见：[docs/offline_setup.md](docs/offline_setup.md)

---

## 11. 已知限制与注意事项

1. MinerU 处理 Office 文件依赖 LibreOffice
- 代码见 [raganything/parser.py](raganything/parser.py)
- 没有 LibreOffice 时，`.doc/.ppt/.xls` 这类会失败

2. 部分图片格式依赖 Pillow 转换
- 如 BMP/TIFF/GIF/WebP 需要 Pillow

3. TXT/MD 转 PDF 依赖 reportlab
- 缺失时文本解析会失败

4. 若使用 Docling，请确保系统存在 `docling` 命令
- 代码见 [raganything/parser.py](raganything/parser.py)

5. Web UI 的增强功能依赖额外模块
- [app.py](app.py) 中 `database/auth/student_profile/...` 不在本仓库
- 缺失时会自动基础模式运行

---

## 12. 相关文档

- 批处理：[docs/batch_processing.md](docs/batch_processing.md)
- 上下文感知处理：[docs/context_aware_processing.md](docs/context_aware_processing.md)
- 增强 Markdown：[docs/enhanced_markdown.md](docs/enhanced_markdown.md)
- 离线部署：[docs/offline_setup.md](docs/offline_setup.md)

---

## 13. 许可证

MIT，见 [LICENSE](LICENSE)
