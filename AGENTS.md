内容要求：
1. 本项目是 RAG 系统。
2. 当前目标是从零开发 frontend/ Next.js 前端，完全替代 Streamlit UI。
3. app.py 只能作为旧功能参考，不允许把 Streamlit 代码直接翻译成 React。
4. api_server.py 是当前后端 API 入口。
5. 前端必须通过 NEXT_PUBLIC_API_BASE_URL 调用后端。
6. 不允许在前端暴露 LLM、embedding、vector DB、数据库等密钥。
7. 不允许把 RAGAnything、LightRAG、文档解析、embedding、索引构建放到前端。
8. 前端只负责 UI、交互、状态、API 调用、错误提示、引用展示。
9. 每次修改 frontend/ 后必须运行 lint 和 build。
10. 所有改动应是小步提交，避免一次性大重构。