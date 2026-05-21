"""
新工科 AI 助教系统 - 增强版
核心特性：
1. 知识图谱驱动的认知状态追踪
2. 布鲁姆分类法分级测验（L1/L2/L3）
3. Elo 等级分系统（游戏化激励）
4. 元认知分析（专注度/探索度/深度）
5. 闭环学习：画像指导测试，测试更新画像
"""

import streamlit as st
import sys
import os
import asyncio
import re
import json
from pathlib import Path
from datetime import datetime

# === 1. Windows 异步策略补丁 ===
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 解决 Streamlit 中 "event loop is already running" 问题
import nest_asyncio
nest_asyncio.apply()

# === 2. 基础环境配置 ===
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加模块搜索路径（支持从根目录或子目录运行）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMODULE_DIR = os.path.join(BASE_DIR, "RAG-Anything")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SUBMODULE_DIR)
sys.path.append(os.getcwd())

from sentence_transformers import SentenceTransformer
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig
from dotenv import load_dotenv

# 尝试从子模块导入新功能
ENHANCED_MODE = False
try:
    print("正在加载 database...")
    from database import (
        init_database, get_user_by_id, needs_preassessment, mark_preassessment_done,
        save_chat_history, save_quiz_result, get_user_weak_knowledge,
        save_knowledge_graph, get_kg_nodes, update_user_knowledge_state,
        get_user_overall_elo, record_knowledge_access, save_metacognition_profile,
        add_mistake, get_all_user_knowledge_states,
        get_user_learning_history_keywords, get_available_documents, get_document_content_for_quiz
    )
    print("正在加载 auth...")
    from auth import show_login_page, is_logged_in, get_current_user, get_current_user_id, show_user_info_sidebar, logout_user, start_study_session
    print("正在加载 student_profile...")
    from student_profile import show_profile_page, show_mini_profile_card, generate_smart_feedback
    print("正在加载 knowledge_graph...")
    from knowledge_graph import extract_knowledge_graph_from_document, KnowledgeGraph
    print("正在加载 elo_rating...")
    from elo_rating import EloSystem, EloTier, format_elo_display
    print("正在加载 quiz_engine...")
    from quiz_engine import DiagnosticQuizEngine, BloomLevel, QuizPromptTemplates
    print("正在加载 metacognition...")
    from metacognition import MetacognitionTracker
    print("正在加载 analytics...")
    from analytics import extract_topic_from_question
    ENHANCED_MODE = True
    print("✅ 所有增强模块加载成功！")
except ImportError as e:
    import traceback
    print(f"⚠️ 增强模块加载失败: {e}")
    print("详细错误信息:")
    traceback.print_exc()
    print("将使用基础模式运行")
    ENHANCED_MODE = False

load_dotenv(dotenv_path=".env", override=False)
load_dotenv(dotenv_path="rag.env", override=False)

st.set_page_config(page_title="新工科 AI 助教", layout="wide", page_icon="🎓")

# === 3. 永久事件循环管理 ===
if "loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    st.session_state.loop = loop
else:
    asyncio.set_event_loop(st.session_state.loop)

# === 4. 初始化全局系统组件 ===
if ENHANCED_MODE:
    if "elo_system" not in st.session_state:
        st.session_state.elo_system = EloSystem()
    if "metacognition_tracker" not in st.session_state:
        st.session_state.metacognition_tracker = MetacognitionTracker()
    if "quiz_engine" not in st.session_state:
        st.session_state.quiz_engine = None
    if "current_kg" not in st.session_state:
        st.session_state.current_kg = None
    if "current_doc_name" not in st.session_state:
        st.session_state.current_doc_name = None
    if "meta_session_id" not in st.session_state:
        st.session_state.meta_session_id = None

# === 5. 辅助函数 ===
def process_math_format(text):
    """清洗数学公式格式"""
    if not isinstance(text, str):
        return str(text)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    def remove_code_ticks(match):
        content = match.group(1)
        if '\\' in content or '=' in content or '^' in content:
            return f"${content.strip('$')}$"
        return match.group(0)
    text = re.sub(r'`(.*?)`', remove_code_ticks, text)
    return text


# === 6. 模型加载 ===
@st.cache_resource
def load_local_model_only():
    print("正在加载本地 BGE-Small 中文模型...")
    return SentenceTransformer('BAAI/bge-small-zh-v1.5')


# === 7. LLM 调用函数 ===
def get_llm_functions():
    """获取 LLM 调用函数"""
    api_key = os.getenv("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST")
    
    async def safe_deepseek_call(prompt, system_prompt="You are a helpful AI tutor.", 
                                  history_messages=[], **kwargs):
        kwargs.pop('response_format', None)
        kwargs.pop('keyword_extraction', None)
        kwargs.pop('image_data', None)
        
        if "messages" in kwargs:
            clean_msgs = []
            for msg in kwargs["messages"]:
                content = msg.get("content")
                if isinstance(content, list):
                    text_content = "".join([
                        item.get("text", "") for item in content 
                        if isinstance(item, dict) and item.get("type") == "text"
                    ])
                    clean_msgs.append({"role": msg["role"], "content": text_content})
                else:
                    clean_msgs.append(msg)
            kwargs["messages"] = clean_msgs

        response = await openai_complete_if_cache(
            "deepseek-chat", prompt, system_prompt=system_prompt,
            history_messages=history_messages, api_key=api_key, base_url=base_url, **kwargs
        )
        raw_text = response.replace("```json", "").replace("```", "").strip() if isinstance(response, str) else str(response)
        return process_math_format(raw_text)

    async def qwen_vision_func(prompt, system_prompt="You are a helpful AI assistant.", 
                                image_data=None, **kwargs):
        qwen_api_key = os.getenv("QWEN_API_KEY")
        qwen_base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        if not qwen_api_key:
            kwargs.pop('image_data', None)
            kwargs.pop('keyword_extraction', None)
            return await safe_deepseek_call(prompt, system_prompt, **kwargs)

        messages = [{"role": "system", "content": system_prompt}]
        user_content = [{"type": "text", "text": prompt}]
        if image_data:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            })
        messages.append({"role": "user", "content": user_content})

        kwargs.pop('image_data', None)
        kwargs.pop('response_format', None)
        kwargs.pop('keyword_extraction', None)

        return await openai_complete_if_cache(
            "qwen-vl-max", None, system_prompt=None,
            history_messages=messages, api_key=qwen_api_key, base_url=qwen_base_url, **kwargs
        )

    return safe_deepseek_call, qwen_vision_func


# === 8. 核心 RAG 业务逻辑 ===
async def run_rag(file_path, query, level, user_id=None):
    """执行 RAG 查询"""
    api_key = os.getenv("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST")
    
    local_model = load_local_model_only()

    async def _current_loop_embed(texts):
        return await asyncio.to_thread(lambda: local_model.encode(texts))

    embedding_func = EmbeddingFunc(
        embedding_dim=512,
        max_token_size=512,
        func=_current_loop_embed
    )

    safe_deepseek_call, qwen_vision_func = get_llm_functions()

    # 构建 Prompt
    query_suffix = ""
    if "初学者" in level:
        query_suffix = """\n\n【指令：直觉科普模式】
        1. 🚫 严禁使用晦涩专业术语，必须用大白话。
        2. ✅ 核心：使用生活中的类比（如把电路比作水管）。
        3. 语气：幽默风趣的科普博主。
        """
    elif "专家" in level:
        query_suffix = """\n\n【指令：深度研讨模式】
        1. ⚠️ 跳过基础定义，假设用户是同行。
        2. ✅ 核心：切入问题本质、底层机制、局限性。
        3. 语气：极度简练、学术、高冷。
        """
    else:
        query_suffix = """\n\n【指令：标准教学模式】
        1. 目标：帮助通过期末考试。
        2. ✅ 结构：定义 -> 公式 -> 物理意义 -> 考点。
        3. 语气：耐心的大学助教。
        """
    
    # 【增强】注入用户画像
    if ENHANCED_MODE and user_id:
        try:
            weak_knowledge = get_user_weak_knowledge(user_id)
            if weak_knowledge:
                weak_names = [w.get('knowledge_name', w['knowledge_id']) for w in weak_knowledge[:3]]
                query_suffix += f"\n\n【用户画像】用户在以下知识点较弱：{', '.join(weak_names)}。如果问题涉及这些内容，请详细展开解释。"
            
            tracker = st.session_state.metacognition_tracker
            meta_addon = tracker.get_personalized_system_prompt_addon(user_id)
            if meta_addon:
                query_suffix += meta_addon
        except:
            pass

    rag = RAGAnything(
        config=RAGAnythingConfig(working_dir="./rag_storage", parser="mineru", parse_method="auto"),
        llm_model_func=safe_deepseek_call,
        vision_model_func=qwen_vision_func,
        embedding_func=embedding_func
    )

    if file_path:
        await rag.process_document_complete(file_path=file_path, output_dir="./output", parse_method="auto")

    return await rag.aquery(query + query_suffix, mode="naive")


# === 9. 冷启动摸底测试 ===
async def generate_preassessment_questions(context: str, topic: str = "文档内容") -> list:
    """根据文档内容动态生成冷启动诊断题目"""
    safe_deepseek_call, _ = get_llm_functions()
    
    prompt = f"""【出题任务】
基于以下文档内容，生成 3 道诊断测试题（覆盖三个布鲁姆认知层级）。

【文档内容】
{context[:8000]}  

【输出要求】
请生成 3 道选择题，分别对应：
1. L1 记忆/理解层级：考察基本概念、定义、术语
2. L2 应用层级：考察公式应用、计算、实际运用
3. L3 分析/评估层级：考察综合分析、推理判断、多因素考量

【输出格式】（严格 JSON 数组）
[
    {{
        "id": "pre_1",
        "bloom_level": 1,
        "question": "【L1 概念题】题目内容...",
        "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
        "correct": "A",
        "topic": "知识点名称"
    }},
    {{
        "id": "pre_2",
        "bloom_level": 2,
        "question": "【L2 应用题】题目内容...",
        "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
        "correct": "B",
        "topic": "知识点名称"
    }},
    {{
        "id": "pre_3",
        "bloom_level": 3,
        "question": "【L3 分析题】题目内容...",
        "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
        "correct": "C",
        "topic": "知识点名称"
    }}
]

【注意事项】
- 题目必须紧扣文档内容，不要凭空捏造
- 选项设计要合理，干扰项应是常见错误理解
- 每道题明确标注考察的知识点
- 直接输出 JSON 数组，不要添加其他内容"""

    system_prompt = "你是一位专业的教育评估专家，擅长根据教材内容设计诊断性测试题。请严格按 JSON 格式输出。"
    
    try:
        response = await safe_deepseek_call(prompt, system_prompt=system_prompt)
        
        raw_json = str(response).strip()
        if "```json" in raw_json:
            raw_json = raw_json.split("```json")[1].split("```")[0]
        elif "```" in raw_json:
            raw_json = raw_json.split("```")[1].split("```")[0]
        
        questions = json.loads(raw_json)
        return questions
        
    except Exception as e:
        print(f"❌ 生成诊断题目失败: {e}")
        return None


def show_preassessment(user_id: int):
    """显示冷启动摸底测试（基于用户已上传的文档）"""
    st.title("📋 学前诊断测试")
    
    # 初始化状态
    if "preassessment_file_processed" not in st.session_state:
        st.session_state.preassessment_file_processed = False
    if "preassessment_context" not in st.session_state:
        st.session_state.preassessment_context = None
    if "preassessment_questions" not in st.session_state:
        st.session_state.preassessment_questions = None
    if "preassessment_answers" not in st.session_state:
        st.session_state.preassessment_answers = {}
    
    # 获取已上传的文件路径
    file_path = st.session_state.get("preassessment_file_path")
    
    # 如果没有文件路径，说明流程有问题，跳过测试
    if not file_path or not os.path.exists(file_path):
        st.warning("未检测到上传的文档，跳过诊断测试。")
        mark_preassessment_done(user_id)
        st.session_state.preassessment_in_progress = False
        update_user_knowledge_state(
            user_id=user_id,
            knowledge_id="通用知识",
            knowledge_name="通用知识",
            elo_rating=1000,
            games_played=0,
            wins=0,
            peak_elo=1000,
            streak=0
        )
        st.rerun()
        return
    
    # 如果还没有处理文件，解析并生成题目
    if not st.session_state.preassessment_file_processed:
        st.markdown(f"""
        欢迎使用 AI 助教系统！检测到您上传了学习材料：**{os.path.basename(file_path)}**
        
        系统正在根据文档内容**自动生成个性化的诊断题目**，评估您对该内容的掌握程度。
        """)
        
        with st.spinner("📖 正在解析文档并生成诊断题目..."):
            # 提取 PDF 文本内容
            try:
                text_content = ""
                # 尝试多种 PDF 解析库
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(file_path)
                    for page in doc:
                        text_content += page.get_text()
                    doc.close()
                except ImportError:
                    try:
                        import pdfplumber
                        with pdfplumber.open(file_path) as pdf:
                            for page in pdf.pages:
                                text_content += (page.extract_text() or "") + "\n"
                    except ImportError:
                        # 使用 pypdf 作为最后的备选
                        try:
                            from pypdf import PdfReader
                            reader = PdfReader(file_path)
                            for page in reader.pages:
                                text_content += (page.extract_text() or "") + "\n"
                        except ImportError:
                            st.error("缺少 PDF 解析库。请安装: pip install PyMuPDF 或 pip install pdfplumber 或 pip install pypdf")
                            if st.button("跳过诊断测试"):
                                mark_preassessment_done(user_id)
                                st.session_state.preassessment_in_progress = False
                                st.rerun()
                            return
                
                if len(text_content.strip()) < 100:
                    st.error("文档内容过少，跳过诊断测试。")
                    mark_preassessment_done(user_id)
                    st.session_state.preassessment_in_progress = False
                    st.rerun()
                    return
                
                # 保存上下文
                st.session_state.preassessment_context = text_content
                st.session_state.preassessment_doc_name = os.path.basename(file_path)
                
                # 生成诊断题目
                loop = st.session_state.loop
                questions = loop.run_until_complete(
                    generate_preassessment_questions(text_content, os.path.basename(file_path))
                )
                
                if questions and len(questions) >= 3:
                    st.session_state.preassessment_questions = questions
                    st.session_state.preassessment_file_processed = True
                    st.rerun()
                else:
                    st.error("生成题目失败，跳过诊断测试。")
                    mark_preassessment_done(user_id)
                    st.session_state.preassessment_in_progress = False
                    st.rerun()
                    
            except Exception as e:
                st.error(f"文档处理失败: {e}")
                import traceback
                st.code(traceback.format_exc())
                # 失败时跳过测试
                if st.button("跳过诊断测试"):
                    mark_preassessment_done(user_id)
                    st.session_state.preassessment_in_progress = False
                    st.rerun()
        return
    
    # 第二步：显示生成的题目
    st.markdown(f"""
    📄 **基于文档**: {st.session_state.get('preassessment_doc_name', '已上传的文档')}
    
    以下是系统根据您的学习材料生成的 **3 道诊断题**，请认真作答。
    """)
    
    questions = st.session_state.preassessment_questions
    
    # 显示题目
    for i, q in enumerate(questions):
        st.markdown(f"### 第 {i+1} 题")
        st.markdown(q["question"])
        
        answer = st.radio(
            f"请选择答案：",
            q["options"],
            key=f"pre_q_{i}",
            index=None
        )
        
        if answer:
            st.session_state.preassessment_answers[q["id"]] = answer[0]  # 取首字母
        
        st.markdown("---")
    
    # 提交按钮
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("📤 提交诊断测试", type="primary", use_container_width=True):
            answers = st.session_state.preassessment_answers
            
            if len(answers) < len(questions):
                st.error("请完成所有题目后再提交！")
                return
            
            # 评估结果
            correct_count = 0
            for q in questions:
                is_correct = answers.get(q["id"]) == q["correct"]
                if is_correct:
                    correct_count += 1
            
            # 确定初始等级
            if correct_count == 3:
                estimated_level, initial_elo = "进阶", 1300
            elif correct_count >= 2:
                estimated_level, initial_elo = "中等", 1100
            elif correct_count == 1:
                estimated_level, initial_elo = "基础", 900
            else:
                estimated_level, initial_elo = "入门", 800
            
            # 初始化用户各知识点的 Elo
            for q in questions:
                update_user_knowledge_state(
                    user_id=user_id,
                    knowledge_id=q["topic"],
                    knowledge_name=q["topic"],
                    elo_rating=initial_elo,
                    games_played=1,
                    wins=1 if answers.get(q["id"]) == q["correct"] else 0,
                    peak_elo=initial_elo,
                    streak=1 if answers.get(q["id"]) == q["correct"] else -1
                )
            
            # 标记完成
            mark_preassessment_done(user_id)
            
            # 显示结果
            st.success(f"🎉 诊断完成！您答对了 {correct_count}/3 题")
            st.info(f"📊 系统评估您的初始水平为：**{estimated_level}** (Elo: {initial_elo})")
            st.markdown("系统将根据此结果为您个性化调整学习内容。")
            
            # 清理临时状态
            for key in ["preassessment_questions", "preassessment_answers", 
                       "preassessment_file_processed", "preassessment_context", 
                       "preassessment_doc_name", "preassessment_in_progress",
                       "preassessment_file_path"]:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.rerun()
    
    with col2:
        if st.button("⏭️ 跳过测试", help="跳过诊断测试，使用默认设置"):
            mark_preassessment_done(user_id)
            # 设置默认的初始 Elo
            update_user_knowledge_state(
                user_id=user_id,
                knowledge_id="通用知识",
                knowledge_name="通用知识",
                elo_rating=1000,
                games_played=0,
                wins=0,
                peak_elo=1000,
                streak=0
            )
            # 清理临时状态
            for key in ["preassessment_questions", "preassessment_answers", 
                       "preassessment_file_processed", "preassessment_context",
                       "preassessment_doc_name", "preassessment_in_progress",
                       "preassessment_file_path"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


# === 10. 自适应测验生成 ===
async def generate_adaptive_quiz(user_id: int, context: str, user_level: str, 
                                  selected_doc: str = None, learning_history: dict = None):
    """
    生成自适应测验
    
    参数:
        user_id: 用户ID
        context: 文档上下文内容
        user_level: 用户水平（本科生/研究生等）
        selected_doc: 用户选择的目标文档名称
        learning_history: 用户学习历史（包含错题、问题关键词等）
    """
    safe_deepseek_call, _ = get_llm_functions()
    
    if st.session_state.quiz_engine is None:
        st.session_state.quiz_engine = DiagnosticQuizEngine(safe_deepseek_call)
    
    engine = st.session_state.quiz_engine
    weak_points = [w.get('knowledge_name', w['knowledge_id']) for w in get_user_weak_knowledge(user_id)]
    user_elo = get_user_overall_elo(user_id)
    
    # 如果有学习历史中的错题主题，也加入薄弱点
    if learning_history and learning_history.get('mistake_topics'):
        for topic in learning_history['mistake_topics']:
            if topic not in weak_points:
                weak_points.append(topic)
    
    topic = weak_points[0] if weak_points else (selected_doc or "当前学习内容")
    
    session = await engine.generate_adaptive_quiz(
        user_id=user_id, 
        topic=topic, 
        context=context, 
        user_level=user_level,
        user_elo=user_elo, 
        num_questions=3, 
        weak_points=weak_points,
        learning_history=learning_history,
        selected_doc=selected_doc
    )
    return session


# === 11. 主界面 ===
def main():
    if ENHANCED_MODE:
        init_database()
        
        if not is_logged_in():
            show_login_page()
            return
        
        user = get_current_user()
        user_id = get_current_user_id()
        
        # 【已移除自动冷启动测试】用户可以通过"诊断测验"标签页手动选择测验
        # 不再自动触发预评估流程
        
        if st.session_state.meta_session_id is None:
            tracker = st.session_state.metacognition_tracker
            st.session_state.meta_session_id = tracker.start_session(user_id)
    else:
        user_id = None
    
    # === 侧边栏 ===
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=60)
        st.title("⚙️ 学习设置")
        
        user_level = st.radio(
            "我是谁？",
            ["👶 初学者 (通俗易懂)", "👨‍🎓 本科生 (专业推导)", "👨‍🔬 领域专家 (深度研讨)"],
            index=1
        )
        
        st.divider()
        
        if ENHANCED_MODE:
            show_mini_profile_card(user_id)
            st.divider()
        
        st.header("📂 知识库")
        uploaded_file = st.file_uploader("上传教材 (PDF)", type=["pdf"])
        
        if st.button("🗑️ 清空知识库缓存"):
            import shutil
            if os.path.exists("./rag_storage"):
                shutil.rmtree("./rag_storage")
            if os.path.exists("./output"):
                shutil.rmtree("./output")
            if ENHANCED_MODE:
                st.session_state.current_kg = None
                st.session_state.current_doc_name = None
            st.success("缓存已清空！")
        
        if ENHANCED_MODE:
            st.divider()
            show_user_info_sidebar()
    
    # === 文件处理 ===
    file_path = None
    if uploaded_file:
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", uploaded_file.name)
        doc_name = uploaded_file.name
        
        # 检查是否是新文件
        is_new_file = not os.path.exists(file_path)
        if is_new_file:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # 更新当前文档名称
        if ENHANCED_MODE:
            st.session_state.current_doc_name = doc_name
        
        # 【已移除自动冷启动测试】新用户不再自动触发预评估，改为手动选择测验
        # 用户可以通过"诊断测验"标签页手动选择文档并生成测验
        
        # 【新增】检查是否需要提取知识图谱（无论是新文件还是已存在的文件）
        if ENHANCED_MODE:
            # 避免重复触发：检查是否已经在处理中或已处理过该文档
            already_processing = st.session_state.get("need_extract_kg", False)
            already_processed_doc = st.session_state.get("kg_processed_docs", set())
            
            if not already_processing and doc_name not in already_processed_doc:
                # 检查数据库中是否已有该文档的知识图谱
                existing_nodes = get_kg_nodes(doc_name)
                if not existing_nodes:
                    # 需要提取知识图谱
                    st.session_state.need_extract_kg = True
                    st.session_state.kg_extract_file_path = file_path
                    st.session_state.kg_extract_doc_name = doc_name
                else:
                    # 已有知识图谱，标记为已处理
                    if "kg_processed_docs" not in st.session_state:
                        st.session_state.kg_processed_docs = set()
                    st.session_state.kg_processed_docs.add(doc_name)
            
    elif os.path.exists("uploads") and len(os.listdir("uploads")) > 0:
        file_path = os.path.join("uploads", os.listdir("uploads")[0])
        if ENHANCED_MODE:
            st.session_state.current_doc_name = os.path.basename(file_path)
    
    # === 知识图谱提取（后台处理） ===
    if ENHANCED_MODE and st.session_state.get("need_extract_kg", False):
        kg_file_path = st.session_state.get("kg_extract_file_path")
        kg_doc_name = st.session_state.get("kg_extract_doc_name")
        
        if kg_file_path and kg_doc_name:
            with st.spinner("🧠 正在构建知识图谱..."):
                try:
                    # 提取 PDF 文本
                    text_content = ""
                    try:
                        import fitz
                        doc = fitz.open(kg_file_path)
                        for page in doc:
                            text_content += page.get_text()
                        doc.close()
                    except ImportError:
                        try:
                            import pdfplumber
                            with pdfplumber.open(kg_file_path) as pdf:
                                for page in pdf.pages:
                                    text_content += (page.extract_text() or "") + "\n"
                        except ImportError:
                            try:
                                from pypdf import PdfReader
                                reader = PdfReader(kg_file_path)
                                for page in reader.pages:
                                    text_content += (page.extract_text() or "") + "\n"
                            except:
                                text_content = ""
                    
                    if len(text_content) > 100:
                        # 调用 LLM 提取知识图谱
                        safe_deepseek_call, _ = get_llm_functions()
                        loop = st.session_state.loop
                        kg = loop.run_until_complete(
                            extract_knowledge_graph_from_document(text_content[:8000], safe_deepseek_call)
                        )
                        
                        if kg and kg.nodes:
                            # 保存到数据库
                            kg_data = kg.to_dict()
                            saved_count = save_knowledge_graph(kg_doc_name, kg_data)
                            st.session_state.current_kg = kg
                            st.success(f"✅ 知识图谱构建完成！提取了 {saved_count} 个知识点")
                        else:
                            st.warning("知识图谱提取结果为空")
                    else:
                        st.warning("文档内容过少，无法提取知识图谱")
                        
                except Exception as e:
                    st.error(f"知识图谱提取失败: {e}")
                    import traceback
                    print(traceback.format_exc())
            
            # 清理状态
            st.session_state.need_extract_kg = False
            st.session_state.kg_extract_file_path = None
            st.session_state.kg_extract_doc_name = None
    
    # === 主页面 ===
    if ENHANCED_MODE:
        tab_chat, tab_quiz, tab_profile = st.tabs(["💬 智能问答", "📝 诊断测验", "📊 学习画像"])
    else:
        tab_chat = st.container()
    
    # === 智能问答标签页 ===
    with tab_chat:
        st.title("🎓 新工科 AI 助教系统")
        
        if ENHANCED_MODE:
            feedback = generate_smart_feedback(user_id)
            if feedback:
                with st.expander("💡 AI 助教的话", expanded=True):
                    st.markdown(feedback)
            
            overall_elo = get_user_overall_elo(user_id)
            tier = EloTier.from_elo(overall_elo)
            st.caption(f"当前模式：{user_level} | 等级：{tier.icon} {tier.name_cn} | 引擎：DeepSeek-V3")
        else:
            st.caption(f"当前模式：{user_level} | 引擎：DeepSeek-V3 + BGE-Small")
        
        # === 文档选择区域（新用户首次需要选择PDF才能提问）===
        if ENHANCED_MODE:
            chat_available_docs = get_available_documents()
        else:
            chat_available_docs = []
        
        # 初始化选中的文档
        if "selected_chat_doc" not in st.session_state:
            st.session_state.selected_chat_doc = None
        
        # 获取当前有效的文档路径
        effective_file_path = file_path  # 默认使用上传的文件
        
        if ENHANCED_MODE:
            # 如果有多个文档可选，显示选择器
            if len(chat_available_docs) > 1:
                st.markdown("### 📚 选择要提问的文档")
                chat_doc_options = [f"📄 {doc}" for doc in chat_available_docs]
                
                # 设置默认选中项
                default_idx = 0
                if st.session_state.selected_chat_doc:
                    try:
                        default_idx = chat_available_docs.index(st.session_state.selected_chat_doc)
                    except ValueError:
                        default_idx = 0
                
                selected_chat_option = st.selectbox(
                    "选择文档：",
                    chat_doc_options,
                    index=default_idx,
                    key="chat_doc_selector",
                    help="选择要针对哪个文档进行提问"
                )
                
                selected_chat_doc = selected_chat_option.replace("📄 ", "")
                st.session_state.selected_chat_doc = selected_chat_doc
                
                # 更新有效文件路径
                effective_file_path = os.path.join("uploads", selected_chat_doc)
                if not os.path.exists(effective_file_path):
                    effective_file_path = file_path
                
                st.markdown("---")
            
            elif len(chat_available_docs) == 1:
                # 只有一个文档，自动选中
                st.session_state.selected_chat_doc = chat_available_docs[0]
                effective_file_path = os.path.join("uploads", chat_available_docs[0])
                st.info(f"📄 当前文档：{chat_available_docs[0]}")
            
            elif len(chat_available_docs) == 0 and not file_path:
                # 没有文档，提示用户上传
                st.warning("👋 欢迎使用 AI 助教！请先在左侧上传 PDF 教材，然后选择文档进行提问。")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # 检查是否可以提问（必须有文档）
        can_ask = effective_file_path is not None or os.path.exists("./rag_storage")
        
        if prompt := st.chat_input("请输入你的问题...", disabled=not can_ask):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            if ENHANCED_MODE:
                topic = extract_topic_from_question(prompt)
                tracker = st.session_state.metacognition_tracker
                tracker.record_activity(user_id, topic=topic, is_question=True)
                record_knowledge_access(user_id, topic, topic)
            
            st.rerun()
        
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            last_user_query = st.session_state.messages[-1]["content"]
            
            with st.chat_message("assistant"):
                with st.spinner("🧠 DeepSeek 正在思考..."):
                    try:
                        if not effective_file_path and not os.path.exists("./rag_storage"):
                            error_msg = "请先在左侧上传 PDF 教材！"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        else:
                            loop = st.session_state.loop
                            response = loop.run_until_complete(
                                run_rag(effective_file_path, last_user_query, user_level, user_id)
                            )
                            
                            try:
                                if isinstance(response, str):
                                    final_ans = json.loads(response).get("answer", response)
                                else:
                                    final_ans = str(response)
                            except:
                                final_ans = str(response)
                            
                            final_ans = process_math_format(final_ans)
                            st.markdown(final_ans)
                            st.session_state.messages.append({"role": "assistant", "content": final_ans})
                            
                            if ENHANCED_MODE:
                                topic = extract_topic_from_question(last_user_query)
                                save_chat_history(user_id, last_user_query, final_ans, topic)
                            
                    except Exception as e:
                        st.error(f"发生错误: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
    
    # === 诊断测验标签页 ===
    if ENHANCED_MODE:
        with tab_quiz:
            st.title("📝 诊断式测验")
            st.markdown("基于布鲁姆认知分类法的智能测验 - 每次生成独特的个性化题目")
            
            # 显示薄弱知识点提示
            weak_knowledge = get_user_weak_knowledge(user_id)
            if weak_knowledge:
                st.warning(f"⚠️ 检测到 {len(weak_knowledge)} 个薄弱知识点，系统将针对性出题")
            
            # === 文档选择区域 ===
            st.markdown("### 📚 选择测验范围")
            
            # 获取可用文档列表
            available_docs = get_available_documents()
            
            # 如果当前上传了文件，确保它在列表中
            if file_path:
                current_doc_name = os.path.basename(file_path)
                if current_doc_name not in available_docs:
                    available_docs.insert(0, current_doc_name)
            
            if available_docs:
                # 添加"全部文档"选项
                doc_options = ["📁 全部文档"] + [f"📄 {doc}" for doc in available_docs]
                
                selected_option = st.selectbox(
                    "选择要测验的文档：",
                    doc_options,
                    index=0,
                    help="选择特定文档将只针对该文档内容生成题目，选择'全部文档'则综合所有已学内容"
                )
                
                # 解析选择的文档名
                if selected_option == "📁 全部文档":
                    selected_doc = None
                else:
                    selected_doc = selected_option.replace("📄 ", "")
            else:
                st.info("💡 上传学习文档后，可以选择针对特定文档生成测验")
                selected_doc = None
            
            st.markdown("---")
            
            # === 生成测验按钮 ===
            col1, col2 = st.columns([3, 1])
            with col1:
                generate_quiz = st.button("🎯 生成自适应测验", type="primary", use_container_width=True)
            with col2:
                if st.button("🔄 重置", help="清除当前测验，重新开始"):
                    for key in ["current_quiz_session", "quiz_current_idx", 
                               "quiz_answers", "quiz_submitted_questions"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
            
            if generate_quiz:
                if not file_path and not os.path.exists("./rag_storage"):
                    st.error("请先上传学习文档！")
                else:
                    with st.spinner("🧠 正在根据您的学习历史生成个性化测验题目..."):
                        # 获取用户学习历史
                        learning_history = get_user_learning_history_keywords(user_id)
                        
                        # 获取文档内容
                        context = ""
                        if selected_doc:
                            # 获取特定文档的内容
                            context = get_document_content_for_quiz(selected_doc)
                        
                        if not context:
                            # 使用默认的文本块
                            text_chunks_path = "./rag_storage/kv_store_text_chunks.json"
                            if os.path.exists(text_chunks_path):
                                try:
                                    with open(text_chunks_path, 'r', encoding='utf-8') as f:
                                        chunks = json.load(f)
                                    context = "\n".join([
                                        c.get('content', '') for c in list(chunks.values())[:15]
                                        if isinstance(c, dict)
                                    ])
                                except:
                                    context = "工程学科基础知识"
                        
                        loop = st.session_state.loop
                        session = loop.run_until_complete(
                            generate_adaptive_quiz(
                                user_id=user_id, 
                                context=context[:8000], 
                                user_level=user_level,
                                selected_doc=selected_doc,
                                learning_history=learning_history
                            )
                        )
                        
                        if session and session.questions:
                            st.session_state.current_quiz_session = session
                            st.session_state.quiz_current_idx = 0
                            st.session_state.quiz_answers = {}
                            st.session_state.quiz_submitted_questions = {}
                            
                            # 显示生成信息
                            doc_info = f"针对《{selected_doc}》" if selected_doc else "综合所有文档"
                            st.success(f"✅ 已生成 {len(session.questions)} 道个性化题目 ({doc_info})")
                            
                            # 如果有学习历史，显示提示
                            if learning_history.get('mistake_topics'):
                                st.info(f"💡 本次测验重点关注您的薄弱领域：{', '.join(learning_history['mistake_topics'][:3])}")
                            
                            st.rerun()
                        else:
                            st.error("题目生成失败，请重试")
            
            # 显示当前测验
            if "current_quiz_session" in st.session_state:
                session = st.session_state.current_quiz_session
                current_idx = st.session_state.quiz_current_idx
                
                # 初始化已提交答案的状态
                if "quiz_submitted_questions" not in st.session_state:
                    st.session_state.quiz_submitted_questions = {}
                
                if current_idx < len(session.questions):
                    question = session.questions[current_idx]
                    question_submitted = question.id in st.session_state.quiz_submitted_questions
                    
                    st.markdown("---")
                    st.markdown(f"### 第 {current_idx + 1}/{len(session.questions)} 题")
                    
                    bloom_colors = {BloomLevel.L1_RECALL: "🟢", BloomLevel.L2_APPLY: "🟡", BloomLevel.L3_ANALYZE: "🔴"}
                    st.caption(f"难度：{bloom_colors.get(question.bloom_level, '⚪')} {question.bloom_level.name_cn}")
                    
                    st.markdown(question.question_text)
                    
                    if question.code_snippet:
                        st.code(question.code_snippet, language="python")
                    
                    # 如果已提交，禁用选项
                    answer = st.radio(
                        "请选择：", 
                        question.options, 
                        key=f"quiz_q_{current_idx}", 
                        index=None,
                        disabled=question_submitted
                    )
                    if answer:
                        st.session_state.quiz_answers[question.id] = answer[0]
                    
                    # 如果还未提交，显示提交按钮
                    if not question_submitted:
                        if st.button("提交答案", key=f"submit_{current_idx}"):
                            if question.id not in st.session_state.quiz_answers:
                                st.warning("请先选择答案！")
                            else:
                                user_ans = st.session_state.quiz_answers[question.id]
                                engine = st.session_state.quiz_engine
                                
                                is_correct, error_type, explanation = engine.evaluate_answer(
                                    session.session_id, question.id, user_ans
                                )
                                
                                # 保存提交结果到 session_state
                                st.session_state.quiz_submitted_questions[question.id] = {
                                    "is_correct": is_correct,
                                    "error_type": error_type,
                                    "explanation": explanation,
                                    "user_answer": user_ans
                                }
                                
                                # 更新 Elo
                                elo_system = st.session_state.elo_system
                                elo_change = 0
                                for kp in question.knowledge_points:
                                    result = elo_system.record_answer(
                                        user_id=user_id, knowledge_id=kp, knowledge_name=kp,
                                        question_id=question.id, is_correct=is_correct,
                                        bloom_level=question.bloom_level.level,
                                        error_type=error_type.value if error_type else None
                                    )
                                    
                                    update_user_knowledge_state(
                                        user_id=user_id, knowledge_id=kp, knowledge_name=kp,
                                        elo_rating=result['user_new_elo'],
                                        games_played=elo_system.get_user_rating(user_id, kp).games_played,
                                        wins=elo_system.get_user_rating(user_id, kp).wins,
                                        peak_elo=elo_system.get_user_rating(user_id, kp).peak_elo,
                                        streak=result['streak']
                                    )
                                    
                                    elo_change = result['user_elo_change']
                                    st.session_state.quiz_submitted_questions[question.id]["elo_change"] = elo_change
                                    
                                    if result.get('tier_change') and result['tier_change']['promoted']:
                                        st.session_state.quiz_submitted_questions[question.id]["tier_promoted"] = result['tier_change']
                                
                                save_quiz_result(
                                    user_id=user_id, question=question.question_text,
                                    user_answer=user_ans, correct_answer=question.correct_answer,
                                    is_correct=is_correct,
                                    topic=question.knowledge_points[0] if question.knowledge_points else "未分类",
                                    session_id=session.session_id, question_id=question.id,
                                    bloom_level=question.bloom_level.level,
                                    error_type=error_type.value if error_type else None,
                                    elo_change=elo_change
                                )
                                
                                if not is_correct:
                                    add_mistake(
                                        user_id=user_id, question=question.question_text,
                                        user_answer=user_ans, correct_answer=question.correct_answer,
                                        analysis=explanation,
                                        topic=question.knowledge_points[0] if question.knowledge_points else "未分类",
                                        error_type=error_type.value if error_type else None
                                    )
                                
                                st.rerun()
                    
                    # 如果已提交，显示结果和下一题按钮
                    else:
                        result_data = st.session_state.quiz_submitted_questions[question.id]
                        is_correct = result_data["is_correct"]
                        explanation = result_data["explanation"]
                        elo_change = result_data.get("elo_change", 0)
                        
                        if is_correct:
                            st.success("✅ 回答正确！")
                        else:
                            st.error(f"❌ 错误。正确答案是 {question.correct_answer}")
                        
                        if elo_change > 0:
                            st.success(f"Elo +{elo_change:.0f}")
                        elif elo_change < 0:
                            st.error(f"Elo {elo_change:.0f}")
                        
                        if result_data.get("tier_promoted"):
                            st.balloons()
                            tc = result_data["tier_promoted"]
                            st.success(f"🎉 晋级！{tc['old']} → {tc['new']}")
                        
                        with st.expander("📖 查看解析", expanded=True):
                            st.markdown(explanation)
                        
                        if current_idx < len(session.questions) - 1:
                            if st.button("下一题 →", key=f"next_{current_idx}"):
                                st.session_state.quiz_current_idx += 1
                                st.rerun()
                        else:
                            st.success("🎉 测验完成！")
                            engine = st.session_state.quiz_engine
                            feedback = engine.get_diagnostic_feedback(session.session_id)
                            st.markdown(feedback)
                            
                            if st.button("返回", key="quiz_return"):
                                # 清理测验相关的 session state
                                keys_to_delete = ["current_quiz_session", "quiz_current_idx", 
                                                 "quiz_answers", "quiz_submitted_questions"]
                                for key in keys_to_delete:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                st.rerun()
        
        # === 学习画像标签页 ===
        with tab_profile:
            show_profile_page(user_id, st.session_state.get('current_doc_name'))


# === 运行 ===
if __name__ == "__main__":
    main()
