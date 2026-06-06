from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    smoke_dir = REPO_ROOT / ".cache" / "smoke-quiz-e2e"
    shutil.rmtree(smoke_dir, ignore_errors=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ENABLE_LLM_QUIZ"] = "false"

    import backend.storage.sqlite as sqlite_storage

    sqlite_storage.APP_DATA_DIR = smoke_dir
    sqlite_storage.METADATA_DB_PATH = smoke_dir / "metadata.sqlite3"

    from fastapi import HTTPException

    from backend.schemas import (
        ConversationCreate,
        QuizGenerateRequest,
        QuizSubmitAnswer,
        QuizSubmitRequest,
    )
    from backend.services.conversation_service import (
        append_message,
        create_conversation,
        delete_conversation,
    )
    from backend.services.memory_service import create_memory_candidate, delete_memory
    from backend.services.quiz_service import (
        delete_wrong_question,
        generate_quiz,
        list_wrong_questions,
        mark_wrong_question_reviewed,
        submit_quiz,
    )

    try:
        conversation = create_conversation(
            ConversationCreate(
                id="conv_quiz_smoke",
                title="Quiz smoke conversation",
                document_ids=["doc_quiz"],
                level="undergraduate",
                mode="hybrid",
                chat_mode="multimodal",
            )
        )
        append_message(
            conversation_id=conversation.id,
            message_id="user_quiz_1",
            role="user",
            content="请解释梯度下降为什么要沿负梯度方向更新。",
            document_ids=["doc_quiz"],
            level="undergraduate",
            mode="hybrid",
            chat_mode="multimodal",
        )
        append_message(
            conversation_id=conversation.id,
            message_id="assistant_quiz_1",
            role="assistant",
            content="负梯度方向是函数局部下降最快的方向，学习率控制每一步更新幅度。",
            document_ids=["doc_quiz"],
            level="undergraduate",
            mode="hybrid",
            chat_mode="multimodal",
        )
        append_message(
            conversation_id=conversation.id,
            message_id="user_quiz_2",
            role="user",
            content="再说明过大学习率可能带来什么问题。",
            document_ids=["doc_quiz"],
            level="undergraduate",
            mode="hybrid",
            chat_mode="multimodal",
        )
        append_message(
            conversation_id=conversation.id,
            message_id="assistant_quiz_2",
            role="assistant",
            content="学习率过大可能导致震荡或越过最优点，训练损失难以下降。",
            document_ids=["doc_quiz"],
            level="undergraduate",
            mode="hybrid",
            chat_mode="multimodal",
        )

        session = asyncio.run(
            generate_quiz(QuizGenerateRequest(conversation_id=conversation.id, count=3))
        )
        assert session.conversation_id == conversation.id
        assert len(session.questions) == 3
        assert all(len(question.choices) >= 2 for question in session.questions)

        empty_recent_conversation = create_conversation(
            ConversationCreate(
                id="conv_quiz_empty_recent",
                title="Empty conversation",
                document_ids=["doc_quiz"],
                level="undergraduate",
                mode="hybrid",
                chat_mode="multimodal",
            )
        )
        recent_session = asyncio.run(
            generate_quiz(
                QuizGenerateRequest(
                    conversation_id=empty_recent_conversation.id,
                    document_ids=["doc_quiz"],
                    count=3,
                )
            )
        )
        assert len(recent_session.questions) == 3
        assert recent_session.conversation_id == conversation.id
        assert any(
            source_id.startswith("quiz_")
            for question in recent_session.questions
            for source_id in question.source_message_ids
        )
        delete_conversation(empty_recent_conversation.id)

        meta_conversation = create_conversation(
            ConversationCreate(
                id="conv_quiz_meta_history",
                title="Meta history should be cleaned",
                document_ids=["doc_meta_quiz"],
                level="undergraduate",
                mode="hybrid",
                chat_mode="multimodal",
            )
        )
        append_message(
            conversation_id=meta_conversation.id,
            message_id="user_meta_quiz",
            role="user",
            content="请基于这两个pdf，给出10道题目供我检测。",
            document_ids=["doc_meta_quiz"],
            level="undergraduate",
            mode="hybrid",
            chat_mode="multimodal",
        )
        append_message(
            conversation_id=meta_conversation.id,
            message_id="assistant_meta_quiz",
            role="assistant",
            content=(
                "根据用户要求，数量：10道，格式：直接输出题目。\n\n"
                "**Quiz 1**\n"
                "问题：连续时间傅里叶变换用于把时域信号表示为频域成分。\n"
                "答案：频域表示可以帮助分析系统对不同频率成分的响应。\n"
                "**Quiz 2**\n"
                "问题：LTI 系统的频率响应描述输入频率分量被系统改变的方式。\n"
                "答案：输出频谱等于输入频谱乘以系统频率响应。"
            ),
            document_ids=["doc_meta_quiz"],
            level="undergraduate",
            mode="hybrid",
            chat_mode="multimodal",
        )
        empty_meta_conversation = create_conversation(
            ConversationCreate(
                id="conv_quiz_empty_meta",
                title="Empty meta conversation",
                document_ids=["doc_meta_quiz"],
                level="undergraduate",
                mode="hybrid",
                chat_mode="multimodal",
            )
        )
        meta_session = asyncio.run(
            generate_quiz(
                QuizGenerateRequest(
                    conversation_id=empty_meta_conversation.id,
                    document_ids=["doc_meta_quiz"],
                    count=3,
                )
            )
        )
        assert len(meta_session.questions) == 3
        combined_meta_prompts = "\n".join(question.prompt for question in meta_session.questions)
        assert "题目数量" not in combined_meta_prompts
        assert "格式" not in combined_meta_prompts
        delete_conversation(empty_meta_conversation.id)
        delete_conversation(meta_conversation.id)

        wrong_choice = next(
            choice.id
            for choice in session.questions[0].choices
            if choice.id != session.questions[0].correct_choice_id
        )
        answers = [
            QuizSubmitAnswer(
                question_id=session.questions[0].id,
                selected_choice_id=wrong_choice,
            ),
            *[
                QuizSubmitAnswer(
                    question_id=question.id,
                    selected_choice_id=question.correct_choice_id,
                )
                for question in session.questions[1:]
            ],
        ]
        result = submit_quiz(session.id, QuizSubmitRequest(answers=answers))
        assert result.total_count == 3
        assert result.correct_count == 2

        wrong_questions = list_wrong_questions(reviewed=False)
        assert len(wrong_questions) == 1
        assert wrong_questions[0].quiz_session_id == session.id
        assert wrong_questions[0].selected_choice_id == wrong_choice

        reviewed = mark_wrong_question_reviewed(wrong_questions[0].id)
        assert reviewed.reviewed_at
        assert not list_wrong_questions(reviewed=False)
        reviewed_questions = list_wrong_questions(reviewed=True)
        assert len(reviewed_questions) == 1

        delete_wrong_question(reviewed_questions[0].id)
        assert not list_wrong_questions()

        memory_conversation = create_conversation(
            ConversationCreate(
                id="conv_quiz_memory_smoke",
                title="Memory quiz smoke",
                document_ids=["doc_memory_quiz"],
                level="undergraduate",
                mode="hybrid",
                chat_mode="multimodal",
            )
        )
        preference_memory = create_memory_candidate(
            memory_type="workflow_preference",
            key="custom_depth",
            value="用户使用过自定义回答深度设置。",
            confidence=0.7,
            source_conversation_id=memory_conversation.id,
            evidence="来自界面使用记录。",
            scope_type="document",
            scope_id="doc_memory_quiz",
            evidence_message_ids=[],
        )
        assert preference_memory is not None
        try:
            asyncio.run(
                generate_quiz(
                    QuizGenerateRequest(
                        conversation_id=memory_conversation.id,
                        document_ids=["doc_memory_quiz"],
                        count=3,
                    )
                )
            )
        except HTTPException as exc:
            assert exc.status_code == 400
        else:
            raise AssertionError("preference memory must not be used for quiz generation")

        memory = create_memory_candidate(
            memory_type="knowledge_gap",
            key="gradient_descent_review",
            value="用户需要复习梯度下降、负梯度方向和学习率过大导致震荡的问题。",
            confidence=0.86,
            source_conversation_id=memory_conversation.id,
            evidence="来自学习反馈和复习记录。",
            scope_type="document",
            scope_id="doc_memory_quiz",
            evidence_message_ids=[],
        )
        assert memory is not None
        memory_session = asyncio.run(
            generate_quiz(
                QuizGenerateRequest(
                    conversation_id=memory_conversation.id,
                    document_ids=["doc_memory_quiz"],
                    count=3,
                )
            )
        )
        assert len(memory_session.questions) == 3
        assert memory_session.conversation_id == memory_conversation.id
        delete_memory(preference_memory.id)
        delete_memory(memory.id)
        delete_conversation(memory_conversation.id)

        delete_conversation(conversation.id)
    finally:
        shutil.rmtree(smoke_dir, ignore_errors=True)

    print("smoke_quiz_e2e ok")


if __name__ == "__main__":
    main()
