from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from backend.config import as_bool
from backend.core.ids import safe_document_id
from backend.core.text import response_content_to_text
from backend.schemas import (
    ConversationMessage,
    ImageAssetPublic,
    QuizChoice,
    QuizGenerateRequest,
    QuizQuestion,
    QuizQuestionResult,
    QuizSession,
    QuizSubmitRequest,
    QuizSubmitResponse,
    WrongQuestion,
)
from backend.services.conversation_service import (
    list_conversations,
    list_messages,
    read_conversation,
)
from backend.services.profile_service import get_profile
from backend.storage.sqlite import metadata_connection

CHOICE_IDS = ("A", "B", "C", "D")
MESSAGE_ID_PATTERN = re.compile(r"\bmsg_[A-Za-z0-9._:-]+\b")
QUIZ_MEMORY_TYPES = {
    "knowledge_gap",
    "learning_gap",
    "concept_gap",
    "mistake",
    "wrong_question",
}
META_STUDY_TERMS = (
    "用户明确要求",
    "用户要求",
    "生成 quiz",
    "生成quiz",
    "quizs",
    "题目数量",
    "自定义设置",
    "answer-depth",
    "memory type",
    "记忆类型",
    "用户行为",
    "操作习惯",
    "界面设置",
    "候选记忆",
    "审核记忆",
    "证据消息",
    "来源对话",
)
STRONG_META_STUDY_TERMS = (
    "用户明确要求",
    "用户指令",
    "用户行为",
    "自定义设置",
    "answer-depth",
    "memory id",
    "记忆类型",
    "workflow_preference",
    "custom_depth",
    "重复出现",
    "题目数量",
)


@dataclass(frozen=True)
class QuizSource:
    id: str
    role: str
    content: str
    status: str = "sent"
    related_images: tuple[ImageAssetPublic, ...] = ()


async def generate_quiz(payload: QuizGenerateRequest) -> QuizSession:
    count = _quiz_count(payload.count)
    sources, title, source_conversation_id = _resolve_quiz_sources(payload)
    if not sources:
        raise HTTPException(
            status_code=400,
            detail="还没有可用于出题的对话或记忆。请先完成问答，或在记忆审核中接受至少一条记忆。",
        )

    questions = await _generate_questions_with_llm(sources, count)
    if len(questions) < count:
        fallback = _fallback_questions(sources, count)
        existing_prompts = {question.prompt for question in questions}
        questions.extend(
            question for question in fallback if question.prompt not in existing_prompts
        )
    questions = questions[:count]
    if len(questions) < count:
        raise HTTPException(status_code=400, detail="当前材料还不足以生成小测验。")

    questions = _attach_related_images_to_questions(questions, sources)
    profile_id = _profile_id()
    now = _utc_now()
    session = QuizSession(
        id=_new_id("quiz"),
        profile_id=profile_id,
        conversation_id=source_conversation_id,
        title=title,
        questions=questions,
        created_at=now,
    )
    with metadata_connection() as connection:
        connection.execute(
            """
            INSERT INTO quiz_sessions (
                id, profile_id, conversation_id, title, questions_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session.id,
                session.profile_id,
                session.conversation_id,
                session.title,
                _json_dumps([_model_dump(question) for question in session.questions]),
                session.created_at,
            ),
        )
    return session


def submit_quiz(session_id: str, payload: QuizSubmitRequest) -> QuizSubmitResponse:
    session = read_quiz_session(session_id)
    answers_by_question = {
        answer.question_id: answer.selected_choice_id for answer in payload.answers
    }
    results: list[QuizQuestionResult] = []
    correct_count = 0
    for question in session.questions:
        selected_choice_id = answers_by_question.get(question.id)
        is_correct = selected_choice_id == question.correct_choice_id
        if is_correct:
            correct_count += 1
        results.append(
            QuizQuestionResult(
                question=question,
                selected_choice_id=selected_choice_id,
                is_correct=is_correct,
                correct_choice_id=question.correct_choice_id,
                explanation=question.explanation,
            )
        )

    now = _utc_now()
    profile_id = _profile_id()
    with metadata_connection() as connection:
        connection.execute(
            """
            INSERT INTO quiz_attempts (
                id, quiz_session_id, profile_id, conversation_id, answers_json,
                correct_count, total_count, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _new_id("attempt"),
                session.id,
                profile_id,
                session.conversation_id,
                _json_dumps([_model_dump(answer) for answer in payload.answers]),
                correct_count,
                len(session.questions),
                now,
            ),
        )
        for result in results:
            if result.is_correct:
                continue
            question = result.question
            connection.execute(
                """
                INSERT INTO wrong_questions (
                    id, profile_id, quiz_session_id, conversation_id, question_id,
                    prompt, choices_json, selected_choice_id, correct_choice_id,
                    explanation, source_message_ids_json, related_images_json,
                    created_at, reviewed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    _new_id("wrong"),
                    profile_id,
                    session.id,
                    session.conversation_id,
                    question.id,
                    question.prompt,
                    _json_dumps([_model_dump(choice) for choice in question.choices]),
                    result.selected_choice_id or "",
                    question.correct_choice_id,
                    question.explanation,
                    _json_dumps(question.source_message_ids),
                    _json_dumps(
                        [
                            _model_dump(image, exclude_none=True)
                            for image in question.related_images
                        ]
                    ),
                    now,
                ),
            )

    return QuizSubmitResponse(
        session_id=session.id,
        correct_count=correct_count,
        total_count=len(session.questions),
        results=results,
    )


def read_quiz_session(session_id: str) -> QuizSession:
    profile_id = _profile_id()
    with metadata_connection() as connection:
        row = connection.execute(
            """
            SELECT * FROM quiz_sessions
            WHERE id = ? AND profile_id = ?
            """,
            (_require_id(session_id, "Quiz session id"), profile_id),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="小测验不存在。")
    return _session_from_row(dict(row))


def list_wrong_questions(reviewed: bool | None = None) -> list[WrongQuestion]:
    profile_id = _profile_id()
    if reviewed is True:
        where = "profile_id = ? AND reviewed_at IS NOT NULL"
    elif reviewed is False:
        where = "profile_id = ? AND reviewed_at IS NULL"
    else:
        where = "profile_id = ?"
    with metadata_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT * FROM wrong_questions
            WHERE {where}
            ORDER BY created_at DESC
            """,
            (profile_id,),
        ).fetchall()
    return [_wrong_question_from_row(dict(row)) for row in rows]


def mark_wrong_question_reviewed(wrong_question_id: str) -> WrongQuestion:
    wrong_question = read_wrong_question(wrong_question_id)
    with metadata_connection() as connection:
        connection.execute(
            """
            UPDATE wrong_questions
            SET reviewed_at = ?
            WHERE id = ? AND profile_id = ?
            """,
            (_utc_now(), wrong_question.id, wrong_question.profile_id),
        )
    return read_wrong_question(wrong_question.id)


def delete_wrong_question(wrong_question_id: str) -> None:
    wrong_question = read_wrong_question(wrong_question_id)
    with metadata_connection() as connection:
        connection.execute(
            """
            DELETE FROM wrong_questions
            WHERE id = ? AND profile_id = ?
            """,
            (wrong_question.id, wrong_question.profile_id),
        )


def read_wrong_question(wrong_question_id: str) -> WrongQuestion:
    profile_id = _profile_id()
    with metadata_connection() as connection:
        row = connection.execute(
            """
            SELECT * FROM wrong_questions
            WHERE id = ? AND profile_id = ?
            """,
            (_require_id(wrong_question_id, "Wrong question id"), profile_id),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="错题不存在。")
    return _wrong_question_from_row(dict(row))


def _resolve_quiz_sources(
    payload: QuizGenerateRequest,
) -> tuple[list[QuizSource], str, str]:
    conversation = None
    conversation_id = _optional_text(payload.conversation_id)
    document_ids = _normalize_document_ids(payload.document_ids)

    if conversation_id:
        try:
            conversation = read_conversation(conversation_id)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
        if conversation:
            if not document_ids:
                document_ids = conversation.document_ids
            messages = _learning_answer_sources(list_messages(conversation.id))
            if messages:
                return (
                    messages,
                    f"{conversation.title} 的学习复习小测验",
                    conversation.id,
                )

    quiz_sources, quiz_conversation_id = _quiz_record_sources(
        document_ids=document_ids,
        conversation_id=conversation_id,
        limit=12,
    )
    recent_sources, recent_conversation_id = _recent_learning_sources(
        document_ids=document_ids,
        exclude_conversation_id=conversation.id if conversation else conversation_id,
        limit=12,
    )
    combined_history_sources = _dedupe_sources([*quiz_sources, *recent_sources])[:12]
    if combined_history_sources:
        return (
            combined_history_sources,
            "学习历史与小测记录复习",
            quiz_conversation_id
            or recent_conversation_id
            or conversation_id
            or "learning_history",
        )

    memory_sources = _memory_sources(
        document_ids=document_ids,
        conversation_id=conversation_id,
        limit=12,
    )
    if memory_sources:
        title = "记忆小测验"
        if conversation:
            title = f"{conversation.title} 的知识缺口小测验"
        return memory_sources, title, conversation_id or "memory"

    return [], "问题记忆小测验", conversation_id or "memory"


def _learning_answer_sources(
    messages: list[ConversationMessage],
    *,
    limit: int = 12,
) -> list[QuizSource]:
    sources: list[QuizSource] = []
    for message in messages:
        if message.status != "sent" or not message.content.strip():
            continue
        if message.role != "assistant":
            continue
        content = _study_content_from_assistant(message.content)
        if not content:
            continue
        sources.append(
            QuizSource(
                id=f"learn_{message.id}",
                role="assistant",
                content=f"历史学习记录：{content}",
                related_images=tuple(message.related_images),
            )
        )
    return sources[-max(1, limit) :]


def _quiz_record_sources(
    *,
    document_ids: list[str],
    conversation_id: str | None,
    limit: int,
) -> tuple[list[QuizSource], str | None]:
    profile_id = _profile_id()
    matching_conversation_ids = _matching_conversation_ids(document_ids)
    if conversation_id:
        matching_conversation_ids.add(conversation_id)
    where_parts = ["profile_id = ?"]
    params: list[Any] = [profile_id]
    if matching_conversation_ids:
        placeholders = ", ".join("?" for _item in matching_conversation_ids)
        where_parts.append(f"conversation_id IN ({placeholders})")
        params.extend(sorted(matching_conversation_ids))

    sources: list[tuple[str, QuizSource]] = []
    with metadata_connection() as connection:
        wrong_rows = connection.execute(
            f"""
            SELECT *
            FROM wrong_questions
            WHERE {" AND ".join(where_parts)}
            ORDER BY reviewed_at IS NOT NULL ASC, created_at DESC
            LIMIT 30
            """,
            params,
        ).fetchall()
        session_rows = connection.execute(
            f"""
            SELECT *
            FROM quiz_sessions
            WHERE {" AND ".join(where_parts)}
            ORDER BY created_at DESC
            LIMIT 20
            """,
            params,
        ).fetchall()

    first_conversation_id = None
    for row in wrong_rows:
        data = dict(row)
        content = _wrong_question_source_text(data)
        if not content:
            continue
        first_conversation_id = first_conversation_id or str(data["conversation_id"])
        sources.append(
            (
                str(data["created_at"]),
                QuizSource(
                    id=f"wrong_{data['id']}",
                    role="wrong_question",
                    content=content,
                    related_images=tuple(
                        _images_from_json(data.get("related_images_json"))
                    ),
                ),
            )
        )

    for row in session_rows:
        data = dict(row)
        created_at = str(data["created_at"])
        first_conversation_id = first_conversation_id or str(data["conversation_id"])
        for question in _questions_from_json(data.get("questions_json")):
            content = _quiz_question_source_text(question)
            if not content:
                continue
            sources.append(
                (
                    created_at,
                    QuizSource(
                        id=f"quiz_{data['id']}_{question.id}",
                        role="quiz_record",
                        content=content,
                        related_images=tuple(question.related_images),
                    ),
                )
            )

    sorted_sources = [
        source for _created_at, source in sorted(sources, key=lambda item: item[0], reverse=True)
    ]
    return sorted_sources[:limit], first_conversation_id


def _dedupe_sources(sources: list[QuizSource]) -> list[QuizSource]:
    result: list[QuizSource] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    for source in sources:
        fingerprint = _clip(source.content, 160)
        if source.id in seen_ids or fingerprint in seen_content:
            continue
        seen_ids.add(source.id)
        seen_content.add(fingerprint)
        result.append(source)
    return result


def _recent_learning_sources(
    *,
    document_ids: list[str],
    exclude_conversation_id: str | None,
    limit: int,
) -> tuple[list[QuizSource], str | None]:
    excluded = _optional_text(exclude_conversation_id)
    sources: list[QuizSource] = []
    source_conversation_id = None
    for conversation in list_conversations()[:12]:
        if excluded and conversation.id == excluded:
            continue
        if document_ids and not _has_document_overlap(conversation.document_ids, document_ids):
            continue
        next_sources = _learning_answer_sources(list_messages(conversation.id), limit=limit)
        if not next_sources:
            continue
        if source_conversation_id is None:
            source_conversation_id = conversation.id
        sources.extend(next_sources)
        if len(sources) >= limit:
            break
    return sources[-limit:], source_conversation_id


def _matching_conversation_ids(document_ids: list[str]) -> set[str]:
    if not document_ids:
        return set()
    normalized_document_ids = {safe_document_id(str(item or "")) for item in document_ids}
    result: set[str] = set()
    for conversation in list_conversations():
        if _has_document_overlap(conversation.document_ids, list(normalized_document_ids)):
            result.add(conversation.id)
    return result


def _wrong_question_source_text(row: dict[str, Any]) -> str:
    prompt = _required_text(row.get("prompt"))
    choices = _choices_from_json(row.get("choices_json"))
    correct_choice_id = _required_text(row.get("correct_choice_id"))
    selected_choice_id = _required_text(row.get("selected_choice_id"))
    explanation = _required_text(row.get("explanation"))
    correct_choice = next(
        (choice.text for choice in choices if choice.id == correct_choice_id),
        correct_choice_id,
    )
    selected_choice = next(
        (choice.text for choice in choices if choice.id == selected_choice_id),
        selected_choice_id,
    )
    text = (
        f"错题记录：{prompt}\n"
        f"用户错选：{selected_choice}\n"
        f"正确答案：{correct_choice}\n"
        f"解析：{explanation}"
    )
    if _is_meta_study_text(text):
        return ""
    return text


def _quiz_question_source_text(question: QuizQuestion) -> str:
    correct_choice = next(
        (choice.text for choice in question.choices if choice.id == question.correct_choice_id),
        question.correct_choice_id,
    )
    text = (
        f"历史 quiz 记录：{question.prompt}\n"
        f"正确答案：{correct_choice}\n"
        f"解析：{question.explanation}"
    )
    if _is_meta_study_text(text):
        return ""
    return text


def _study_content_from_assistant(content: str) -> str:
    text = str(content or "").strip()
    if not text:
        return ""

    marker_positions = [
        position
        for position in (
            text.find("### 一"),
            text.find("## 一"),
            text.find("**Quiz 1"),
            text.find("Quiz 1"),
            text.find("**题目 1"),
            text.find("题目 1"),
        )
        if position >= 0
    ]
    for pattern in (
        r"(?m)^\s*#{1,4}\s*1[.、]",
        r"(?m)^\s*(?:\*\*)?Quiz\s*1",
        r"(?m)^\s*(?:\*\*)?题目\s*1",
        r"(?m)^\s*1[.、]\s*[^\d\s]",
    ):
        match = re.search(pattern, text)
        if match:
            marker_positions.append(match.start())
    if marker_positions:
        text = text[min(marker_positions) :]

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _is_meta_line(stripped):
            continue
        lines.append(stripped)
    cleaned = "\n".join(lines)
    if len(re.findall(r"[\u4e00-\u9fff]", cleaned)) < 10:
        return ""
    if _is_meta_study_text(cleaned) and len(cleaned) < 260:
        return ""
    return cleaned


def _is_meta_line(line: str) -> bool:
    normalized = line.strip().lower()
    if normalized in {"---", "--"}:
        return True
    prefixes = (
        "根据用户",
        "用户要求",
        "严格遵循",
        "我们对",
        "以下是严格",
        "- 数量",
        "- 语言",
        "- 来源",
        "- 格式",
        "- 输出",
        "- 题目",
        "- 多文档",
        "- 每题",
        "- 每道题",
        "- 所有题",
        "- 图片",
        "- 不分类",
        "严格满足用户要求",
    )
    return any(line.startswith(prefix) for prefix in prefixes)


def _is_meta_study_text(text: str) -> bool:
    normalized = str(text or "").lower()
    if any(term.lower() in normalized for term in STRONG_META_STUDY_TERMS):
        return True
    hits = sum(1 for term in META_STUDY_TERMS if term.lower() in normalized)
    if hits >= 2:
        return True
    if "custom_depth" in normalized or "workflow_preference" in normalized:
        return True
    if "用户曾使用过" in normalized and "设置" in normalized:
        return True
    return False


def _has_document_overlap(left: list[str], right: list[str]) -> bool:
    left_ids = {safe_document_id(str(item or "")) for item in left}
    right_ids = {safe_document_id(str(item or "")) for item in right}
    return bool(left_ids & right_ids)


def _memory_sources(
    *,
    document_ids: list[str],
    conversation_id: str | None,
    limit: int,
) -> list[QuizSource]:
    profile_id = _profile_id()
    allowed_scopes = _allowed_scopes(document_ids, conversation_id)
    with metadata_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM user_memories
            WHERE profile_id = ?
              AND status IN ('active', 'candidate')
              AND (expires_at IS NULL OR expires_at = '' OR expires_at > ?)
            ORDER BY
              CASE status WHEN 'active' THEN 0 ELSE 1 END,
              updated_at DESC
            LIMIT 50
            """,
            (profile_id, _utc_now()),
        ).fetchall()

    scored: list[tuple[int, str, QuizSource]] = []
    for row in rows:
        data = dict(row)
        memory_type = str(data.get("memory_type") or "")
        if memory_type not in QUIZ_MEMORY_TYPES:
            continue
        scope_type = str(data.get("scope_type") or "global")
        scope_id = str(data.get("scope_id") or "")
        scope = (scope_type, scope_id)
        status = str(data.get("status") or "candidate")
        is_scoped_match = scope in allowed_scopes
        is_global = scope_type == "global"
        if not is_scoped_match and not is_global:
            continue
        score = 0
        if status == "active":
            score += 20
        if is_scoped_match and not is_global:
            score += 12
        if is_global:
            score += 4
        if _string_list_from_json(data.get("evidence_message_ids_json")):
            score += 3
        source = QuizSource(
            id=f"mem_{data['id']}",
            role="memory",
            content=_memory_source_text(data),
        )
        scored.append((score, str(data.get("updated_at") or ""), source))

    return [source for _score, _updated_at, source in sorted(scored, reverse=True)[:limit]]


def _memory_source_text(row: dict[str, Any]) -> str:
    parts = [
        f"知识缺口：{row.get('value') or ''}",
    ]
    evidence = _optional_text(row.get("evidence"))
    if evidence:
        parts.append(f"证据：{evidence}")
    return "\n".join(parts)


async def _generate_questions_with_llm(
    sources: list[QuizSource],
    count: int,
) -> list[QuizQuestion]:
    if not as_bool(os.getenv("ENABLE_LLM_QUIZ"), True):
        return []
    try:
        from backend.rag.factory import build_model_functions

        llm_func, _vision_func = build_model_functions()
        raw = await llm_func(
            _quiz_prompt(sources, count),
            system_prompt=(
                "你是严谨的中文助教，只能根据给定学习材料生成复习选择题。"
                "必须返回合法 JSON，不要输出 Markdown。"
            ),
            temperature=0.2,
        )
        return _questions_from_llm_text(
            response_content_to_text(raw),
            count=count,
            valid_message_ids={source.id for source in sources},
        )
    except Exception:
        return []


def _quiz_prompt(sources: list[QuizSource], count: int) -> str:
    material = "\n".join(
        f"[{source.id}][{source.role}] {_clip(source.content, 700)}"
        for source in sources[-12:]
    )
    return f"""
请基于下面的学习材料，生成 {count} 道中文单选题，用于检查用户是否理解这些材料。

要求：
- 优先考查历史学习回答、历史 quiz 记录和错题记录中涉及的知识点、结论、条件、推理或易错点。
- 只考查学习材料中已经出现的信息，不引入外部事实。
- 不要考查用户要求生成几道题、输出格式、用户偏好、界面设置、操作习惯、记忆状态或本系统功能。
- 如果材料来自历史 quiz 或错题记录，应围绕其中的学科知识重新出题，不要考“这条历史记录写了什么”。
- 每题 4 个选项，选项 id 固定为 A、B、C、D。
- explanation 要说明正确项为什么对，并指出它对应的材料依据。
- source_message_ids 只能使用学习材料中出现过的 id。
- 返回 JSON，格式为：
{{
  "questions": [
    {{
      "prompt": "题干",
      "choices": [
        {{"id": "A", "text": "选项"}},
        {{"id": "B", "text": "选项"}},
        {{"id": "C", "text": "选项"}},
        {{"id": "D", "text": "选项"}}
      ],
      "correct_choice_id": "A",
      "explanation": "解析",
      "source_message_ids": ["msg_x"]
    }}
  ]
}}

学习材料：
{material}
""".strip()


def _questions_from_llm_text(
    text: str,
    *,
    count: int,
    valid_message_ids: set[str],
) -> list[QuizQuestion]:
    parsed = _parse_json_object(text)
    raw_questions = parsed.get("questions") if isinstance(parsed, dict) else None
    if not isinstance(raw_questions, list):
        return []

    questions: list[QuizQuestion] = []
    for raw_question in raw_questions:
        if not isinstance(raw_question, dict):
            continue
        question = _normalize_question(raw_question, valid_message_ids)
        if question:
            questions.append(question)
        if len(questions) >= count:
            break
    return questions


def _normalize_question(
    raw_question: dict[str, Any],
    valid_message_ids: set[str],
) -> QuizQuestion | None:
    prompt = _required_text(raw_question.get("prompt"))
    explanation = _required_text(raw_question.get("explanation"))
    if not prompt or not explanation:
        return None

    raw_choices = raw_question.get("choices")
    if not isinstance(raw_choices, list):
        return None
    choices: list[QuizChoice] = []
    for index, raw_choice in enumerate(raw_choices[:4]):
        if not isinstance(raw_choice, dict):
            continue
        choice_id = str(raw_choice.get("id") or CHOICE_IDS[index]).strip().upper()
        if choice_id not in CHOICE_IDS:
            choice_id = CHOICE_IDS[index]
        text = _required_text(raw_choice.get("text"))
        if text:
            choices.append(QuizChoice(id=choice_id, text=text[:400]))

    if len(choices) < 2:
        return None
    correct_choice_id = str(raw_question.get("correct_choice_id") or "").strip().upper()
    if correct_choice_id not in {choice.id for choice in choices}:
        return None

    source_ids = []
    raw_source_ids = raw_question.get("source_message_ids")
    if isinstance(raw_source_ids, list):
        source_ids = [
            str(message_id)
            for message_id in raw_source_ids
            if str(message_id) in valid_message_ids
        ][:4]

    return QuizQuestion(
        id=_new_id("qq"),
        prompt=prompt[:500],
        choices=choices,
        correct_choice_id=correct_choice_id,
        explanation=explanation[:800],
        source_message_ids=source_ids,
    )


def _fallback_questions(sources: list[QuizSource], count: int) -> list[QuizQuestion]:
    pairs = _source_pairs(sources)
    if not pairs:
        pairs = [(sources[index], sources[index]) for index in range(len(sources))]

    templates = (
        "根据材料“{focus}”，哪一项最符合应掌握的核心结论？",
        "复习这条材料“{focus}”时，应该优先记住哪一点？",
        "如果把“{focus}”整理成复习卡片，哪一项最贴近材料内容？",
    )
    questions: list[QuizQuestion] = []
    for index in range(count):
        prompt_source, answer_source = pairs[index % len(pairs)]
        focus = _clip(prompt_source.content, 70)
        answer = _clip(answer_source.content, 180)
        correct_text = _answer_choice_text(answer)
        source_message_ids = []
        for source_id in (prompt_source.id, answer_source.id):
            if source_id not in source_message_ids:
                source_message_ids.append(source_id)
        questions.append(
            QuizQuestion(
                id=_new_id("qq"),
                prompt=templates[index % len(templates)].format(focus=focus),
                choices=[
                    QuizChoice(id="A", text=correct_text),
                    QuizChoice(id="B", text="应该脱离材料，直接猜测一个新的结论。"),
                    QuizChoice(id="C", text="只需要记住关键词，不需要理解条件和推理。"),
                    QuizChoice(id="D", text="材料没有提供任何可复习的信息。"),
                ],
                correct_choice_id="A",
                explanation=f"正确项来自学习材料：{correct_text}",
                source_message_ids=source_message_ids,
            )
        )
    return questions


def _attach_related_images_to_questions(
    questions: list[QuizQuestion],
    sources: list[QuizSource],
) -> list[QuizQuestion]:
    images_by_source_id = {
        source.id: source.related_images for source in sources if source.related_images
    }
    if not images_by_source_id:
        return questions

    attached_questions: list[QuizQuestion] = []
    for question in questions:
        related_images = _related_images_for_source_ids(
            question.source_message_ids,
            images_by_source_id,
        )
        if not related_images:
            attached_questions.append(question)
            continue
        data = _model_dump(question)
        data["related_images"] = [
            _model_dump(image, exclude_none=True) for image in related_images
        ]
        attached_questions.append(QuizQuestion(**data))
    return attached_questions


def _related_images_for_source_ids(
    source_ids: list[str],
    images_by_source_id: dict[str, tuple[ImageAssetPublic, ...]],
) -> list[ImageAssetPublic]:
    related_images: list[ImageAssetPublic] = []
    seen: set[tuple[str, str]] = set()
    for source_id in source_ids:
        source_images = images_by_source_id.get(source_id)
        if source_images is None and source_id.startswith("msg_"):
            source_images = images_by_source_id.get(f"learn_{source_id}")
        for image in source_images or ():
            key = (image.document_id, image.image_id)
            if key in seen:
                continue
            seen.add(key)
            related_images.append(image)
            if len(related_images) >= 6:
                return related_images
    return related_images


def _source_pairs(sources: list[QuizSource]) -> list[tuple[QuizSource, QuizSource]]:
    pairs: list[tuple[QuizSource, QuizSource]] = []
    pending_user = None
    for source in sources:
        if source.role == "user":
            pending_user = source
            continue
        if source.role == "assistant" and pending_user is not None:
            pairs.append((pending_user, source))
            pending_user = None
    return pairs


def _answer_choice_text(text: str) -> str:
    normalized = _clip(_strip_markdown(text), 180)
    if not normalized:
        return "应回到材料中的定义、步骤或结论进行复习。"
    return normalized


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _session_from_row(row: dict[str, Any]) -> QuizSession:
    return QuizSession(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        conversation_id=str(row["conversation_id"]),
        title=str(row["title"]),
        questions=_questions_from_json(row.get("questions_json")),
        created_at=str(row["created_at"]),
    )


def _wrong_question_from_row(row: dict[str, Any]) -> WrongQuestion:
    source_message_ids = _string_list_from_json(row.get("source_message_ids_json"))
    related_images = _images_from_json(row.get("related_images_json"))
    if not related_images:
        related_images = _related_images_from_wrong_question_row(
            row,
            source_message_ids,
        )
    return WrongQuestion(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        quiz_session_id=str(row["quiz_session_id"]),
        conversation_id=str(row["conversation_id"]),
        question_id=str(row["question_id"]),
        prompt=str(row["prompt"]),
        choices=_choices_from_json(row.get("choices_json")),
        selected_choice_id=str(row.get("selected_choice_id") or ""),
        correct_choice_id=str(row["correct_choice_id"]),
        explanation=str(row.get("explanation") or ""),
        source_message_ids=source_message_ids,
        related_images=related_images,
        created_at=str(row["created_at"]),
        reviewed_at=row.get("reviewed_at"),
    )


def _related_images_from_wrong_question_row(
    row: dict[str, Any],
    source_message_ids: list[str],
) -> list[ImageAssetPublic]:
    conversation_id = _optional_text(row.get("conversation_id"))
    if not conversation_id:
        return []
    message_ids = _message_ids_from_source_refs(
        source_message_ids,
        [
            row.get("prompt"),
            row.get("explanation"),
        ],
    )
    if not message_ids:
        return []
    try:
        messages = list_messages(conversation_id)
    except Exception:
        return []
    related_images: list[ImageAssetPublic] = []
    seen: set[tuple[str, str]] = set()
    for message in messages:
        if message.id not in message_ids:
            continue
        for image in message.related_images:
            key = (image.document_id, image.image_id)
            if key in seen:
                continue
            seen.add(key)
            related_images.append(image)
            if len(related_images) >= 6:
                return related_images
    return related_images


def _message_ids_from_source_refs(
    source_ids: list[str],
    text_values: list[Any],
) -> set[str]:
    message_ids: set[str] = set()
    for source_id in source_ids:
        text = str(source_id or "").strip()
        if text.startswith("learn_"):
            text = text.removeprefix("learn_")
        if text.startswith("msg_"):
            message_ids.add(text)
    for value in text_values:
        message_ids.update(MESSAGE_ID_PATTERN.findall(str(value or "")))
    return message_ids


def _questions_from_json(value: Any) -> list[QuizQuestion]:
    records = _records_from_json(value)
    questions: list[QuizQuestion] = []
    for record in records:
        try:
            questions.append(QuizQuestion(**record))
        except Exception:
            continue
    return questions


def _choices_from_json(value: Any) -> list[QuizChoice]:
    choices: list[QuizChoice] = []
    for record in _records_from_json(value):
        try:
            choices.append(QuizChoice(**record))
        except Exception:
            continue
    return choices


def _images_from_json(value: Any) -> list[ImageAssetPublic]:
    images: list[ImageAssetPublic] = []
    for record in _records_from_json(value):
        try:
            images.append(ImageAssetPublic(**record))
        except Exception:
            continue
    return images


def _records_from_json(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _string_list_from_json(value: Any) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()][:8]


def _allowed_scopes(
    document_ids: list[str],
    conversation_id: str | None,
) -> set[tuple[str, str]]:
    scopes: set[tuple[str, str]] = {("global", "")}
    for document_id in document_ids:
        scopes.add(("document", document_id))
    if conversation_id:
        scopes.add(("conversation", conversation_id))
    for course in _course_scope_ids(document_ids):
        scopes.add(("course", course))
    return scopes


def _course_scope_ids(document_ids: list[str]) -> list[str]:
    if not document_ids:
        return []
    placeholders = ", ".join("?" for _item in document_ids)
    with metadata_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT DISTINCT course FROM managed_files
            WHERE profile_id = ? AND document_id IN ({placeholders})
              AND course IS NOT NULL AND TRIM(course) != ''
            """,
            (_profile_id(), *document_ids),
        ).fetchall()
    return [str(row["course"]) for row in rows]


def _normalize_document_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        document_id = safe_document_id(str(item or ""))
        if document_id and document_id not in seen:
            result.append(document_id)
            seen.add(document_id)
    return result


def _strip_markdown(text: str) -> str:
    text = re.sub(r"`{1,3}[^`]*`{1,3}", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
    text = re.sub(r"[*_>#-]+", " ", text)
    return " ".join(text.split())


def _clip(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}..."


def _required_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _quiz_count(value: int) -> int:
    return max(1, min(int(value or 3), 5))


def _require_id(value: str, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail=f"{label} is required.")
    return text


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _profile_id() -> str:
    return get_profile().id


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _model_dump(model: Any, **kwargs: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)
    return model.dict(**kwargs)
