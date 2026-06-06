"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/Button";
import { quizApi } from "./api";
import { choiceText, QuestionCard } from "./QuestionCard";
import type { WrongQuestion } from "./types";

type WrongBookFilter = "all" | "open" | "reviewed";

const filters: { value: WrongBookFilter; label: string }[] = [
  { value: "all", label: "全部" },
  { value: "open", label: "待复习" },
  { value: "reviewed", label: "已复习" },
];

export function WrongBookPage() {
  const [wrongQuestions, setWrongQuestions] = useState<WrongQuestion[]>([]);
  const [filter, setFilter] = useState<WrongBookFilter>("all");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    void loadWrongQuestions();
  }, []);

  const visibleWrongQuestions = useMemo(() => {
    if (filter === "open") {
      return wrongQuestions.filter((question) => !question.reviewed_at);
    }
    if (filter === "reviewed") {
      return wrongQuestions.filter((question) => question.reviewed_at);
    }
    return wrongQuestions;
  }, [filter, wrongQuestions]);
  const openCount = wrongQuestions.filter((question) => !question.reviewed_at).length;
  const reviewedCount = wrongQuestions.length - openCount;

  async function loadWrongQuestions() {
    setIsLoading(true);
    setError("");
    try {
      setWrongQuestions(await quizApi.wrongQuestions(null));
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "错题本加载失败。");
    } finally {
      setIsLoading(false);
    }
  }

  async function markReviewed(wrongQuestionId: string) {
    setError("");
    setNotice("");
    try {
      const reviewedQuestion = await quizApi.markReviewed(wrongQuestionId);
      setWrongQuestions((current) =>
        current.map((question) =>
          question.id === wrongQuestionId ? reviewedQuestion : question,
        ),
      );
      setNotice("已标记为复习完成。");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "标记复习失败。");
    }
  }

  async function removeWrongQuestion(wrongQuestionId: string) {
    setError("");
    setNotice("");
    try {
      await quizApi.removeWrongQuestion(wrongQuestionId);
      setWrongQuestions((current) =>
        current.filter((question) => question.id !== wrongQuestionId),
      );
      setNotice("已删除这条错题记录。");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "删除错题失败。");
    }
  }

  return (
    <div className="study-page wrong-book-page">
      <section className="study-main">
        <div className="study-page-heading">
          <div>
            <p className="section-label">错题本</p>
            <h2>错题记录</h2>
            <p>所有提交后答错的题目都会保存在这里，并作为后续相关出题的引用材料。</p>
          </div>
          <div className="study-heading-actions">
            <Link className="button button-ghost" href="/">
              返回问答
            </Link>
            <Link className="button button-primary" href="/quiz?auto=1">
              一键出题
            </Link>
          </div>
        </div>

        {error ? <p className="message-feedback-error">{error}</p> : null}
        {notice ? <p className="chat-status">{notice}</p> : null}

        <div className="wrong-book-tabs" role="tablist" aria-label="错题筛选">
          {filters.map((item) => (
            <button
              key={item.value}
              type="button"
              className={filter === item.value ? "active" : ""}
              onClick={() => setFilter(item.value)}
            >
              {item.label}
            </button>
          ))}
        </div>

        {isLoading ? <p className="empty-state quiz-empty-state">正在加载错题记录...</p> : null}
        {!isLoading && visibleWrongQuestions.length === 0 ? (
          <p className="empty-state quiz-empty-state">当前筛选下暂无错题记录。</p>
        ) : null}

        <div className="wrong-record-list">
          {visibleWrongQuestions.map((question, index) => (
            <WrongRecord
              key={question.id}
              index={index}
              question={question}
              onMarkReviewed={() => void markReviewed(question.id)}
              onRemove={() => void removeWrongQuestion(question.id)}
            />
          ))}
        </div>
      </section>

      <aside className="study-side" aria-label="错题统计">
        <section className="panel">
          <p className="section-label">记录</p>
          <h2>错题统计</h2>
          <dl className="study-stat-list">
            <div>
              <dt>全部错题</dt>
              <dd>{wrongQuestions.length} 题</dd>
            </div>
            <div>
              <dt>待复习</dt>
              <dd>{openCount} 题</dd>
            </div>
            <div>
              <dt>已复习</dt>
              <dd>{reviewedCount} 题</dd>
            </div>
          </dl>
          <Button type="button" variant="secondary" onClick={() => void loadWrongQuestions()}>
            刷新记录
          </Button>
        </section>
      </aside>
    </div>
  );
}

function WrongRecord({
  question,
  index,
  onMarkReviewed,
  onRemove,
}: {
  question: WrongQuestion;
  index: number;
  onMarkReviewed: () => void;
  onRemove: () => void;
}) {
  return (
    <article className="wrong-record">
      <div className="wrong-record-heading">
        <div>
          <strong>{question.reviewed_at ? "已复习" : "待复习"}</strong>
          <span>{formatDateTime(question.created_at)}</span>
        </div>
        <div className="wrong-record-actions">
          {!question.reviewed_at ? (
            <Button type="button" variant="secondary" onClick={onMarkReviewed}>
              标记已复习
            </Button>
          ) : null}
          <Button type="button" variant="ghost" onClick={onRemove}>
            删除
          </Button>
        </div>
      </div>

      <QuestionCard
        question={{
          id: question.question_id,
          prompt: question.prompt,
          choices: question.choices,
          correct_choice_id: question.correct_choice_id,
          explanation: question.explanation,
          source_message_ids: question.source_message_ids,
        }}
        index={index}
        selectedChoiceId={question.selected_choice_id}
        showAnswer
      />

      <p className="wrong-record-answer">
        你的答案：{choiceText(question, question.selected_choice_id) || "未作答"}
      </p>
    </article>
  );
}

function formatDateTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}
