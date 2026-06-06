"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/Button";
import { loadConversations } from "@/lib/conversationStorage";
import type { Conversation } from "@/types/chat";
import { quizApi } from "./api";
import { QuestionCard } from "./QuestionCard";
import type { QuizSession, QuizSubmitResponse } from "./types";

export function QuizPracticePage() {
  const searchParams = useSearchParams();
  const conversationId = searchParams.get("conversationId") || "";
  const shouldAutoStart = searchParams.get("auto") === "1";
  const autoStartedRef = useRef(false);

  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [isContextReady, setIsContextReady] = useState(false);
  const [session, setSession] = useState<QuizSession | null>(null);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [result, setResult] = useState<QuizSubmitResponse | null>(null);
  const [wrongRecordCount, setWrongRecordCount] = useState<number | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const canSubmit = Boolean(
    session && !result && session.questions.every((question) => answers[question.id]),
  );
  const resultByQuestionId = useMemo(() => {
    return new Map(result?.results.map((item) => [item.question.id, item]) ?? []);
  }, [result]);

  useEffect(() => {
    const conversations = loadConversations();
    const matchedConversation =
      conversations.find((item) => item.id === conversationId) ||
      conversations[0] ||
      null;
    setConversation(matchedConversation);
    setIsContextReady(true);
  }, [conversationId]);

  const loadWrongRecordCount = useCallback(async () => {
    try {
      const wrongQuestions = await quizApi.wrongQuestions(null);
      setWrongRecordCount(wrongQuestions.length);
    } catch {
      setWrongRecordCount(null);
    }
  }, []);

  const generateQuiz = useCallback(async () => {
    if (isGenerating) return;
    setIsGenerating(true);
    setError("");
    setNotice("");
    setResult(null);
    setAnswers({});
    try {
      const nextSession = await quizApi.generate({
        conversationId: conversation?.id,
        documentIds: conversation?.documentIds,
      });
      setSession(nextSession);
    } catch (nextError) {
      setError(formatQuizError(nextError));
    } finally {
      setIsGenerating(false);
    }
  }, [conversation, isGenerating]);

  useEffect(() => {
    void loadWrongRecordCount();
  }, [loadWrongRecordCount]);

  useEffect(() => {
    if (!isContextReady || !shouldAutoStart || autoStartedRef.current) return;
    autoStartedRef.current = true;
    void generateQuiz();
  }, [generateQuiz, isContextReady, shouldAutoStart]);

  async function submitQuiz() {
    if (!session || isSubmitting) return;
    setIsSubmitting(true);
    setError("");
    setNotice("");
    try {
      const response = await quizApi.submit(
        session.id,
        session.questions.map((question) => ({
          question_id: question.id,
          selected_choice_id: answers[question.id],
        })),
      );
      const wrongCount = response.results.filter((item) => !item.is_correct).length;
      setResult(response);
      setNotice(
        wrongCount
          ? `${wrongCount} 道错题已写入错题本，后续相关出题会引用这些错题记录。`
          : "本次全部正确，没有新增错题记录。",
      );
      void loadWrongRecordCount();
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "提交失败。");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="study-page quiz-practice-page">
      <section className="study-main">
        <div className="study-page-heading">
          <div>
            <p className="section-label">一键出题</p>
            <h2>做题练习</h2>
            <p>{contextText(conversation)}</p>
          </div>
          <div className="study-heading-actions">
            <Link className="button button-ghost" href="/">
              返回问答
            </Link>
            <Link className="button button-secondary" href="/wrong-book">
              错题本
            </Link>
          </div>
        </div>

        {error ? <p className="message-feedback-error">{error}</p> : null}
        {notice ? <p className="chat-status">{notice}</p> : null}

        <div className="quiz-practice-toolbar">
          <Button
            type="button"
            variant="primary"
            disabled={isGenerating}
            onClick={() => void generateQuiz()}
          >
            {isGenerating ? "正在出题..." : session ? "重新出题" : "开始出题"}
          </Button>
          {result ? (
            <strong>
              得分 {result.correct_count}/{result.total_count}
            </strong>
          ) : null}
        </div>

        {!session && !isGenerating ? (
          <p className="empty-state quiz-empty-state">
            点击开始出题后，会基于当前对话、历史小测和错题记录生成 3 道复习题。
          </p>
        ) : null}
        {isGenerating ? (
          <p className="empty-state quiz-empty-state">正在基于学习记录生成题目...</p>
        ) : null}

        {session ? (
          <div className="quiz-session quiz-practice-session">
            <div className="quiz-session-heading">
              <strong>{session.title}</strong>
              <span>{session.questions.length} 题</span>
            </div>
            {session.questions.map((question, index) => {
              const questionResult = resultByQuestionId.get(question.id);
              return (
                <QuestionCard
                  key={question.id}
                  question={question}
                  index={index}
                  selectedChoiceId={
                    answers[question.id] || questionResult?.selected_choice_id || undefined
                  }
                  result={questionResult}
                  onSelect={(choiceId) =>
                    setAnswers((current) => ({
                      ...current,
                      [question.id]: choiceId,
                    }))
                  }
                />
              );
            })}
            <div className="quiz-submit-row quiz-practice-submit">
              <Button
                type="button"
                variant="primary"
                disabled={!canSubmit || isSubmitting}
                onClick={() => void submitQuiz()}
              >
                {isSubmitting ? "判分中..." : result ? "已提交" : "提交答案"}
              </Button>
            </div>
          </div>
        ) : null}
      </section>

      <aside className="study-side" aria-label="小测记录说明">
        <section className="panel">
          <p className="section-label">记录引用</p>
          <h2>学习闭环</h2>
          <dl className="study-stat-list">
            <div>
              <dt>当前错题记录</dt>
              <dd>{wrongRecordCount === null ? "未加载" : `${wrongRecordCount} 题`}</dd>
            </div>
            <div>
              <dt>题目数量</dt>
              <dd>3 题</dd>
            </div>
          </dl>
          <p className="quiz-hint">
            提交后所有错题会进入错题本；后续生成相关题目时，后端会把错题记录作为复习材料之一引用。
          </p>
        </section>
      </aside>
    </div>
  );
}

function contextText(conversation: Conversation | null) {
  if (!conversation) {
    return "未绑定当前对话，将使用可用的学习历史、历史小测和错题记录。";
  }
  if (!conversation.documentNames.length) {
    return `当前对话：${conversation.title}`;
  }
  return `当前对话：${conversation.title} · ${conversation.documentNames.join("、")}`;
}

function formatQuizError(error: unknown) {
  const message = error instanceof Error ? error.message : "出题失败。";
  if (/not found/i.test(message)) {
    return "小测验接口未找到。请重启 FastAPI 后端，让新的 quiz API 生效。";
  }
  return message || "出题失败。";
}
