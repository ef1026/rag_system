import { request } from "@/lib/api";
import type {
  QuizSession,
  QuizSubmitAnswer,
  QuizSubmitResponse,
  WrongQuestion,
} from "./types";

export const quizApi = {
  generate: ({
    conversationId,
    documentIds,
  }: {
    conversationId?: string | null;
    documentIds?: string[];
  }) =>
    request<QuizSession>("/api/quizzes/generate", {
      method: "POST",
      body: JSON.stringify({
        conversation_id: conversationId || null,
        document_ids: documentIds || [],
        count: 3,
      }),
    }),
  submit: (sessionId: string, answers: QuizSubmitAnswer[]) =>
    request<QuizSubmitResponse>(
      `/api/quizzes/${encodeURIComponent(sessionId)}/submit`,
      {
        method: "POST",
        body: JSON.stringify({ answers }),
      },
    ),
  wrongQuestions: (reviewed?: boolean | null) => {
    const query =
      reviewed === null || reviewed === undefined
        ? ""
        : `?reviewed=${reviewed ? "true" : "false"}`;
    return request<WrongQuestion[]>(`/api/wrong-questions${query}`);
  },
  markReviewed: (wrongQuestionId: string) =>
    request<WrongQuestion>(
      `/api/wrong-questions/${encodeURIComponent(wrongQuestionId)}/review`,
      { method: "POST" },
    ),
  removeWrongQuestion: (wrongQuestionId: string) =>
    request<{ ok: boolean }>(
      `/api/wrong-questions/${encodeURIComponent(wrongQuestionId)}`,
      { method: "DELETE" },
    ),
};
