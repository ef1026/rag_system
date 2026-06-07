import { request } from "@/lib/api";
import type { ImageAssetPublic } from "@/types/api";
import type { RelatedImage } from "@/types/chat";
import type {
  QuizSession,
  QuizSubmitAnswer,
  QuizSubmitResponse,
  QuizQuestion,
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
    request<ApiQuizSession>("/api/quizzes/generate", {
      method: "POST",
      body: JSON.stringify({
        conversation_id: conversationId || null,
        document_ids: documentIds || [],
        count: 3,
      }),
    }).then(normalizeQuizSession),
  submit: (sessionId: string, answers: QuizSubmitAnswer[]) =>
    request<ApiQuizSubmitResponse>(
      `/api/quizzes/${encodeURIComponent(sessionId)}/submit`,
      {
        method: "POST",
        body: JSON.stringify({ answers }),
      },
    ).then(normalizeQuizSubmitResponse),
  wrongQuestions: (reviewed?: boolean | null) => {
    const query =
      reviewed === null || reviewed === undefined
        ? ""
        : `?reviewed=${reviewed ? "true" : "false"}`;
    return request<ApiWrongQuestion[]>(`/api/wrong-questions${query}`).then((items) =>
      items.map(normalizeWrongQuestion),
    );
  },
  markReviewed: (wrongQuestionId: string) =>
    request<ApiWrongQuestion>(
      `/api/wrong-questions/${encodeURIComponent(wrongQuestionId)}/review`,
      { method: "POST" },
    ).then(normalizeWrongQuestion),
  removeWrongQuestion: (wrongQuestionId: string) =>
    request<{ ok: boolean }>(
      `/api/wrong-questions/${encodeURIComponent(wrongQuestionId)}`,
      { method: "DELETE" },
    ),
};

type ApiQuizQuestion = Omit<QuizQuestion, "related_images"> & {
  related_images?: ImageAssetPublic[];
};

type ApiQuizSession = Omit<QuizSession, "questions"> & {
  questions: ApiQuizQuestion[];
};

type ApiQuizQuestionResult = Omit<QuizSubmitResponse["results"][number], "question"> & {
  question: ApiQuizQuestion;
};

type ApiQuizSubmitResponse = Omit<QuizSubmitResponse, "results"> & {
  results: ApiQuizQuestionResult[];
};

type ApiWrongQuestion = Omit<WrongQuestion, "related_images"> & {
  related_images?: ImageAssetPublic[];
};

function normalizeQuizSession(session: ApiQuizSession): QuizSession {
  return {
    ...session,
    questions: session.questions.map(normalizeQuizQuestion),
  };
}

function normalizeQuizSubmitResponse(response: ApiQuizSubmitResponse): QuizSubmitResponse {
  return {
    ...response,
    results: response.results.map((result) => ({
      ...result,
      question: normalizeQuizQuestion(result.question),
    })),
  };
}

function normalizeWrongQuestion(question: ApiWrongQuestion): WrongQuestion {
  return {
    ...question,
    related_images: mapRelatedImages(question.related_images) || [],
  };
}

function normalizeQuizQuestion(question: ApiQuizQuestion): QuizQuestion {
  return {
    ...question,
    related_images: mapRelatedImages(question.related_images) || [],
  };
}

function mapRelatedImages(images: ImageAssetPublic[] | undefined): RelatedImage[] | undefined {
  if (!images?.length) return undefined;

  const relatedImages = images
    .filter((image) => image.image_id && image.document_id && image.url)
    .map((image) => ({
      imageId: image.image_id,
      documentId: image.document_id,
      documentName: image.document_name || image.document_id,
      url: image.url,
      ...(image.caption ? { caption: image.caption } : {}),
      ...(typeof image.page === "number" ? { page: image.page } : {}),
      ...(image.bbox !== undefined ? { bbox: image.bbox } : {}),
      ...(image.source_type ? { sourceType: image.source_type } : {}),
      ...(image.relevance_reason
        ? { relevanceReason: image.relevance_reason }
        : {}),
    }));

  return relatedImages.length ? relatedImages : undefined;
}
