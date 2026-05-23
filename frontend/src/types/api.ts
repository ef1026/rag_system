import type { AnswerLevel } from "./chat";
import type { DocumentSummary, SourceItem } from "./document";

export type ApiHealth = {
  ok: boolean;
  has_api_key: boolean;
  mineru_backend: string;
  mineru_device: string;
};

export type UploadResponse = {
  document: DocumentSummary;
};

export type ProcessResponse = {
  document: DocumentSummary;
  message: string;
  sources: SourceItem[];
};

export type ChatRequest = {
  question: string;
  document_id?: string;
  level: AnswerLevel;
  mode: string;
};

export type ChatResponse = {
  answer: string;
  sources: SourceItem[];
};
