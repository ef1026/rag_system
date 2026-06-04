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
  readiness_warnings?: string[];
  text_indexed?: boolean;
  multimodal_enabled?: boolean;
  multimodal_status?: string;
  image_status?: string;
  table_status?: string;
  equation_status?: string;
  formula_status?: string;
  skipped_multimodal_items_count?: number;
  skipped_by_reason?: Record<string, number | undefined>;
  multimodal_warnings_count?: number;
  warnings_summary?: string[];
};

export type ChatRequest = {
  question: string;
  document_id?: string;
  document_ids?: string[];
  level: AnswerLevel;
  mode: string;
  vlm_enhanced?: boolean | "auto";
  conversation_id?: string;
  client_user_message_id?: string;
  use_profile?: boolean;
  use_memory?: boolean;
};

export type ChatDocumentUsed = {
  document_id: string;
  name: string;
  status: DocumentSummary["status"];
};

export type ChatPartialFailure = {
  document_id: string;
  name?: string;
  error: string;
};

export type ImageAssetPublic = {
  image_id: string;
  document_id: string;
  document_name: string;
  url: string;
  caption?: string | null;
  page?: number | null;
  bbox?: unknown;
  source_type?: string | null;
  relevance_reason?: string | null;
};

export type ChatResponse = {
  answer: string;
  sources: SourceItem[];
  related_images?: ImageAssetPublic[];
  inline_image_refs?: string[];
  documents_used?: ChatDocumentUsed[];
  partial_failures?: ChatPartialFailure[];
  conversation_id?: string | null;
  user_message_id?: string | null;
  assistant_message_id?: string | null;
};

export type WarmupResponse = {
  ok: boolean;
  document: DocumentSummary;
  storage_dir: string;
  warmup_seconds: number;
};

export type RAGRetrievalStatus = {
  default_mode: string;
  rerank_requested: boolean;
  rerank_enabled: boolean;
  rerank_model: string | null;
  rerank_provider: string | null;
  rerank_model_loaded: boolean;
  rerank_last_error?: string | null;
  reason?: string | null;
  rerank_top_n?: number | null;
};

export type RAGStatusResponse = {
  parser: string;
  parser_output_dir: string;
  retrieval: RAGRetrievalStatus;
  multimodal: {
    enabled: boolean;
    image_processing: boolean;
    table_processing: boolean;
    equation_processing: boolean;
    formula_processing: boolean;
    vlm_model?: string | null;
  };
  embedding: {
    model: string;
  };
};

export type CacheResponse = {
  ok: boolean;
};
