export type DocumentStatus =
  | "uploaded"
  | "parsed"
  | "indexing"
  | "ready_for_chat"
  | "partial_success"
  | "failed";

export type DocumentSummary = {
  id: string;
  name: string;
  size: number;
  status: DocumentStatus;
  readiness_warnings?: string[];
};

export type SourceItem = {
  id: string;
  type: "text" | "equation" | "table" | string;
  page: number | null;
  page_end?: number | null;
  text: string;
  document_id?: string | null;
  document_name?: string | null;
  chunk_id?: string | null;
  content_index?: number | null;
  rank?: number | null;
  score?: number | null;
  score_type?: string | null;
  match_score?: number | null;
  match_method?: string | null;
  citation_mode?: string | null;
};
