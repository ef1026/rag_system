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
  text: string;
};
