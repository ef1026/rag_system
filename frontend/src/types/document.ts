export type DocumentStatus = "uploaded" | "processed";

export type DocumentSummary = {
  id: string;
  name: string;
  size: number;
  status: DocumentStatus;
};

export type SourceItem = {
  id: string;
  type: "text" | "equation" | "table" | string;
  page: number | null;
  text: string;
};
