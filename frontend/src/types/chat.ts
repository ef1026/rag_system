import type { SourceItem } from "./document";

export type AnswerLevel = "beginner" | "undergraduate" | "expert";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceItem[];
};
