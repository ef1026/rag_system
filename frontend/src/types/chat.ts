import type { SourceItem } from "./document";

export type AnswerLevel = "beginner" | "undergraduate" | "expert" | "custom";
export type ChatMode = "multimodal" | "fast_text";
export type ChatMessageStatus = "sent" | "failed";

export type RelatedImage = {
  imageId: string;
  documentId: string;
  documentName: string;
  url: string;
  caption?: string;
  page?: number | null;
  bbox?: unknown;
  sourceType?: string;
  relevanceReason?: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
  documentId?: string;
  documentIds?: string[];
  mode?: string;
  chatMode?: ChatMode;
  level?: AnswerLevel;
  relatedImages?: RelatedImage[];
  inlineImageRefs?: string[];
  error?: string;
  status?: ChatMessageStatus;
  sources?: SourceItem[];
};

export type Conversation = {
  id: string;
  title: string;
  documentIds: string[];
  documentNames: string[];
  mode: string;
  chatMode: ChatMode;
  level: AnswerLevel;
  useProfile: boolean;
  useMemory: boolean;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
};
