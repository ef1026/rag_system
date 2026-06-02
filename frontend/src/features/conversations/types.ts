import type { Conversation, ChatMessage, RelatedImage } from "@/types/chat";
import type { SourceItem } from "@/types/document";

export type ApiConversation = {
  id: string;
  profile_id: string;
  title: string;
  document_ids: string[];
  level: string;
  mode: string;
  chat_mode: string;
  file_context: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type ApiConversationCreate = {
  id?: string;
  title?: string;
  document_ids?: string[];
  level?: string;
  mode?: string;
  chat_mode?: string;
  file_context?: Record<string, unknown>;
};

export type ApiConversationPatch = {
  title?: string;
  document_ids?: string[];
  level?: string;
  mode?: string;
  chat_mode?: string;
  file_context?: Record<string, unknown>;
};

export type ApiConversationMessageImport = {
  id?: string;
  role: "user" | "assistant";
  content: string;
  status?: string;
  document_ids?: string[];
  level?: string;
  mode?: string;
  chat_mode?: string;
  sources?: SourceItem[];
  related_images?: ApiRelatedImage[];
  inline_image_refs?: string[];
  error?: string;
  created_at?: string;
};

export type ApiConversationImportItem = ApiConversationCreate & {
  id: string;
  title: string;
  messages: ApiConversationMessageImport[];
  created_at?: string;
  updated_at?: string;
};

export type ApiConversationImportResponse = {
  imported_count: number;
  message_count: number;
  conversations: ApiConversation[];
};

export type ApiRelatedImage = {
  image_id: string;
  document_id: string;
  document_name: string;
  url: string;
  caption?: string;
  page?: number | null;
  bbox?: unknown;
  source_type?: string;
  relevance_reason?: string;
};

export function conversationToCreate(
  conversation: Conversation,
): ApiConversationCreate {
  return {
    id: conversation.id,
    title: conversation.title,
    document_ids: conversation.documentIds,
    level: conversation.level,
    mode: conversation.mode,
    chat_mode: conversation.chatMode,
  };
}

export function conversationToImport(
  conversation: Conversation,
): ApiConversationImportItem {
  return {
    ...conversationToCreate(conversation),
    id: conversation.id,
    title: conversation.title,
    messages: conversation.messages.map(messageToImport),
    created_at: conversation.createdAt,
    updated_at: conversation.updatedAt,
  };
}

function messageToImport(message: ChatMessage): ApiConversationMessageImport {
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    status: message.status || "sent",
    document_ids: message.documentIds,
    level: message.level,
    mode: message.mode,
    chat_mode: message.chatMode,
    sources: message.sources,
    related_images: message.relatedImages?.map(relatedImageToApi),
    inline_image_refs: message.inlineImageRefs,
    error: message.error,
    created_at: message.createdAt,
  };
}

function relatedImageToApi(image: RelatedImage): ApiRelatedImage {
  return {
    image_id: image.imageId,
    document_id: image.documentId,
    document_name: image.documentName,
    url: image.url,
    ...(image.caption ? { caption: image.caption } : {}),
    ...(image.page !== undefined ? { page: image.page } : {}),
    ...(image.bbox !== undefined ? { bbox: image.bbox } : {}),
    ...(image.sourceType ? { source_type: image.sourceType } : {}),
    ...(image.relevanceReason ? { relevance_reason: image.relevanceReason } : {}),
  };
}
