import type {
  AnswerLevel,
  ChatMessage,
  ChatMessageStatus,
  ChatMode,
  Conversation,
  RelatedImage,
} from "@/types/chat";
import type { SourceItem } from "@/types/document";

export const CONVERSATIONS_STORAGE_KEY = "rag-conversations:v1";
export const MAX_CONVERSATIONS = 50;
export const MAX_MESSAGES_PER_CONVERSATION = 100;

export function loadConversations() {
  const storage = getStorage();
  if (!storage) return [];

  const rawConversations = storage.getItem(CONVERSATIONS_STORAGE_KEY);
  if (!rawConversations) return [];

  try {
    const parsedConversations = JSON.parse(rawConversations);
    if (!Array.isArray(parsedConversations)) return [];

    return trimConversations(
      parsedConversations
        .map(normalizeConversation)
        .filter((conversation): conversation is Conversation => conversation !== null),
    );
  } catch {
    return [];
  }
}

export function saveConversations(conversations: Conversation[]) {
  const storage = getStorage();
  if (!storage) return;

  try {
    storage.setItem(
      CONVERSATIONS_STORAGE_KEY,
      JSON.stringify(trimConversations(conversations)),
    );
  } catch {
    // localStorage can be unavailable or full; the in-memory state should continue.
  }
}

export function trimConversations(conversations: Conversation[]) {
  return [...conversations]
    .map((conversation) => ({
      ...conversation,
      messages: trimConversationMessages(conversation.messages),
    }))
    .sort(
      (left, right) =>
        new Date(right.updatedAt).getTime() - new Date(left.updatedAt).getTime(),
    )
    .slice(0, MAX_CONVERSATIONS);
}

export function trimConversationMessages(messages: ChatMessage[]) {
  return messages.slice(-MAX_MESSAGES_PER_CONVERSATION);
}

export function createConversationTitle(question: string) {
  const normalizedQuestion = question.trim().replace(/\s+/g, " ");
  const title = Array.from(normalizedQuestion).slice(0, 20).join("");
  return title || "新对话";
}

function normalizeConversation(value: unknown): Conversation | null {
  if (!isRecord(value)) return null;

  const level = normalizeLevel(value.level);
  const chatMode = normalizeChatMode(value.chatMode);
  if (
    typeof value.id !== "string" ||
    typeof value.title !== "string" ||
    !Array.isArray(value.documentIds) ||
    !Array.isArray(value.documentNames) ||
    typeof value.mode !== "string" ||
    !level ||
    !chatMode ||
    !Array.isArray(value.messages) ||
    typeof value.createdAt !== "string" ||
    typeof value.updatedAt !== "string"
  ) {
    return null;
  }

  const documentIds = value.documentIds.filter(
    (documentId): documentId is string => typeof documentId === "string",
  );
  const documentNames = value.documentNames.filter(
    (documentName): documentName is string => typeof documentName === "string",
  );
  const messages = value.messages
    .map(normalizeChatMessage)
    .filter((message): message is ChatMessage => message !== null);

  return {
    id: value.id,
    title: value.title || "新对话",
    documentIds,
    documentNames,
    mode: value.mode,
    chatMode,
    level,
    messages: trimConversationMessages(messages),
    createdAt: value.createdAt,
    updatedAt: value.updatedAt,
  };
}

function normalizeChatMessage(value: unknown): ChatMessage | null {
  if (!isRecord(value)) return null;

  const role = value.role;
  if (
    typeof value.id !== "string" ||
    (role !== "user" && role !== "assistant") ||
    typeof value.content !== "string" ||
    typeof value.createdAt !== "string"
  ) {
    return null;
  }

  const documentId = typeof value.documentId === "string" ? value.documentId : undefined;
  const documentIds = Array.isArray(value.documentIds)
    ? value.documentIds.filter(
        (documentId): documentId is string => typeof documentId === "string",
      )
    : undefined;
  const mode = typeof value.mode === "string" ? value.mode : undefined;
  const level = normalizeLevel(value.level);
  const chatMode = normalizeChatMode(value.chatMode);
  const status = normalizeStatus(value.status);
  const relatedImages = normalizeRelatedImages(value.relatedImages);
  const inlineImageRefs = normalizeInlineImageRefs(value.inlineImageRefs);
  const error = typeof value.error === "string" ? value.error : undefined;
  const sources = Array.isArray(value.sources) ? (value.sources as SourceItem[]) : undefined;

  return {
    id: value.id,
    role,
    content: value.content,
    createdAt: value.createdAt,
    ...(documentId ? { documentId } : {}),
    ...(documentIds ? { documentIds } : {}),
    ...(mode ? { mode } : {}),
    ...(chatMode ? { chatMode } : {}),
    ...(level ? { level } : {}),
    ...(relatedImages ? { relatedImages } : {}),
    ...(inlineImageRefs ? { inlineImageRefs } : {}),
    ...(error ? { error } : {}),
    ...(status ? { status } : {}),
    ...(sources ? { sources } : {}),
  };
}

function normalizeRelatedImages(value: unknown): RelatedImage[] | undefined {
  if (!Array.isArray(value)) return undefined;

  const images = value
    .map((image): RelatedImage | null => {
      if (!isRecord(image) || typeof image.url !== "string") return null;

      const imageId =
        typeof image.imageId === "string"
          ? image.imageId
          : typeof image.image_id === "string"
            ? image.image_id
            : typeof image.id === "string"
              ? image.id
              : "";
      const documentId =
        typeof image.documentId === "string"
          ? image.documentId
          : typeof image.document_id === "string"
            ? image.document_id
            : "";
      if (!imageId || !documentId) return null;
      const documentName =
        typeof image.documentName === "string"
          ? image.documentName
          : typeof image.document_name === "string"
            ? image.document_name
            : documentId;

      const caption =
        typeof image.caption === "string"
          ? image.caption
          : typeof image.alt === "string"
            ? image.alt
            : undefined;
      const page =
        typeof image.page === "number" || image.page === null ? image.page : undefined;
      const relevanceReason =
        typeof image.relevanceReason === "string"
          ? image.relevanceReason
          : typeof image.relevance_reason === "string"
            ? image.relevance_reason
            : undefined;
      const sourceType =
        typeof image.sourceType === "string"
          ? image.sourceType
          : typeof image.source_type === "string"
            ? image.source_type
            : undefined;

      return {
        imageId,
        documentId,
        documentName,
        url: image.url,
        ...(caption ? { caption } : {}),
        ...(page !== undefined ? { page } : {}),
        ...(image.bbox !== undefined ? { bbox: image.bbox } : {}),
        ...(sourceType ? { sourceType } : {}),
        ...(relevanceReason ? { relevanceReason } : {}),
      };
    })
    .filter((image): image is RelatedImage => image !== null);

  return images.length ? images : undefined;
}

function normalizeInlineImageRefs(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const refs = value.filter(
    (imageId): imageId is string => typeof imageId === "string" && imageId.length > 0,
  );
  return refs.length ? refs : undefined;
}

function normalizeStatus(value: unknown): ChatMessageStatus | undefined {
  return value === "sent" || value === "failed" ? value : undefined;
}

function normalizeLevel(value: unknown): AnswerLevel | undefined {
  if (
    value === "beginner" ||
    value === "undergraduate" ||
    value === "expert" ||
    value === "custom"
  ) {
    return value;
  }
  return undefined;
}

function normalizeChatMode(value: unknown): ChatMode | undefined {
  if (value === "multimodal" || value === "fast_text") {
    return value;
  }
  return undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function getStorage() {
  if (typeof window === "undefined") return null;

  try {
    return window.localStorage;
  } catch {
    return null;
  }
}
