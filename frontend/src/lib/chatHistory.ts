import type { AnswerLevel, ChatMessage, ChatMessageStatus } from "@/types/chat";
import type { SourceItem } from "@/types/document";

const CHAT_HISTORY_PREFIX = "rag-chat-history:v1:";

export const CHAT_HISTORY_LIMIT = 100;

export function getChatHistoryKey(documentId: string) {
  return `${CHAT_HISTORY_PREFIX}${documentId}`;
}

export function trimChatHistory(messages: ChatMessage[]) {
  return messages.slice(-CHAT_HISTORY_LIMIT);
}

export function loadChatHistory(documentId: string) {
  const storage = getStorage();
  if (!storage || !documentId) return [];

  const rawHistory = storage.getItem(getChatHistoryKey(documentId));
  if (!rawHistory) return [];

  try {
    const parsedHistory = JSON.parse(rawHistory);
    if (!Array.isArray(parsedHistory)) return [];

    return trimChatHistory(
      parsedHistory
        .map((message) => normalizeChatMessage(message, documentId))
        .filter((message): message is ChatMessage => message !== null),
    );
  } catch {
    return [];
  }
}

export function saveChatHistory(documentId: string, messages: ChatMessage[]) {
  const storage = getStorage();
  if (!storage || !documentId) return;

  try {
    storage.setItem(
      getChatHistoryKey(documentId),
      JSON.stringify(trimChatHistory(messages)),
    );
  } catch {
    // localStorage can be unavailable or full; the in-memory chat should still work.
  }
}

export function clearChatHistory(documentId: string) {
  const storage = getStorage();
  if (!storage || !documentId) return;

  storage.removeItem(getChatHistoryKey(documentId));
}

function normalizeChatMessage(value: unknown, documentId: string): ChatMessage | null {
  if (!isRecord(value)) return null;

  const role = value.role;
  const level = value.level;
  const messageDocumentId = value.documentId;
  if (
    typeof value.id !== "string" ||
    (role !== "user" && role !== "assistant") ||
    typeof value.content !== "string" ||
    typeof value.createdAt !== "string" ||
    typeof messageDocumentId !== "string" ||
    messageDocumentId !== documentId ||
    typeof value.mode !== "string" ||
    !isAnswerLevel(level)
  ) {
    return null;
  }

  const status = normalizeStatus(value.status);
  const sources = Array.isArray(value.sources)
    ? (value.sources as SourceItem[])
    : undefined;

  return {
    id: value.id,
    role,
    content: value.content,
    createdAt: value.createdAt,
    documentId: messageDocumentId,
    mode: value.mode,
    level,
    ...(status ? { status } : {}),
    ...(sources ? { sources } : {}),
  };
}

function normalizeStatus(value: unknown): ChatMessageStatus | undefined {
  return value === "sent" || value === "failed" ? value : undefined;
}

function isAnswerLevel(value: unknown): value is AnswerLevel {
  return (
    value === "beginner" ||
    value === "undergraduate" ||
    value === "expert" ||
    value === "custom"
  );
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
