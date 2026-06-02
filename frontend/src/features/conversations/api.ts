import { request } from "@/lib/api";
import type { Conversation } from "@/types/chat";
import {
  conversationToCreate,
  conversationToImport,
  type ApiConversation,
  type ApiConversationImportResponse,
  type ApiConversationPatch,
} from "./types";

export const conversationsApi = {
  list: () => request<ApiConversation[]>("/api/conversations"),
  create: (conversation: Conversation) =>
    request<ApiConversation>("/api/conversations", {
      method: "POST",
      body: JSON.stringify(conversationToCreate(conversation)),
    }),
  patch: (conversationId: string, payload: ApiConversationPatch) =>
    request<ApiConversation>(`/api/conversations/${encodeURIComponent(conversationId)}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  remove: (conversationId: string) =>
    request<{ ok: boolean }>(`/api/conversations/${encodeURIComponent(conversationId)}`, {
      method: "DELETE",
    }),
  clearMessages: (conversationId: string) =>
    request<{ ok: boolean }>(
      `/api/conversations/${encodeURIComponent(conversationId)}/messages`,
      { method: "DELETE" },
    ),
  importConversations: (conversations: Conversation[]) =>
    request<ApiConversationImportResponse>("/api/conversations/import", {
      method: "POST",
      body: JSON.stringify({
        conversations: conversations.map(conversationToImport),
      }),
    }),
};
