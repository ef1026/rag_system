import { request } from "@/lib/api";
import type { MemoryExtractResponse, UserMemory } from "./types";

export const memoryApi = {
  list: (status?: string) =>
    request<UserMemory[]>(
      `/api/memory${status ? `?status=${encodeURIComponent(status)}` : ""}`,
    ),
  extract: (conversationId?: string) =>
    request<MemoryExtractResponse>("/api/memory/extract", {
      method: "POST",
      body: JSON.stringify({
        ...(conversationId ? { conversation_id: conversationId } : {}),
      }),
    }),
  accept: (memoryId: string) =>
    request<UserMemory>(`/api/memory/${encodeURIComponent(memoryId)}/accept`, {
      method: "POST",
    }),
  dismiss: (memoryId: string) =>
    request<UserMemory>(`/api/memory/${encodeURIComponent(memoryId)}/dismiss`, {
      method: "POST",
    }),
  remove: (memoryId: string) =>
    request<{ ok: boolean }>(`/api/memory/${encodeURIComponent(memoryId)}`, {
      method: "DELETE",
    }),
};
