import { request } from "@/lib/api";
import type {
  MemoryCandidateSource,
  MemoryExtractRequest,
  MemoryExtractResponse,
  UserMemory,
  UserMemoryPatch,
} from "./types";

export const memoryApi = {
  list: (status?: string) =>
    request<UserMemory[]>(
      `/api/memory${status ? `?status=${encodeURIComponent(status)}` : ""}`,
    ),
  sources: () => request<MemoryCandidateSource[]>("/api/memory/sources"),
  extract: (payload: MemoryExtractRequest = {}) =>
    request<MemoryExtractResponse>("/api/memory/extract", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  accept: (memoryId: string) =>
    request<UserMemory>(`/api/memory/${encodeURIComponent(memoryId)}/accept`, {
      method: "POST",
    }),
  patch: (memoryId: string, payload: UserMemoryPatch) =>
    request<UserMemory>(`/api/memory/${encodeURIComponent(memoryId)}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
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
