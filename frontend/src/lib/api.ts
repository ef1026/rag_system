import type {
  ApiHealth,
  ChatRequest,
  ChatResponse,
  ProcessResponse,
  UploadResponse,
} from "@/types/api";
import type { DocumentSummary } from "@/types/document";
import { parseApiError } from "./errors";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    throw await parseApiError(response);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<ApiHealth>("/api/health"),
  documents: () => request<DocumentSummary[]>("/api/documents"),
  upload: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return request<UploadResponse>("/api/documents/upload", {
      method: "POST",
      body: formData,
    });
  },
  process: (documentId: string) =>
    request<ProcessResponse>(
      `/api/documents/${encodeURIComponent(documentId)}/process`,
      { method: "POST" },
    ),
  chat: (payload: ChatRequest) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  clearRuntime: () =>
    request<{ ok: boolean }>("/api/runtime/clear", { method: "POST" }),
};
