import type {
  ApiHealth,
  CacheResponse,
  ChatRequest,
  ChatResponse,
  ProcessResponse,
  RAGStatusResponse,
  UploadResponse,
  WarmupResponse,
} from "@/types/api";
import type { DocumentSummary } from "@/types/document";
import { parseApiError } from "./errors";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "http://127.0.0.1:8000";

export function getApiUrl(pathOrUrl: string) {
  if (/^https?:\/\//i.test(pathOrUrl)) return pathOrUrl;
  const normalizedPath = pathOrUrl.startsWith("/") ? pathOrUrl : `/${pathOrUrl}`;
  return `${API_BASE_URL}${normalizedPath}`;
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(getApiUrl(path), {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    if (process.env.NODE_ENV !== "production") {
      const body = await response.clone().text();
      console.debug("api error response", {
        path,
        status: response.status,
        body,
      });
    }
    throw await parseApiError(response);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<ApiHealth>("/api/health"),
  ragStatus: () => request<RAGStatusResponse>("/api/rag/status"),
  documents: () => request<DocumentSummary[]>("/api/documents"),
  upload: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return request<UploadResponse>("/api/upload", {
      method: "POST",
      body: formData,
    });
  },
  process: (documentId: string) =>
    request<ProcessResponse>(
      `/api/documents/${encodeURIComponent(documentId)}/process`,
      { method: "POST" },
    ),
  warmup: (documentId: string) =>
    request<WarmupResponse>(
      `/api/documents/${encodeURIComponent(documentId)}/warmup`,
      { method: "POST" },
    ),
  chat: (payload: ChatRequest) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  clearRuntime: () =>
    request<CacheResponse>("/api/cache", { method: "DELETE" }),
};
