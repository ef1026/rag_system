import { request } from "@/lib/api";
import type {
  Folder,
  ManagedFile,
  ManagedFileFilters,
  ManagedFilePatch,
  ManagedFileRegister,
  RegisterManagedFileResponse,
  SyncExistingDocumentsResponse,
  TagSummary,
} from "./types";

type OkResponse = {
  ok: boolean;
};

export const fileManagerApi = {
  files: (filters: ManagedFileFilters = {}) =>
    request<ManagedFile[]>(`/api/file-manager/files${toQueryString(filters)}`),
  register: (payload: ManagedFileRegister) =>
    request<RegisterManagedFileResponse>("/api/file-manager/files/register", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updateFile: (fileId: string, payload: ManagedFilePatch) =>
    request<ManagedFile>(`/api/file-manager/files/${encodeURIComponent(fileId)}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  deleteFile: (fileId: string) =>
    request<OkResponse>(`/api/file-manager/files/${encodeURIComponent(fileId)}`, {
      method: "DELETE",
    }),
  folders: () => request<Folder[]>("/api/file-manager/folders"),
  createFolder: (payload: { name: string; parent_id?: string | null }) =>
    request<Folder>("/api/file-manager/folders", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updateFolder: (
    folderId: string,
    payload: { name?: string; parent_id?: string | null; sort_order?: number },
  ) =>
    request<Folder>(`/api/file-manager/folders/${encodeURIComponent(folderId)}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  deleteFolder: (folderId: string) =>
    request<OkResponse>(`/api/file-manager/folders/${encodeURIComponent(folderId)}`, {
      method: "DELETE",
    }),
  tags: () => request<TagSummary[]>("/api/file-manager/tags"),
  syncExistingDocuments: () =>
    request<SyncExistingDocumentsResponse>(
      "/api/file-manager/sync-existing-documents",
      { method: "POST" },
    ),
};

function toQueryString(filters: ManagedFileFilters) {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    if (value === undefined || value === "") return;
    params.set(key, String(value));
  });
  const query = params.toString();
  return query ? `?${query}` : "";
}
