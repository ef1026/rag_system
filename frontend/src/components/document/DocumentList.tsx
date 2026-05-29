import { formatBytes } from "@/lib/format";
import type { DocumentStatus, DocumentSummary } from "@/types/document";

type DocumentListProps = {
  documents: DocumentSummary[];
  focusedId: string;
  selectedIds: string[];
  isLoading: boolean;
  onFocus: (id: string) => void;
  onToggle: (document: DocumentSummary, selected: boolean) => void;
};

const statusLabels: Record<DocumentStatus, string> = {
  uploaded: "待解析",
  parsed: "已解析，待索引",
  indexing: "索引中",
  ready_for_chat: "可提问",
  partial_success: "索引不完整，请重新解析",
  failed: "失败",
};

export function DocumentList({
  documents,
  focusedId,
  selectedIds,
  isLoading,
  onFocus,
  onToggle,
}: DocumentListProps) {
  if (isLoading) {
    return <p className="empty-state">正在读取文档...</p>;
  }

  if (!documents.length) {
    return <p className="empty-state">还没有上传 PDF。</p>;
  }

  return (
    <div className="document-list">
      {documents.map((document) => {
        const selected = selectedIds.includes(document.id);
        const ready = document.status === "ready_for_chat";
        const checkboxDisabled = !ready;
        return (
          <article
            className={`document-row ${selected ? "selected" : ""} ${
              document.id === focusedId ? "focused" : ""
            }`}
            key={document.id}
          >
            <input
              className="document-checkbox"
              type="checkbox"
              checked={selected}
              disabled={checkboxDisabled}
              aria-label={`选择文档 ${document.name}`}
              onClick={(event) => event.stopPropagation()}
              onChange={(event) => onToggle(document, event.target.checked)}
            />
            <button
              type="button"
              className="document-main-button"
              onClick={() => onFocus(document.id)}
            >
              <span className="document-main">
                <strong>{document.name}</strong>
                <small>{formatBytes(document.size)}</small>
              </span>
            </button>
            <span className={`document-status status-${document.status}`}>
              {statusLabels[document.status]}
            </span>
          </article>
        );
      })}
    </div>
  );
}
