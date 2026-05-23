import { formatBytes } from "@/lib/format";
import type { DocumentSummary } from "@/types/document";

type DocumentListProps = {
  documents: DocumentSummary[];
  selectedId: string;
  isLoading: boolean;
  onSelect: (id: string) => void;
};

export function DocumentList({
  documents,
  selectedId,
  isLoading,
  onSelect,
}: DocumentListProps) {
  if (isLoading) {
    return <p className="empty-state">正在读取文档...</p>;
  }

  if (!documents.length) {
    return <p className="empty-state">还没有上传 PDF。</p>;
  }

  return (
    <div className="document-list">
      {documents.map((document) => (
        <button
          type="button"
          className={`document-row ${document.id === selectedId ? "selected" : ""}`}
          key={document.id}
          onClick={() => onSelect(document.id)}
        >
          <span className="document-icon">PDF</span>
          <span className="document-main">
            <strong>{document.name}</strong>
            <small>{formatBytes(document.size)}</small>
          </span>
          <span className={`document-status status-${document.status}`}>
            {document.status === "processed" ? "已解析" : "待解析"}
          </span>
        </button>
      ))}
    </div>
  );
}
