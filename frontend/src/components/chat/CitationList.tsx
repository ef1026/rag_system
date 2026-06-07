import { sourceLabel } from "@/lib/format";
import type { SourceItem } from "@/types/document";

type CitationListProps = {
  sources?: SourceItem[];
  onSourceSelect?: (source: SourceItem) => void;
};

export function CitationList({ sources, onSourceSelect }: CitationListProps) {
  if (!sources?.length) return null;

  return (
    <section className="message-citations" aria-label="回答检索证据">
      <div className="message-citations-heading">
        <span>本回答参考了以下检索证据</span>
        <strong>{sources.length}</strong>
      </div>
      <div className="citation-list">
        {sources.map((source) => (
          <button
            type="button"
            className="citation-card"
            key={source.id}
            disabled={!source.document_id}
            onClick={() => onSourceSelect?.(source)}
          >
            <span className="citation-meta">
              <strong>{source.document_name || source.document_id || "当前文档"}</strong>
              <span>{pageLabel(source)}</span>
              {typeof source.rank === "number" ? <span>#{source.rank}</span> : null}
              <span>{sourceLabel(source.type)}</span>
              {source.match_method ? <span>{matchMethodLabel(source.match_method)}</span> : null}
            </span>
            <span className="citation-text">{source.text}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function pageLabel(source: SourceItem) {
  if (typeof source.page !== "number") return "页码未识别";
  if (
    typeof source.page_end === "number" &&
    source.page_end > source.page
  ) {
    return `第 ${source.page}-${source.page_end} 页`;
  }
  return `第 ${source.page} 页`;
}

function matchMethodLabel(method: string) {
  if (method === "metadata") return "元数据命中";
  if (method === "hash") return "哈希命中";
  if (method === "substring") return "片段命中";
  if (method === "fuzzy") return "相似匹配";
  if (method === "fallback") return "片段回退";
  return method;
}
