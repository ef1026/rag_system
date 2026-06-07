import { sourceLabel } from "@/lib/format";
import type { SourceItem } from "@/types/document";

type CitationListProps = {
  sources?: SourceItem[];
  onSourceSelect?: (source: SourceItem) => void;
};

export function CitationList({ sources, onSourceSelect }: CitationListProps) {
  if (!sources?.length) return null;

  const fallbackCount = sources.filter(
    (source) => source.citation_mode === "storage_fallback",
  ).length;
  const allFallback = fallbackCount === sources.length;
  const hasFallback = fallbackCount > 0;

  return (
    <section className="message-citations" aria-label="回答检索证据">
      <div className="message-citations-heading">
        <span>
          {allFallback
            ? "检索证据不可用，以下为文档片段回退结果"
            : "本回答参考了以下检索证据"}
        </span>
        <strong>{sources.length}</strong>
      </div>
      {hasFallback && !allFallback ? (
        <p className="citation-note">部分引用为文档片段回退结果。</p>
      ) : null}
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
              {source.citation_mode ? (
                <span>{citationModeLabel(source.citation_mode)}</span>
              ) : null}
              {source.score_type ? <span>{scoreTypeLabel(source.score_type)}</span> : null}
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

function citationModeLabel(mode: string) {
  if (mode === "retrieval_context") return "检索证据";
  if (mode === "storage_fallback") return "回退片段";
  return mode;
}

function scoreTypeLabel(scoreType: string) {
  if (scoreType === "retrieval_rank") return "检索排序";
  if (scoreType === "storage_fallback") return "非检索排序";
  if (scoreType === "rerank") return "重排分";
  if (scoreType === "match") return "匹配分";
  return scoreType;
}
