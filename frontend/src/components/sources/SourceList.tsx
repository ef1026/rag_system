import { sourceLabel } from "@/lib/format";
import type { SourceItem } from "@/types/document";

type SourceListProps = {
  sources: SourceItem[];
};

export function SourceList({ sources }: SourceListProps) {
  if (!sources.length) {
    return <p className="empty-state">解析或问答后，这里会显示可引用片段。</p>;
  }

  return (
    <div className="source-list">
      {sources.map((source) => (
        <article className="source-item" key={source.id}>
          <div className="source-meta">
            {source.document_name || source.document_id ? (
              <span>{source.document_name || source.document_id}</span>
            ) : null}
            <span>{sourceLabel(source.type)}</span>
            <span>{pageLabel(source)}</span>
            {typeof source.rank === "number" ? <span>#{source.rank}</span> : null}
            {source.citation_mode ? (
              <span>{citationModeLabel(source.citation_mode)}</span>
            ) : null}
            {source.score_type ? <span>{scoreTypeLabel(source.score_type)}</span> : null}
            {source.match_method ? <span>{matchMethodLabel(source.match_method)}</span> : null}
          </div>
          <p>{source.text}</p>
        </article>
      ))}
    </div>
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

function matchMethodLabel(method: string) {
  if (method === "metadata") return "元数据命中";
  if (method === "hash") return "哈希命中";
  if (method === "substring") return "片段命中";
  if (method === "fuzzy") return "相似匹配";
  if (method === "fallback") return "片段回退";
  return method;
}
