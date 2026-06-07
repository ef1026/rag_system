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
            {source.match_method ? <span>{source.match_method}</span> : null}
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
