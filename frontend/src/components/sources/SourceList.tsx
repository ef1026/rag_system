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
            <span>{sourceLabel(source.type)}</span>
            <span>{source.page ? `第 ${source.page} 页` : "页码未识别"}</span>
          </div>
          <p>{source.text}</p>
        </article>
      ))}
    </div>
  );
}
