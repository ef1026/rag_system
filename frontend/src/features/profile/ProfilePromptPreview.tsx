import type { UserMemory } from "@/features/memory/types";

type ProfilePromptPreviewProps = {
  promptContext: string;
  retrievalHints: string[];
  usedProfileFields: string[];
  usedMemories: UserMemory[];
  warnings: string[];
  isLoading: boolean;
};

export function ProfilePromptPreview({
  promptContext,
  retrievalHints,
  usedProfileFields,
  usedMemories,
  warnings,
  isLoading,
}: ProfilePromptPreviewProps) {
  return (
    <section className="panel profile-preview">
      <div className="panel-heading">
        <div>
          <p className="section-label">提示词预览</p>
          <h2>个性化上下文</h2>
        </div>
      </div>
      <pre className="prompt-preview-text">
        {isLoading ? "正在加载画像上下文..." : promptContext}
      </pre>
      <div className="personalization-preview-grid">
        <PreviewList title="已使用画像字段" items={usedProfileFields} />
        <PreviewList title="检索提示" items={retrievalHints} />
        <PreviewList
          title="本次使用的记忆"
          items={usedMemories.map((memory) =>
            `${memory.memory_type}${memory.key ? `:${memory.key}` : ""} (${formatScope(memory)})`,
          )}
        />
        <PreviewList title="警告" items={warnings} muted />
      </div>
    </section>
  );
}

function PreviewList({
  title,
  items,
  muted,
}: {
  title: string;
  items: string[];
  muted?: boolean;
}) {
  return (
    <section className="preview-list">
      <h3>{title}</h3>
      {items.length ? (
        <ul className={muted ? "muted-list" : undefined}>
          {items.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : (
        <p className="empty-state compact">暂无</p>
      )}
    </section>
  );
}

function formatScope(memory: UserMemory) {
  const scopeLabel = {
    global: "全局",
    course: "课程",
    document: "文档",
    conversation: "对话",
  }[memory.scope_type];
  return memory.scope_id ? `${scopeLabel}:${memory.scope_id}` : scopeLabel;
}
