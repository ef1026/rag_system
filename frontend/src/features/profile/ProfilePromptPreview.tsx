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
          <p className="section-label">Prompt Preview</p>
          <h2>Personalization context</h2>
        </div>
      </div>
      <pre className="prompt-preview-text">
        {isLoading ? "Loading profile context..." : promptContext}
      </pre>
      <div className="personalization-preview-grid">
        <PreviewList title="Profile fields" items={usedProfileFields} />
        <PreviewList title="Retrieval hints" items={retrievalHints} />
        <PreviewList
          title="Used memories"
          items={usedMemories.map((memory) =>
            `${memory.memory_type}${memory.key ? `:${memory.key}` : ""} (${memory.scope_type}${memory.scope_id ? `:${memory.scope_id}` : ""})`,
          )}
        />
        <PreviewList title="Warnings" items={warnings} muted />
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
        <p className="empty-state compact">None</p>
      )}
    </section>
  );
}
