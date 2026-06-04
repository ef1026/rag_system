"use client";

import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/Button";
import { memoryApi } from "./api";
import type { UserMemory } from "./types";

const statuses = ["candidate", "active", "dismissed"] as const;
const scopeTypes = ["global", "course", "document", "conversation"] as const;

type MemoryDraft = {
  value: string;
  scope_type: UserMemory["scope_type"];
  scope_id: string;
};

export function MemoryReviewPanel() {
  const [memories, setMemories] = useState<UserMemory[]>([]);
  const [drafts, setDrafts] = useState<Record<string, MemoryDraft>>({});
  const [editingId, setEditingId] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isExtracting, setIsExtracting] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    void loadMemories();
  }, []);

  const grouped = useMemo(() => {
    return statuses.map((status) => ({
      status,
      memories: memories.filter((memory) => memory.status === status),
    }));
  }, [memories]);

  async function loadMemories() {
    setIsLoading(true);
    setError("");
    try {
      setMemories(await memoryApi.list());
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Memory loading failed.");
    } finally {
      setIsLoading(false);
    }
  }

  async function extractCandidates() {
    setIsExtracting(true);
    setError("");
    setNotice("");
    try {
      const result = await memoryApi.extract();
      setNotice(`Created or refreshed ${result.created_count} candidate memories.`);
      await loadMemories();
    } catch (nextError) {
      setError(
        nextError instanceof Error ? nextError.message : "Memory extraction failed.",
      );
    } finally {
      setIsExtracting(false);
    }
  }

  function startEditing(memory: UserMemory) {
    setEditingId(memory.id);
    setDrafts((current) => ({
      ...current,
      [memory.id]: {
        value: memory.value,
        scope_type: memory.scope_type,
        scope_id: memory.scope_id || "",
      },
    }));
  }

  function updateDraft(memoryId: string, update: Partial<MemoryDraft>) {
    const defaultDraft: MemoryDraft = {
      value: "",
      scope_type: "global",
      scope_id: "",
    };
    setDrafts((current) => ({
      ...current,
      [memoryId]: {
        ...(current[memoryId] || defaultDraft),
        ...update,
      },
    }));
  }

  async function saveDraft(memory: UserMemory) {
    const draft = drafts[memory.id];
    if (!draft) return;
    await runAction(async () => {
      await memoryApi.patch(memory.id, {
        value: draft.value,
        scope_type: draft.scope_type,
        scope_id: draft.scope_type === "global" ? null : draft.scope_id || null,
      });
      setEditingId("");
    });
  }

  async function runAction(action: () => Promise<unknown>) {
    setError("");
    setNotice("");
    try {
      await action();
      await loadMemories();
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Memory update failed.");
    }
  }

  return (
    <section className="panel memory-review-panel">
      <div className="panel-heading">
        <div>
          <p className="section-label">Learning Memory</p>
          <h2>Review inferred preferences</h2>
        </div>
        <Button
          type="button"
          variant="secondary"
          disabled={isExtracting}
          onClick={() => void extractCandidates()}
        >
          {isExtracting ? "Extracting..." : "Extract candidates"}
        </Button>
      </div>

      {error ? <p className="error-text">{error}</p> : null}
      {notice ? <p className="chat-status">{notice}</p> : null}
      {isLoading ? <p className="empty-state">Loading memory...</p> : null}

      <div className="memory-groups">
        {grouped.map((group) => (
          <section className="memory-group" key={group.status}>
            <div className="memory-group-heading">
              <h3>{labelForStatus(group.status)}</h3>
              <span>{group.memories.length}</span>
            </div>
            {group.memories.length === 0 ? (
              <p className="empty-state">No {group.status} memories.</p>
            ) : (
              <div className="memory-list">
                {group.memories.map((memory) => (
                  <MemoryItem
                    key={memory.id}
                    memory={memory}
                    draft={drafts[memory.id]}
                    isEditing={editingId === memory.id}
                    onEdit={() => startEditing(memory)}
                    onDraftChange={(update) => updateDraft(memory.id, update)}
                    onSave={() => void saveDraft(memory)}
                    onCancel={() => setEditingId("")}
                    onAccept={() => void runAction(() => memoryApi.accept(memory.id))}
                    onDismiss={() => void runAction(() => memoryApi.dismiss(memory.id))}
                    onDelete={() => void runAction(() => memoryApi.remove(memory.id))}
                  />
                ))}
              </div>
            )}
          </section>
        ))}
      </div>
    </section>
  );
}

function MemoryItem({
  memory,
  draft,
  isEditing,
  onEdit,
  onDraftChange,
  onSave,
  onCancel,
  onAccept,
  onDismiss,
  onDelete,
}: {
  memory: UserMemory;
  draft?: MemoryDraft;
  isEditing: boolean;
  onEdit: () => void;
  onDraftChange: (update: Partial<MemoryDraft>) => void;
  onSave: () => void;
  onCancel: () => void;
  onAccept: () => void;
  onDismiss: () => void;
  onDelete: () => void;
}) {
  return (
    <article className="memory-item">
      <div className="memory-item-main">
        <strong>{memory.key || memory.memory_type}</strong>
        {isEditing ? (
          <div className="memory-edit-form">
            <textarea
              value={draft?.value ?? memory.value}
              onChange={(event) => onDraftChange({ value: event.target.value })}
            />
            <div className="memory-scope-row">
              <label>
                <span>Scope</span>
                <select
                  value={draft?.scope_type ?? memory.scope_type}
                  onChange={(event) =>
                    onDraftChange({
                      scope_type: event.target.value as UserMemory["scope_type"],
                    })
                  }
                >
                  {scopeTypes.map((scopeType) => (
                    <option key={scopeType} value={scopeType}>
                      {scopeType}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>Scope id</span>
                <input
                  value={draft?.scope_id ?? memory.scope_id ?? ""}
                  disabled={(draft?.scope_type ?? memory.scope_type) === "global"}
                  onChange={(event) => onDraftChange({ scope_id: event.target.value })}
                />
              </label>
            </div>
          </div>
        ) : (
          <p>{memory.value}</p>
        )}
      </div>
      <dl className="memory-meta">
        <MetaItem label="Type" value={memory.memory_type} />
        <MetaItem label="Confidence" value={`${Math.round(memory.confidence * 100)}%`} />
        <MetaItem label="Scope" value={formatScope(memory)} />
        <MetaItem label="Auto" value={memory.auto_apply ? "on" : "off"} />
        {memory.source_conversation_id ? (
          <MetaItem label="Source" value={memory.source_conversation_id} />
        ) : null}
        {memory.evidence_message_ids.length ? (
          <MetaItem label="Evidence ids" value={memory.evidence_message_ids.join(", ")} />
        ) : null}
      </dl>
      {memory.evidence ? <p className="memory-evidence">{memory.evidence}</p> : null}
      <div className="memory-actions">
        {isEditing ? (
          <>
            <Button type="button" variant="secondary" onClick={onSave}>
              Save draft
            </Button>
            <Button type="button" variant="ghost" onClick={onCancel}>
              Cancel
            </Button>
          </>
        ) : null}
        {memory.status === "candidate" && !isEditing ? (
          <>
            <Button type="button" variant="secondary" onClick={onEdit}>
              Edit
            </Button>
            <Button type="button" variant="secondary" onClick={onAccept}>
              Accept
            </Button>
            <Button type="button" variant="ghost" onClick={onDismiss}>
              Dismiss
            </Button>
          </>
        ) : null}
        <Button type="button" variant="ghost" onClick={onDelete}>
          Delete
        </Button>
      </div>
    </article>
  );
}

function MetaItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd title={value}>{value}</dd>
    </div>
  );
}

function formatScope(memory: UserMemory) {
  return memory.scope_id ? `${memory.scope_type}:${memory.scope_id}` : memory.scope_type;
}

function labelForStatus(status: (typeof statuses)[number]) {
  if (status === "candidate") return "Candidates";
  if (status === "active") return "Active";
  return "Dismissed";
}
