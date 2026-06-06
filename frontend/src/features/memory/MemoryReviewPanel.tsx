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
      setError(nextError instanceof Error ? nextError.message : "记忆加载失败。");
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
      setNotice(`已创建或刷新 ${result.created_count} 条候选记忆。`);
      await loadMemories();
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "候选记忆提取失败。");
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
      setError(nextError instanceof Error ? nextError.message : "记忆更新失败。");
    }
  }

  return (
    <section className="panel memory-review-panel">
      <div className="panel-heading">
        <div>
          <p className="section-label">学习记忆</p>
          <h2>审核候选记忆</h2>
        </div>
        <Button
          type="button"
          variant="secondary"
          disabled={isExtracting}
          onClick={() => void extractCandidates()}
        >
          {isExtracting ? "提取中..." : "提取候选"}
        </Button>
      </div>

      {error ? <p className="error-text">{error}</p> : null}
      {notice ? <p className="chat-status">{notice}</p> : null}
      {isLoading ? <p className="empty-state">正在加载记忆...</p> : null}

      <div className="memory-groups">
        {grouped.map((group) => (
          <section className="memory-group" key={group.status}>
            <div className="memory-group-heading">
              <h3>{labelForStatus(group.status)}</h3>
              <span>{group.memories.length}</span>
            </div>
            {group.memories.length === 0 ? (
              <p className="empty-state">暂无{labelForStatus(group.status)}记忆。</p>
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
                <span>范围</span>
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
                      {scopeLabel(scopeType)}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>范围 ID</span>
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
        <MetaItem label="类型" value={memory.memory_type} />
        <MetaItem label="置信度" value={`${Math.round(memory.confidence * 100)}%`} />
        <MetaItem label="范围" value={formatScope(memory)} />
        <MetaItem label="自动应用" value={memory.auto_apply ? "开" : "关"} />
        {memory.source_conversation_id ? (
          <MetaItem label="来源对话" value={memory.source_conversation_id} />
        ) : null}
        {memory.evidence_message_ids.length ? (
          <MetaItem label="证据消息" value={memory.evidence_message_ids.join(", ")} />
        ) : null}
      </dl>
      {memory.evidence ? <p className="memory-evidence">{memory.evidence}</p> : null}
      <div className="memory-actions">
        {isEditing ? (
          <>
            <Button type="button" variant="secondary" onClick={onSave}>
              保存草稿
            </Button>
            <Button type="button" variant="ghost" onClick={onCancel}>
              取消
            </Button>
          </>
        ) : null}
        {memory.status === "candidate" && !isEditing ? (
          <>
            <Button type="button" variant="secondary" onClick={onEdit}>
              编辑
            </Button>
            <Button type="button" variant="secondary" onClick={onAccept}>
              接受
            </Button>
            <Button type="button" variant="ghost" onClick={onDismiss}>
              拒绝
            </Button>
          </>
        ) : null}
        <Button type="button" variant="ghost" onClick={onDelete}>
          删除
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
  return memory.scope_id
    ? `${scopeLabel(memory.scope_type)}:${memory.scope_id}`
    : scopeLabel(memory.scope_type);
}

function scopeLabel(scopeType: UserMemory["scope_type"]) {
  if (scopeType === "course") return "课程";
  if (scopeType === "document") return "文档";
  if (scopeType === "conversation") return "对话";
  return "全局";
}

function labelForStatus(status: (typeof statuses)[number]) {
  if (status === "candidate") return "候选";
  if (status === "active") return "已启用";
  return "已拒绝";
}
