"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/Button";
import { quizApi } from "@/features/quiz/api";
import type { WrongQuestion } from "@/features/quiz/types";
import { request } from "@/lib/api";
import { memoryApi } from "./api";
import type { MemoryCandidateSource, UserMemory } from "./types";

const statuses = ["active", "dismissed"] as const;
const scopeTypes = ["global", "course", "document", "conversation"] as const;
const sourceFilters = ["all", "conversation", "wrong_question"] as const;

type MemoryDraft = {
  value: string;
  scope_type: UserMemory["scope_type"];
  scope_id: string;
};

type SourceFilter = (typeof sourceFilters)[number];

type ConversationMessageDetail = {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
};

export function MemoryReviewPanel() {
  const memoryHistoryRef = useRef<HTMLDivElement | null>(null);
  const [memories, setMemories] = useState<UserMemory[]>([]);
  const [sources, setSources] = useState<MemoryCandidateSource[]>([]);
  const [selectedSourceKeys, setSelectedSourceKeys] = useState<Record<string, boolean>>({});
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>("all");
  const [detailSource, setDetailSource] = useState<MemoryCandidateSource | null>(null);
  const [detailMessages, setDetailMessages] = useState<ConversationMessageDetail[]>([]);
  const [detailWrongQuestion, setDetailWrongQuestion] = useState<WrongQuestion | null>(null);
  const [detailError, setDetailError] = useState("");
  const [drafts, setDrafts] = useState<Record<string, MemoryDraft>>({});
  const [editingId, setEditingId] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingSources, setIsLoadingSources] = useState(true);
  const [isExtracting, setIsExtracting] = useState(false);
  const [isDetailLoading, setIsDetailLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    void loadMemories();
    void loadSources();
  }, []);

  const grouped = useMemo(() => {
    return statuses.map((status) => ({
      status,
      memories: dedupeMemories(memories.filter((memory) => memory.status === status)),
    }));
  }, [memories]);

  const uniqueSources = useMemo(() => dedupeSources(sources), [sources]);

  const selectedSources = useMemo(
    () => uniqueSources.filter((source) => selectedSourceKeys[sourceKey(source)]),
    [selectedSourceKeys, uniqueSources],
  );

  const selectedCount = selectedSources.length;
  const allSourcesSelected =
    uniqueSources.length > 0 && selectedCount === uniqueSources.length;

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

  async function loadSources() {
    setIsLoadingSources(true);
    setError("");
    try {
      const nextSources = await memoryApi.sources();
      setSources(nextSources);
      setSelectedSourceKeys((current) => mergeSelectedSources(current, nextSources));
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "候选来源加载失败。");
    } finally {
      setIsLoadingSources(false);
    }
  }

  async function extractCandidates() {
    if (!selectedSources.length) {
      setError("请先选择至少一条聊天记录或错题。");
      return;
    }
    setIsExtracting(true);
    setError("");
    setNotice("");
    try {
      const result = await memoryApi.extract({
        conversation_ids: selectedSources
          .filter((source) => source.source_type === "conversation")
          .map((source) => source.id),
        wrong_question_ids: selectedSources
          .filter((source) => source.source_type === "wrong_question")
          .map((source) => source.id),
        limit: Math.max(50, selectedSources.length * 4),
        activate: true,
      });
      setNotice(`已启用 ${result.created_count} 条记忆。`);
      await loadMemories();
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "候选记忆提取失败。");
    } finally {
      setIsExtracting(false);
    }
  }

  function toggleSource(source: MemoryCandidateSource) {
    const key = sourceKey(source);
    setSelectedSourceKeys((current) => ({
      ...current,
      [key]: !current[key],
    }));
  }

  function toggleAllSources() {
    setSelectedSourceKeys(
      Object.fromEntries(
        uniqueSources.map((source) => [sourceKey(source), !allSourcesSelected]),
      ),
    );
  }

  async function openSourceDetail(source: MemoryCandidateSource) {
    setDetailSource(source);
    setDetailMessages([]);
    setDetailWrongQuestion(null);
    setDetailError("");
    setIsDetailLoading(true);
    try {
      if (source.source_type === "conversation") {
        const messages = await request<ConversationMessageDetail[]>(
          `/api/conversations/${encodeURIComponent(source.id)}/messages`,
        );
        setDetailMessages(messages);
      } else {
        const wrongQuestions = await quizApi.wrongQuestions(null);
        const wrongQuestion = wrongQuestions.find((item) => item.id === source.id);
        if (!wrongQuestion) {
          throw new Error("未找到这条错题记录。");
        }
        setDetailWrongQuestion(wrongQuestion);
      }
    } catch (nextError) {
      setDetailError(
        nextError instanceof Error ? nextError.message : "详情加载失败。",
      );
    } finally {
      setIsDetailLoading(false);
    }
  }

  function closeSourceDetail() {
    setDetailSource(null);
    setDetailMessages([]);
    setDetailWrongQuestion(null);
    setDetailError("");
  }

  function scrollToMemoryHistory() {
    memoryHistoryRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
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
        <Button type="button" variant="ghost" onClick={scrollToMemoryHistory}>
          查看记忆
        </Button>
      </div>

      {error ? <p className="error-text">{error}</p> : null}
      {notice ? <p className="chat-status">{notice}</p> : null}
      <MemorySourcePicker
        sources={uniqueSources}
        selectedSourceKeys={selectedSourceKeys}
        selectedCount={selectedCount}
        allSourcesSelected={allSourcesSelected}
        sourceFilter={sourceFilter}
        isLoading={isLoadingSources}
        onToggle={toggleSource}
        onToggleAll={toggleAllSources}
        onOpenDetail={(source) => void openSourceDetail(source)}
        onFilterChange={setSourceFilter}
        onExtract={() => void extractCandidates()}
        isExtracting={isExtracting}
      />
      {isLoading ? <p className="empty-state">正在加载记忆...</p> : null}

      <div className="memory-groups" ref={memoryHistoryRef}>
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
      {detailSource ? (
        <MemorySourceDetailDialog
          source={detailSource}
          messages={detailMessages}
          wrongQuestion={detailWrongQuestion}
          isLoading={isDetailLoading}
          error={detailError}
          onClose={closeSourceDetail}
        />
      ) : null}
    </section>
  );
}

function MemorySourcePicker({
  sources,
  selectedSourceKeys,
  selectedCount,
  allSourcesSelected,
  sourceFilter,
  isLoading,
  onToggle,
  onToggleAll,
  onOpenDetail,
  onFilterChange,
  onExtract,
  isExtracting,
}: {
  sources: MemoryCandidateSource[];
  selectedSourceKeys: Record<string, boolean>;
  selectedCount: number;
  allSourcesSelected: boolean;
  sourceFilter: SourceFilter;
  isLoading: boolean;
  onToggle: (source: MemoryCandidateSource) => void;
  onToggleAll: () => void;
  onOpenDetail: (source: MemoryCandidateSource) => void;
  onFilterChange: (filter: SourceFilter) => void;
  onExtract: () => void;
  isExtracting: boolean;
}) {
  const sourceCounts = useMemo(
    () => ({
      all: sources.length,
      conversation: sources.filter((source) => source.source_type === "conversation").length,
      wrong_question: sources.filter((source) => source.source_type === "wrong_question").length,
    }),
    [sources],
  );
  const filteredSources = useMemo(
    () =>
      sourceFilter === "all"
        ? sources
        : sources.filter((source) => source.source_type === sourceFilter),
    [sourceFilter, sources],
  );

  return (
    <section className="memory-source-panel" aria-label="候选来源">
      <div className="memory-source-toolbar">
        <div>
          <strong>来源选择</strong>
          <span>已选 {selectedCount} / {sources.length}</span>
        </div>
        <div className="memory-source-actions">
          <Button
            type="button"
            variant="secondary"
            disabled={isExtracting || isLoading || selectedCount === 0}
            onClick={onExtract}
          >
            {isExtracting ? "提取中..." : "提取选中"}
          </Button>
          <Button
            type="button"
            variant="ghost"
            disabled={isLoading || sources.length === 0}
            onClick={onToggleAll}
          >
            {allSourcesSelected ? "全不选" : "全选"}
          </Button>
        </div>
      </div>

      <div className="memory-source-filters" role="tablist" aria-label="来源筛选">
        {sourceFilters.map((filter) => (
          <button
            type="button"
            key={filter}
            className={filter === sourceFilter ? "active" : ""}
            onClick={() => onFilterChange(filter)}
          >
            <span>{sourceFilterLabel(filter)}</span>
            <strong>{sourceCounts[filter]}</strong>
          </button>
        ))}
      </div>

      {isLoading ? <p className="empty-state compact">正在加载候选来源...</p> : null}
      {!isLoading && sources.length === 0 ? (
        <p className="empty-state compact">暂无可加入候选的聊天记录或错题。</p>
      ) : null}
      {!isLoading && sources.length > 0 && filteredSources.length === 0 ? (
        <p className="empty-state compact">当前筛选下暂无来源。</p>
      ) : null}

      <div className="memory-source-list">
        {filteredSources.map((source) => {
          const key = sourceKey(source);
          const selected = Boolean(selectedSourceKeys[key]);
          return (
            <article
              className={`memory-source-item${selected ? " selected" : ""}`}
              key={key}
            >
              <label className="memory-source-select">
                <input
                  type="checkbox"
                  checked={selected}
                  onChange={() => onToggle(source)}
                />
                <span className="memory-source-main">
                  <span className="memory-source-title-row">
                    <strong>{sourceTitle(source)}</strong>
                    <em>{sourceTypeLabel(source.source_type)}</em>
                  </span>
                  <small>{sourceMeta(source)}</small>
                  <span>{source.preview || "暂无预览。"}</span>
                </span>
              </label>
              <Button
                type="button"
                variant="ghost"
                onClick={() => onOpenDetail(source)}
              >
                查看详情
              </Button>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function MemorySourceDetailDialog({
  source,
  messages,
  wrongQuestion,
  isLoading,
  error,
  onClose,
}: {
  source: MemoryCandidateSource;
  messages: ConversationMessageDetail[];
  wrongQuestion: WrongQuestion | null;
  isLoading: boolean;
  error: string;
  onClose: () => void;
}) {
  return (
    <div className="memory-detail-backdrop" role="presentation">
      <section
        className="memory-detail-dialog"
        role="dialog"
        aria-modal="true"
        aria-label={`${sourceTypeLabel(source.source_type)}详情`}
      >
        <header className="memory-detail-header">
          <div>
            <p className="section-label">{sourceTypeLabel(source.source_type)}</p>
            <h3>{sourceTitle(source)}</h3>
            <span>{sourceMeta(source)}</span>
          </div>
          <Button type="button" variant="ghost" onClick={onClose}>
            关闭
          </Button>
        </header>

        {isLoading ? <p className="empty-state compact">正在加载详情...</p> : null}
        {error ? <p className="error-text">{error}</p> : null}

        {!isLoading && !error && source.source_type === "conversation" ? (
          messages.length ? (
            <div className="memory-detail-message-list">
              {messages.map((message) => (
                <article className="memory-detail-message" key={message.id}>
                  <div>
                    <strong>{message.role === "user" ? "用户" : "助手"}</strong>
                    <span>{formatDate(message.created_at)}</span>
                  </div>
                  <p>{message.content}</p>
                </article>
              ))}
            </div>
          ) : (
            <p className="empty-state compact">这段聊天暂无可展示消息。</p>
          )
        ) : null}

        {!isLoading && !error && wrongQuestion ? (
          <div className="memory-detail-wrong">
            <div>
              <span>题目</span>
              <p>{wrongQuestion.prompt}</p>
            </div>
            <div className="memory-detail-choice-grid">
              <DetailField
                label="用户选择"
                value={choiceText(wrongQuestion, wrongQuestion.selected_choice_id)}
              />
              <DetailField
                label="正确答案"
                value={choiceText(wrongQuestion, wrongQuestion.correct_choice_id)}
              />
            </div>
            <div>
              <span>解析</span>
              <p>{wrongQuestion.explanation || "暂无解析。"}</p>
            </div>
          </div>
        ) : null}
      </section>
    </div>
  );
}

function DetailField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <p>{value}</p>
    </div>
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
      <div className="memory-item-heading">
        <div className="memory-item-title">
          <strong title={memory.key || memory.memory_type}>
            {memory.key || memory.memory_type}
          </strong>
          <span>{memory.memory_type}</span>
        </div>
        <span className="memory-confidence">
          {Math.round(memory.confidence * 100)}%
        </span>
      </div>
      <div className="memory-item-main">
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
      <div className="memory-meta">
        <span>{formatScope(memory)}</span>
        <span>{memory.auto_apply ? "自动应用" : "手动应用"}</span>
        {memory.source_conversation_id ? (
          <span title={memory.source_conversation_id}>
            对话 {shortId(memory.source_conversation_id)}
          </span>
        ) : null}
        {memory.evidence_message_ids.length ? (
          <span title={memory.evidence_message_ids.join(", ")}>
            证据 {memory.evidence_message_ids.length}
          </span>
        ) : null}
      </div>
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
  if (status === "active") return "已启用";
  return "已拒绝";
}

function dedupeSources(sources: MemoryCandidateSource[]) {
  const result: MemoryCandidateSource[] = [];
  const seen = new Set<string>();
  for (const source of sources) {
    const key = `${source.source_type}:${source.id}`;
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(source);
  }
  return result;
}

function dedupeMemories(memories: UserMemory[]) {
  const result: UserMemory[] = [];
  const seen = new Set<string>();
  for (const memory of memories) {
    const key = [
      memory.memory_type,
      memory.key || "",
      memory.scope_type,
      memory.scope_id || "",
      memory.value.slice(0, 160),
    ].join(":");
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(memory);
  }
  return result;
}

function mergeSelectedSources(
  current: Record<string, boolean>,
  sources: MemoryCandidateSource[],
): Record<string, boolean> {
  const sourceKeys = sources.map(sourceKey);
  if (Object.keys(current).length === 0) {
    return Object.fromEntries(sourceKeys.map((key) => [key, true] as const));
  }
  return Object.fromEntries(
    sourceKeys.map((key) => [key, current[key] ?? true] as const),
  );
}

function sourceKey(source: MemoryCandidateSource) {
  return `${source.source_type}:${source.id}`;
}

function sourceTypeLabel(sourceType: MemoryCandidateSource["source_type"]) {
  return sourceType === "conversation" ? "聊天记录" : "错题";
}

function sourceFilterLabel(filter: SourceFilter) {
  if (filter === "conversation") return "聊天";
  if (filter === "wrong_question") return "错题";
  return "全部";
}

function sourceTitle(source: MemoryCandidateSource) {
  const title = source.title.trim();
  if (title && title !== "新对话") return title;
  const preview = source.preview.replace(/^(用户|助手|错题)：/, "").trim();
  return preview ? preview.slice(0, 34) : title || sourceTypeLabel(source.source_type);
}

function sourceMeta(source: MemoryCandidateSource) {
  const parts = [formatDate(source.updated_at || source.created_at)];
  if (source.source_type === "conversation") {
    parts.push(`${source.message_count} 条消息`);
  }
  if (source.document_ids.length) {
    parts.push(`${source.document_ids.length} 个文档`);
  }
  if (source.reviewed_at) {
    parts.push("已复习");
  }
  return parts.join(" / ");
}

function shortId(value: string) {
  if (value.length <= 10) return value;
  return `${value.slice(0, 6)}...${value.slice(-4)}`;
}

function choiceText(question: WrongQuestion, choiceId: string) {
  const choice = question.choices.find((item) => item.id === choiceId);
  return choice ? `${choice.id}. ${choice.text}` : choiceId || "未记录";
}

function formatDate(value: string | null) {
  if (!value) return "时间未知";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}
