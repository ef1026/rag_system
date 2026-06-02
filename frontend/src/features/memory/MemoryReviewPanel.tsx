"use client";

import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/Button";
import { memoryApi } from "./api";
import type { UserMemory } from "./types";

const statuses = ["candidate", "active", "dismissed"] as const;

export function MemoryReviewPanel() {
  const [memories, setMemories] = useState<UserMemory[]>([]);
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
                  <article className="memory-item" key={memory.id}>
                    <div className="memory-item-main">
                      <strong>{memory.key || memory.memory_type}</strong>
                      <p>{memory.value}</p>
                    </div>
                    <dl className="memory-meta">
                      <div>
                        <dt>Type</dt>
                        <dd>{memory.memory_type}</dd>
                      </div>
                      <div>
                        <dt>Confidence</dt>
                        <dd>{Math.round(memory.confidence * 100)}%</dd>
                      </div>
                      {memory.source_conversation_id ? (
                        <div>
                          <dt>Source</dt>
                          <dd>{memory.source_conversation_id}</dd>
                        </div>
                      ) : null}
                    </dl>
                    {memory.evidence ? (
                      <p className="memory-evidence">{memory.evidence}</p>
                    ) : null}
                    <div className="memory-actions">
                      {memory.status === "candidate" ? (
                        <>
                          <Button
                            type="button"
                            variant="secondary"
                            onClick={() =>
                              void runAction(() => memoryApi.accept(memory.id))
                            }
                          >
                            Accept
                          </Button>
                          <Button
                            type="button"
                            variant="ghost"
                            onClick={() =>
                              void runAction(() => memoryApi.dismiss(memory.id))
                            }
                          >
                            Dismiss
                          </Button>
                        </>
                      ) : null}
                      <Button
                        type="button"
                        variant="ghost"
                        onClick={() => void runAction(() => memoryApi.remove(memory.id))}
                      >
                        Delete
                      </Button>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
        ))}
      </div>
    </section>
  );
}

function labelForStatus(status: (typeof statuses)[number]) {
  if (status === "candidate") return "Candidates";
  if (status === "active") return "Active";
  return "Dismissed";
}
