"use client";

import { Button } from "@/components/ui/Button";
import type { Conversation } from "@/types/chat";

type ConversationHistoryPanelProps = {
  conversations: Conversation[];
  activeConversationId: string;
  onCreate: () => void;
  onSelect: (conversationId: string) => void;
  onDelete: (conversationId: string) => void;
  onClearAll: () => void;
};

export function ConversationHistoryPanel({
  conversations,
  activeConversationId,
  onCreate,
  onSelect,
  onDelete,
  onClearAll,
}: ConversationHistoryPanelProps) {
  function clearAll() {
    if (!conversations.length) return;
    if (window.confirm("确定要清空全部对话记录吗？")) {
      onClearAll();
    }
  }

  return (
    <>
      <div className="panel-heading conversation-heading">
        <div>
          <p className="section-label">Conversations</p>
          <h2>对话记录</h2>
        </div>
        <Button type="button" variant="primary" onClick={onCreate}>
          新建对话
        </Button>
      </div>

      <div className="conversation-actions">
        <Button
          type="button"
          variant="ghost"
          disabled={!conversations.length}
          onClick={clearAll}
        >
          清空全部
        </Button>
      </div>

      <div className="conversation-scroll">
        {!conversations.length ? (
          <p className="empty-state">还没有对话记录。</p>
        ) : (
          <div className="conversation-list">
            {conversations.map((conversation) => (
              <article
                className={`conversation-row ${
                  conversation.id === activeConversationId ? "selected" : ""
                }`}
                key={conversation.id}
              >
                <button
                  type="button"
                  className="conversation-select"
                  onClick={() => onSelect(conversation.id)}
                >
                  <strong title={conversation.title}>{conversation.title}</strong>
                  <span>
                    {conversation.documentIds.length} 个文档 ·{" "}
                    {formatUpdatedAt(conversation.updatedAt)}
                  </span>
                </button>
                <Button
                  type="button"
                  variant="ghost"
                  className="conversation-delete"
                  aria-label={`删除对话 ${conversation.title}`}
                  onClick={() => onDelete(conversation.id)}
                >
                  删除
                </Button>
              </article>
            ))}
          </div>
        )}
      </div>
    </>
  );
}

function formatUpdatedAt(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "时间未知";

  const now = new Date();
  const isSameDate = date.toDateString() === now.toDateString();
  if (isSameDate) {
    return date.toLocaleTimeString("zh-CN", {
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  return date.toLocaleDateString("zh-CN", {
    month: "numeric",
    day: "numeric",
  });
}
