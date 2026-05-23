"use client";

import { FormEvent, useState } from "react";
import { Button } from "@/components/ui/Button";
import { sourceLabel } from "@/lib/format";
import type { AnswerLevel, ChatMessage } from "@/types/chat";

type ChatPanelProps = {
  messages: ChatMessage[];
  selectedDocumentId: string;
  isAsking: boolean;
  onAsk: (question: string, documentId: string, level: AnswerLevel) => Promise<void>;
};

const levels: { value: AnswerLevel; label: string }[] = [
  { value: "beginner", label: "入门" },
  { value: "undergraduate", label: "本科" },
  { value: "expert", label: "专家" },
];

export function ChatPanel({
  messages,
  selectedDocumentId,
  isAsking,
  onAsk,
}: ChatPanelProps) {
  const [question, setQuestion] = useState("");
  const [level, setLevel] = useState<AnswerLevel>("undergraduate");

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await onAsk(question, selectedDocumentId, level);
    setQuestion("");
  }

  return (
    <section className="chat-panel">
      <div className="panel-heading">
        <div>
          <p className="section-label">Ask</p>
          <h2>基于知识库提问</h2>
        </div>
        <div className="segmented-control" aria-label="回答深度">
          {levels.map((item) => (
            <button
              type="button"
              key={item.value}
              className={item.value === level ? "active" : ""}
              onClick={() => setLevel(item.value)}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <div className="message-list">
        {messages.length === 0 ? (
          <p className="empty-state">输入问题后，回答和引用会保留在当前会话中。</p>
        ) : (
          messages.map((message) => <MessageBubble key={message.id} message={message} />)
        )}
      </div>

      <form className="question-form" onSubmit={submit}>
        <textarea
          value={question}
          placeholder="例如：请解释达朗贝尔公式的推导思路，并列出关键边界条件。"
          onChange={(event) => setQuestion(event.target.value)}
        />
        <Button
          type="submit"
          variant="primary"
          disabled={isAsking || !question.trim()}
        >
          {isAsking ? "生成中..." : "生成回答"}
        </Button>
      </form>
    </section>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  return (
    <article className={`message message-${message.role}`}>
      <div className="message-role">{message.role === "user" ? "你" : "AI 助教"}</div>
      <p>{message.content}</p>
      {message.sources?.length ? (
        <div className="message-sources">
          {message.sources.slice(0, 4).map((source) => (
            <span key={source.id}>
              {sourceLabel(source.type)}
              {source.page ? ` P${source.page}` : ""}
            </span>
          ))}
        </div>
      ) : null}
    </article>
  );
}
