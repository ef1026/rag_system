import Link from "next/link";
import type { Conversation } from "@/types/chat";

type QuizPanelProps = {
  conversation: Conversation | null;
};

export function QuizPanel({ conversation }: QuizPanelProps) {
  const sourceHint = conversation?.messages.length
    ? "优先基于当前学习回答、历史小测和错题记录出题。"
    : "当前对话为空时，使用学习历史、小测记录和错题记录出题。";

  return (
    <section className="quiz-panel" aria-label="小测验">
      <div className="quiz-panel-heading">
        <div>
          <span className="control-label">学习闭环</span>
          <h3>三题小测验</h3>
        </div>
        <span className="quiz-count-badge">3 题</span>
      </div>

      <p className="quiz-hint">{sourceHint}</p>

      <div className="quiz-actions">
        <Link className="button button-primary" href={quizHref(conversation)}>
          一键出题
        </Link>
        <Link className="button button-secondary" href="/wrong-book">
          错题本
        </Link>
      </div>
    </section>
  );
}

function quizHref(conversation: Conversation | null) {
  const params = new URLSearchParams({ auto: "1" });
  if (conversation?.id) {
    params.set("conversationId", conversation.id);
  }
  return `/quiz?${params.toString()}`;
}
