"use client";

import {
  FormEvent,
  KeyboardEvent,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import {
  extractInlineImageIds,
  MarkdownAnswer,
} from "@/components/chat/MarkdownAnswer";
import { RelatedImages } from "@/components/chat/RelatedImages";
import { Button } from "@/components/ui/Button";
import type { RAGRetrievalStatus } from "@/types/api";
import type { AnswerLevel, ChatMode, ChatMessage, Conversation } from "@/types/chat";

type ChatPanelProps = {
  conversation: Conversation | null;
  selectedDocumentIds: string[];
  selectedDocumentNames: string[];
  boundDocumentIssue?: string;
  isAsking: boolean;
  statusMessage?: string;
  isProcessing?: boolean;
  isWarmingUp?: boolean;
  warmupError?: string;
  ragRetrievalStatus?: RAGRetrievalStatus | null;
  customAgentsMd: string;
  isSavingCustomAgents?: boolean;
  customAgentsError?: string;
  customAgentsNotice?: string;
  onAsk: (
    question: string,
    level: AnswerLevel,
    chatMode: ChatMode,
  ) => Promise<boolean>;
  onClearHistory: () => void;
  onLevelChange: (level: AnswerLevel) => void;
  onChatModeChange: (chatMode: ChatMode) => void;
  onCustomAgentsChange: (value: string) => void;
  onSaveCustomAgents: () => Promise<void>;
};

const levels: { value: AnswerLevel; label: string }[] = [
  { value: "beginner", label: "入门" },
  { value: "undergraduate", label: "本科" },
  { value: "expert", label: "专家" },
  { value: "custom", label: "自定义" },
];

const retrievalModes: {
  value: ChatMode;
  label: string;
  description: string;
}[] = [
  {
    value: "multimodal",
    label: "多模态精读",
    description: "更慢，会分析图片和公式。",
  },
  {
    value: "fast_text",
    label: "快速文本",
    description: "更快，主要基于文本索引回答。",
  },
];

const EMPTY_MESSAGES: ChatMessage[] = [];

export function ChatPanel({
  conversation,
  selectedDocumentIds,
  selectedDocumentNames,
  boundDocumentIssue,
  isAsking,
  statusMessage,
  isProcessing,
  isWarmingUp,
  warmupError,
  ragRetrievalStatus,
  customAgentsMd,
  isSavingCustomAgents,
  customAgentsError,
  customAgentsNotice,
  onAsk,
  onClearHistory,
  onLevelChange,
  onChatModeChange,
  onCustomAgentsChange,
  onSaveCustomAgents,
}: ChatPanelProps) {
  const [question, setQuestion] = useState("");
  const [isCustomPromptOpen, setIsCustomPromptOpen] = useState(false);
  const messageListRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const customPromptTextareaRef = useRef<HTMLTextAreaElement>(null);
  const isSubmittingRef = useRef(false);
  const messages = conversation?.messages ?? EMPTY_MESSAGES;
  const level = conversation?.level || "undergraduate";
  const chatMode = conversation?.chatMode || "multimodal";
  const askDisabledReason = getAskDisabledReason({
    hasConversation: Boolean(conversation),
    selectedDocumentIds,
    isProcessing,
    boundDocumentIssue,
  });
  const canAsk = !askDisabledReason;
  const currentDocumentLabel = getSelectedDocumentsLabel(selectedDocumentNames);
  const placeholder = getPlaceholder({
    hasConversation: Boolean(conversation),
    selectedDocumentIds,
    isProcessing,
    boundDocumentIssue,
  });
  const selectedRetrievalMode = retrievalModes.find(
    (item) => item.value === chatMode,
  )!;
  const selectedLevel = levels.find((item) => item.value === level) || levels[1];

  useEffect(() => {
    const element = messageListRef.current;
    if (!element) return;
    element.scrollTo({ top: element.scrollHeight });
  }, [
    messages,
    selectedDocumentIds,
    statusMessage,
    isAsking,
    isWarmingUp,
    warmupError,
  ]);

  useLayoutEffect(() => {
    resizeQuestionInput(textareaRef.current);
  }, [question]);

  useEffect(() => {
    if (!isCustomPromptOpen) return;
    window.requestAnimationFrame(() => customPromptTextareaRef.current?.focus());

    function closeOnEscape(event: globalThis.KeyboardEvent) {
      if (event.key === "Escape") {
        setIsCustomPromptOpen(false);
      }
    }

    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [isCustomPromptOpen]);

  function selectLevel(nextLevel: AnswerLevel) {
    onLevelChange(nextLevel);
    if (nextLevel === "custom") {
      setIsCustomPromptOpen(true);
    }
  }

  async function submitQuestion() {
    const originalQuestion = question;
    const trimmedQuestion = originalQuestion.trim();
    if (!trimmedQuestion || !canAsk || isAsking || isSubmittingRef.current) return;

    isSubmittingRef.current = true;
    setQuestion("");
    try {
      const didAsk = await onAsk(originalQuestion, level, chatMode);
      if (!didAsk) {
        setQuestion(originalQuestion);
      }
    } finally {
      isSubmittingRef.current = false;
      window.requestAnimationFrame(() => textareaRef.current?.focus());
    }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitQuestion();
  }

  function handleQuestionKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.nativeEvent.isComposing || event.key !== "Enter" || event.shiftKey) {
      return;
    }

    event.preventDefault();
    void submitQuestion();
  }

  return (
    <section className="chat-panel">
      <div className="chat-panel-header">
        <div className="chat-title-block">
          <p className="section-label">Chat</p>
          <h2>知识问答</h2>
          <div className="chat-document-row">
            <div className="current-document">
              <span>已选择 {selectedDocumentIds.length} 个文档：</span>
              <strong title={selectedDocumentNames.join("、") || "未选择文档"}>
                {currentDocumentLabel || "未选择文档"}
              </strong>
              <small className="history-note">
                {conversation
                  ? `当前对话：${conversation.title}`
                  : "请新建对话后开始提问"}
              </small>
            </div>
            <div className="mode-selector">
              <div className="mode-selector-heading">
                <span>检索模式</span>
                <span
                  className="mode-help"
                  title="多模态精读：更慢，会分析图片和公式。快速文本：更快，主要基于文本索引回答。"
                  aria-label="检索模式说明"
                >
                  ?
                </span>
              </div>
              <div
                className="segmented-control retrieval-mode-control"
                aria-label="检索模式"
              >
                {retrievalModes.map((item) => (
                  <button
                    type="button"
                    key={item.value}
                    className={item.value === chatMode ? "active" : ""}
                    disabled={!conversation}
                    onClick={() => onChatModeChange(item.value)}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
              <p className="mode-description">{selectedRetrievalMode.description}</p>
              <div className="retrieval-runtime">
                <span>{ragRetrievalStatus?.default_mode || "hybrid"} retrieval</span>
                <span>
                  Rerank {ragRetrievalStatus?.rerank_enabled ? "enabled" : "disabled"}
                </span>
                {ragRetrievalStatus?.rerank_model ? (
                  <span title={ragRetrievalStatus.rerank_provider || undefined}>
                    {ragRetrievalStatus.rerank_model}
                  </span>
                ) : null}
              </div>
            </div>
          </div>
        </div>

        <div className="chat-controls">
          <div className="answer-depth-control">
            <div className="answer-depth-heading">
              <span className="control-label">回答深度</span>
              <span className="depth-current">{selectedLevel.label}</span>
            </div>
            <div
              className="segmented-control answer-depth-segmented"
              aria-label="回答深度"
            >
              {levels.map((item) => (
                <button
                  type="button"
                  key={item.value}
                  className={item.value === level ? "active" : ""}
                  disabled={!conversation}
                  onClick={() => selectLevel(item.value)}
                >
                  {item.label}
                </button>
              ))}
            </div>
            {level === "custom" ? (
              <Button
                type="button"
                variant="secondary"
                className="custom-prompt-open-button"
                disabled={!conversation}
                onClick={() => setIsCustomPromptOpen(true)}
              >
                编辑 AGENTS.md
              </Button>
            ) : null}
          </div>
          <Button
            type="button"
            variant="ghost"
            className="history-clear-button"
            disabled={!conversation || messages.length === 0}
            onClick={onClearHistory}
          >
            清空当前对话
          </Button>
        </div>
      </div>

      <div className="message-list" ref={messageListRef}>
        {!conversation ? <p className="empty-state">请新建对话</p> : null}
        {conversation && messages.length === 0 ? (
          <p className="empty-state">当前会话还没有消息。</p>
        ) : null}
        {conversation && isWarmingUp ? (
          <p className="chat-status">正在预热知识库...</p>
        ) : null}
        {conversation && warmupError ? (
          <p className="chat-status muted">{warmupError}</p>
        ) : null}
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
      </div>

      <form className="question-form" onSubmit={submit}>
        <textarea
          ref={textareaRef}
          value={question}
          placeholder={placeholder}
          rows={2}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={handleQuestionKeyDown}
          readOnly={isAsking}
        />
        <Button
          type="submit"
          variant="primary"
          disabled={isAsking || !question.trim() || !canAsk}
        >
          {isAsking ? "生成中..." : "生成回答"}
        </Button>
        {askDisabledReason ? <p className="composer-note">{askDisabledReason}</p> : null}
        {statusMessage ? <p className="chat-status composer-note">{statusMessage}</p> : null}
      </form>

      {isCustomPromptOpen ? (
        <div
          className="custom-agents-dialog-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setIsCustomPromptOpen(false);
            }
          }}
        >
          <section
            className="custom-agents-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="custom-agents-title"
          >
            <div className="custom-agents-dialog-header">
              <div>
                <p className="section-label">Custom Prompt</p>
                <h3 id="custom-agents-title">自定义 AGENTS.md</h3>
              </div>
              <button
                type="button"
                className="dialog-close-button"
                aria-label="关闭自定义提示词窗口"
                onClick={() => setIsCustomPromptOpen(false)}
              >
                ×
              </button>
            </div>
            <p className="custom-agents-dialog-description">
              这段提示词只在回答深度选择“自定义”时生效。用于约束回答难度、出题格式、讲解风格和评分要求。
            </p>
            <textarea
              ref={customPromptTextareaRef}
              className="custom-agents-dialog-textarea"
              value={customAgentsMd}
              onChange={(event) => onCustomAgentsChange(event.target.value)}
            />
            {customAgentsError ? (
              <p className="error-text">{customAgentsError}</p>
            ) : customAgentsNotice ? (
              <p className="chat-status">{customAgentsNotice}</p>
            ) : null}
            <div className="custom-agents-dialog-actions">
              <Button
                type="button"
                variant="ghost"
                onClick={() => setIsCustomPromptOpen(false)}
              >
                关闭
              </Button>
              <Button
                type="button"
                variant="primary"
                disabled={isSavingCustomAgents}
                onClick={() => void onSaveCustomAgents()}
              >
                {isSavingCustomAgents ? "保存中..." : "保存并用于自定义"}
              </Button>
            </div>
          </section>
        </div>
      ) : null}
    </section>
  );
}

function getAskDisabledReason({
  hasConversation,
  selectedDocumentIds,
  isProcessing,
  boundDocumentIssue,
}: {
  hasConversation: boolean;
  selectedDocumentIds: string[];
  isProcessing?: boolean;
  boundDocumentIssue?: string;
}) {
  if (!hasConversation) return "请新建对话";
  if (boundDocumentIssue) return boundDocumentIssue;
  if (!selectedDocumentIds.length) return "请至少选择一个可提问文档";
  if (isProcessing) return "当前文档正在索引";
  return "";
}

function getPlaceholder({
  hasConversation,
  selectedDocumentIds,
  isProcessing,
  boundDocumentIssue,
}: {
  hasConversation: boolean;
  selectedDocumentIds: string[];
  isProcessing?: boolean;
  boundDocumentIssue?: string;
}) {
  if (!hasConversation) return "请新建对话";
  if (boundDocumentIssue) return boundDocumentIssue;
  if (!selectedDocumentIds.length) return "请至少选择一个可提问文档";
  if (isProcessing) return "当前文档正在索引";
  if (selectedDocumentIds.length > 1) {
    return `正在基于 ${selectedDocumentIds.length} 个文档提问...`;
  }
  return "正在基于 1 个文档提问...";
}

function getSelectedDocumentsLabel(documentNames: string[]) {
  if (!documentNames.length) return "";
  const visibleNames = documentNames.slice(0, 2).join("、");
  return documentNames.length > 2
    ? `${visibleNames} 等 ${documentNames.length} 个文档`
    : visibleNames;
}

function resizeQuestionInput(textarea: HTMLTextAreaElement | null) {
  if (!textarea) return;

  textarea.style.height = "auto";
  const styles = window.getComputedStyle(textarea);
  const lineHeight = Number.parseFloat(styles.lineHeight) || 24;
  const padding =
    Number.parseFloat(styles.paddingTop) + Number.parseFloat(styles.paddingBottom);
  const border =
    Number.parseFloat(styles.borderTopWidth) +
    Number.parseFloat(styles.borderBottomWidth);
  const minHeight = lineHeight * 2 + padding + border;
  const maxHeight = lineHeight * 6 + padding + border;
  const nextHeight = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight);

  textarea.style.height = `${nextHeight}px`;
  textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const fallbackImages = getFallbackRelatedImages(message);

  return (
    <article className={`message message-${message.role}`}>
      <div className="message-role">{message.role === "user" ? "你" : "AI 助教"}</div>
      {message.role === "assistant" ? (
        <>
          <MarkdownAnswer
            content={message.content}
            relatedImages={message.relatedImages}
          />
          <RelatedImages images={fallbackImages} />
        </>
      ) : (
        <p>{message.content}</p>
      )}
      {message.status === "failed" ? (
        <p className="message-state">
          回答生成失败，已保留该问题。
          {message.error ? ` ${message.error}` : ""}
        </p>
      ) : null}
    </article>
  );
}

function getFallbackRelatedImages(message: ChatMessage) {
  if (!message.relatedImages?.length) return undefined;

  const inlineImageIds = new Set(extractInlineImageIds(message.content));
  if (inlineImageIds.size === 0) {
    return message.relatedImages;
  }

  const images = message.relatedImages.filter(
    (image) => !inlineImageIds.has(image.imageId),
  );
  return images.length ? images : undefined;
}
