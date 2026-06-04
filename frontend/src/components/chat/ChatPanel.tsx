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
import type {
  AnswerLevel,
  ChatMode,
  ChatMessage,
  Conversation,
  RelatedImage,
} from "@/types/chat";

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
  onUseProfileChange: (useProfile: boolean) => void;
  onUseMemoryChange: (useMemory: boolean) => void;
  onCustomAgentsChange: (value: string) => void;
  onSaveCustomAgents: () => Promise<void>;
  onMessageFeedback: (
    messageId: string,
    feedback: MessageFeedbackPayload,
  ) => Promise<number>;
};

type MessageFeedbackPayload = {
  rating?: "helpful" | "not_helpful";
  difficulty?: "too_easy" | "too_hard";
  style_feedback?: "needs_examples" | "needs_derivation" | "more_concise";
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
  onUseProfileChange,
  onUseMemoryChange,
  onCustomAgentsChange,
  onSaveCustomAgents,
  onMessageFeedback,
}: ChatPanelProps) {
  const [question, setQuestion] = useState("");
  const [isCustomPromptOpen, setIsCustomPromptOpen] = useState(false);
  const [feedbackState, setFeedbackState] = useState<
    Record<string, { isSaving?: boolean; notice?: string; error?: string }>
  >({});
  const messageListRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const customPromptTextareaRef = useRef<HTMLTextAreaElement>(null);
  const isSubmittingRef = useRef(false);
  const messages = conversation?.messages ?? EMPTY_MESSAGES;
  const level = conversation?.level || "undergraduate";
  const chatMode = conversation?.chatMode || "multimodal";
  const useProfile = conversation?.useProfile ?? true;
  const useMemory = conversation?.useMemory ?? false;
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

  async function submitFeedback(
    messageId: string,
    feedback: MessageFeedbackPayload,
  ) {
    setFeedbackState((current) => ({
      ...current,
      [messageId]: { isSaving: true },
    }));
    try {
      const createdCount = await onMessageFeedback(messageId, feedback);
      setFeedbackState((current) => ({
        ...current,
        [messageId]: {
          notice: createdCount
            ? `Feedback saved. ${createdCount} memory candidate(s) created.`
            : "Feedback saved.",
        },
      }));
    } catch (nextError) {
      setFeedbackState((current) => ({
        ...current,
        [messageId]: {
          error: nextError instanceof Error ? nextError.message : "Feedback failed.",
        },
      }));
    }
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
              <span className="depth-current">{depthLabel(selectedLevel.value)}</span>
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
                  {depthLabel(item.value)}
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
          <div className="personalization-controls">
            <div className="personalization-heading">
              <span className="control-label">个性化</span>
              <span className="personalization-state">
                {useProfile && useMemory
                  ? "Profile + Memory"
                  : useProfile
                    ? "Profile"
                    : useMemory
                      ? "Memory"
                      : "Off"}
              </span>
            </div>
            <div className="personalization-toggle-grid">
              <button
                type="button"
                className={`toggle-pill ${useProfile ? "active" : ""}`}
                disabled={!conversation}
                aria-pressed={useProfile}
                onClick={() => onUseProfileChange(!useProfile)}
              >
                <span className="toggle-indicator" />
                <span>
                  <strong>Profile</strong>
                  <small>显式偏好</small>
                </span>
              </button>
              <button
                type="button"
                className={`toggle-pill ${useMemory ? "active" : ""}`}
                disabled={!conversation}
                aria-pressed={useMemory}
                onClick={() => onUseMemoryChange(!useMemory)}
              >
                <span className="toggle-indicator" />
                <span>
                  <strong>Memory</strong>
                  <small>审核记忆</small>
                </span>
              </button>
            </div>
          </div>
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
          <MessageBubble
            key={message.id}
            message={message}
            feedbackState={feedbackState[message.id]}
            onFeedback={(feedback) => void submitFeedback(message.id, feedback)}
          />
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

function depthLabel(value: AnswerLevel) {
  if (value === "beginner") return "入门";
  if (value === "expert") return "专家";
  if (value === "custom") return "自定义";
  return "本科";
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

function MessageBubble({
  message,
  feedbackState,
  onFeedback,
}: {
  message: ChatMessage;
  feedbackState?: { isSaving?: boolean; notice?: string; error?: string };
  onFeedback: (feedback: MessageFeedbackPayload) => void;
}) {
  const displayContent = getContentWithInlineImages(message);
  const fallbackImages = getFallbackRelatedImages(message, displayContent);

  return (
    <article className={`message message-${message.role}`}>
      <div className="message-role">{message.role === "user" ? "你" : "AI 助教"}</div>
      {message.role === "assistant" ? (
        <>
          <MarkdownAnswer
            content={displayContent}
            relatedImages={message.relatedImages}
          />
          <RelatedImages images={fallbackImages} />
          <MessageFeedback
            isSaving={Boolean(feedbackState?.isSaving)}
            notice={feedbackState?.notice}
            error={feedbackState?.error}
            onFeedback={onFeedback}
          />
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

function MessageFeedback({
  isSaving,
  notice,
  error,
  onFeedback,
}: {
  isSaving: boolean;
  notice?: string;
  error?: string;
  onFeedback: (feedback: MessageFeedbackPayload) => void;
}) {
  const actions: Array<{ label: string; payload: MessageFeedbackPayload }> = [
    { label: "有帮助", payload: { rating: "helpful" } },
    { label: "太简单", payload: { difficulty: "too_easy" } },
    { label: "太难", payload: { difficulty: "too_hard" } },
    { label: "需要例子", payload: { style_feedback: "needs_examples" } },
    { label: "需要推导", payload: { style_feedback: "needs_derivation" } },
    { label: "更简洁", payload: { style_feedback: "more_concise" } },
  ];

  return (
    <div className="message-feedback">
      <div className="message-feedback-actions" aria-label="学习反馈">
        {actions.map((action) => (
          <button
            type="button"
            key={action.label}
            disabled={isSaving}
            onClick={() => onFeedback(action.payload)}
          >
            {action.label}
          </button>
        ))}
      </div>
      {notice ? <p className="message-feedback-note">{notice}</p> : null}
      {error ? <p className="message-feedback-error">{error}</p> : null}
    </div>
  );
}

function getContentWithInlineImages(message: ChatMessage) {
  if (message.role !== "assistant" || !message.relatedImages?.length) {
    return message.content;
  }
  if (extractInlineImageIds(message.content).length > 0) {
    return message.content;
  }

  const referencedImages = pickInlineImages(
    message.relatedImages,
    message.inlineImageRefs,
  );
  if (!referencedImages.length) {
    return message.content;
  }

  return insertImageRefsIntoContent(
    message.content,
    referencedImages.map((image) => image.imageId),
  );
}

function pickInlineImages(
  relatedImages: RelatedImage[],
  inlineImageRefs?: string[],
) {
  const imagesById = new Map(relatedImages.map((image) => [image.imageId, image]));
  const explicitImages =
    inlineImageRefs
      ?.map((imageId) => imagesById.get(imageId))
      .filter((image): image is RelatedImage => Boolean(image)) ?? [];

  return balancedInlineImages(explicitImages.length ? explicitImages : relatedImages);
}

function balancedInlineImages(images: RelatedImage[], maxImages = 3) {
  const selected: RelatedImage[] = [];
  const seenIds = new Set<string>();
  const seenDocuments = new Set<string>();

  for (const image of images) {
    if (seenIds.has(image.imageId) || seenDocuments.has(image.documentId)) continue;
    selected.push(image);
    seenIds.add(image.imageId);
    seenDocuments.add(image.documentId);
    if (selected.length >= maxImages) return selected;
  }

  for (const image of images) {
    if (seenIds.has(image.imageId)) continue;
    selected.push(image);
    seenIds.add(image.imageId);
    if (selected.length >= maxImages) break;
  }

  return selected;
}

function insertImageRefsIntoContent(content: string, imageIds: string[]) {
  if (!imageIds.length || !content.trim()) return content;

  const paragraphs = content.trimEnd().split("\n\n");
  const anchorIndices = paragraphs
    .map((paragraph, index) => ({ paragraph: paragraph.trim(), index }))
    .filter(
      ({ paragraph }) =>
        paragraph &&
        !paragraph.startsWith("```") &&
        !paragraph.startsWith("|") &&
        !paragraph.startsWith("[[image:"),
    )
    .map(({ index }) => index);

  if (!anchorIndices.length) {
    return `${content.trimEnd()}\n\n${imageIds
      .map((imageId) => `[[image:${imageId}]]`)
      .join("\n\n")}`;
  }

  const preferredPositions = [
    anchorIndices[Math.min(1, anchorIndices.length - 1)],
    anchorIndices[Math.floor(anchorIndices.length / 2)],
    anchorIndices[anchorIndices.length - 1],
  ];
  const usedPositions = new Set<number>();
  const insertions = new Map<number, string[]>();

  imageIds.forEach((imageId, index) => {
    let position = preferredPositions[index] ?? anchorIndices[anchorIndices.length - 1];
    if (usedPositions.has(position)) {
      position = anchorIndices.find((candidate) => !usedPositions.has(candidate)) ?? position;
    }
    usedPositions.add(position);
    insertions.set(position, [
      ...(insertions.get(position) ?? []),
      `[[image:${imageId}]]`,
    ]);
  });

  const output: string[] = [];
  paragraphs.forEach((paragraph, index) => {
    output.push(paragraph);
    output.push(...(insertions.get(index) ?? []));
  });
  return output.join("\n\n");
}

function getFallbackRelatedImages(message: ChatMessage, content: string) {
  if (!message.relatedImages?.length) return undefined;

  const inlineImageIds = new Set(extractInlineImageIds(content));
  if (inlineImageIds.size === 0) {
    return message.relatedImages;
  }

  const images = message.relatedImages.filter(
    (image) => !inlineImageIds.has(image.imageId),
  );
  return images.length ? images : undefined;
}
