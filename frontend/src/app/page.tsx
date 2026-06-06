"use client";

import { useEffect, useState } from "react";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { ConversationHistoryPanel } from "@/components/conversation/ConversationHistoryPanel";
import { DocumentList } from "@/components/document/DocumentList";
import { UploadDropzone } from "@/components/document/UploadDropzone";
import { AppShell } from "@/components/layout/AppShell";
import { Button } from "@/components/ui/Button";
import { profileApi } from "@/features/profile/api";
import { QuizPanel } from "@/features/quiz/QuizPanel";
import { useChat } from "@/hooks/useChat";
import { useConversations } from "@/hooks/useConversations";
import { useDocuments } from "@/hooks/useDocuments";
import { useUpload } from "@/hooks/useUpload";
import { api } from "@/lib/api";
import type { ProfileFeedbackRequest } from "@/features/profile/types";
import type { ApiHealth, RAGStatusResponse } from "@/types/api";
import type { AnswerLevel, ChatMode, Conversation } from "@/types/chat";
import type { DocumentSummary } from "@/types/document";

export default function Home() {
  const documents = useDocuments();
  const conversations = useConversations();
  const chat = useChat({
    appendMessage: conversations.appendMessage,
    updateMessage: conversations.updateMessage,
  });
  const upload = useUpload((document) => {
    void documents.refresh();
    documents.setSelectedId(document.id);
  });
  const [health, setHealth] = useState<ApiHealth | null>(null);
  const [ragStatus, setRagStatus] = useState<RAGStatusResponse | null>(null);
  const [customAgentsMd, setCustomAgentsMd] = useState("");
  const [isSavingCustomAgents, setIsSavingCustomAgents] = useState(false);
  const [customAgentsError, setCustomAgentsError] = useState("");
  const [customAgentsNotice, setCustomAgentsNotice] = useState("");
  const [healthError, setHealthError] = useState("");
  const [documentSelectionError, setDocumentSelectionError] = useState("");
  const selectedDocumentIds = conversations.activeConversation?.documentIds || [];
  const selectedConversationDocuments = selectedDocumentIds
    .map((documentId) =>
      documents.documents.find((document) => document.id === documentId),
    )
    .filter((document): document is DocumentSummary => Boolean(document));
  const selectedDocumentNames = selectedDocumentIds.map((documentId, index) => {
    const document = documents.documents.find((item) => item.id === documentId);
    return (
      document?.name ||
      conversations.activeConversation?.documentNames[index] ||
      documentId
    );
  });
  const boundDocumentIssue = getBoundDocumentIssue(
    conversations.activeConversation,
    documents.documents,
  );

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((error: unknown) => {
        setHealthError(error instanceof Error ? error.message : "后端连接失败");
      });
  }, []);

  useEffect(() => {
    api
      .ragStatus()
      .then(setRagStatus)
      .catch(() => setRagStatus(null));
  }, []);

  useEffect(() => {
    profileApi
      .get()
      .then((profile) => {
        setCustomAgentsMd(profile.agents_md || "");
      })
      .catch((error: unknown) => {
        setCustomAgentsError(
          error instanceof Error ? error.message : "自定义提示词加载失败",
        );
      });
  }, []);

  const runtimeStatus = (
    <>
      <span className={health?.ok ? "runtime-dot ready" : "runtime-dot"} />
      <span className="runtime-text">
        {health
          ? `后端已连接 · MinerU ${health.mineru_backend}/${health.mineru_device}`
          : healthError || "正在连接后端..."}
      </span>
      {health && !health.has_api_key ? <strong>未检测到 Qwen API Key</strong> : null}
    </>
  );

  function createConversation() {
    const defaultDocuments = selectedConversationDocuments.length
      ? selectedConversationDocuments
      : documents.selectedDocument?.status === "ready_for_chat"
        ? [documents.selectedDocument]
        : [];
    setDocumentSelectionError("");
    conversations.createConversation(defaultDocuments.slice(0, 3));
  }

  function selectConversation(conversationId: string) {
    const conversation = conversations.conversations.find(
      (item) => item.id === conversationId,
    );
    conversations.selectConversation(conversationId);

    const firstAvailableDocumentId = conversation?.documentIds.find((documentId) =>
      documents.documents.some((document) => document.id === documentId),
    );
    if (firstAvailableDocumentId) {
      documents.setSelectedId(firstAvailableDocumentId);
    }
  }

  async function ask(question: string, level: AnswerLevel, chatMode: ChatMode) {
    const conversation = conversations.activeConversation;
    if (!conversation || boundDocumentIssue) return false;

    conversations.setConversationLevel(conversation.id, level);
    conversations.setConversationChatMode(conversation.id, chatMode);
    return chat.ask({
      question,
      conversationId: conversation.id,
      documentIds: conversation.documentIds,
      level,
      chatMode,
      useProfile: conversation.useProfile,
      useMemory: conversation.useMemory,
    });
  }

  async function saveCustomAgents() {
    setIsSavingCustomAgents(true);
    setCustomAgentsError("");
    setCustomAgentsNotice("");
    try {
      const savedProfile = await profileApi.patch({
        default_depth: "custom",
        agents_md: customAgentsMd,
      });
      setCustomAgentsMd(savedProfile.agents_md || "");
      setCustomAgentsNotice("自定义提示词已保存。");
    } catch (nextError) {
      setCustomAgentsError(
        nextError instanceof Error ? nextError.message : "自定义提示词保存失败",
      );
    } finally {
      setIsSavingCustomAgents(false);
    }
  }

  async function sendMessageFeedback(
    messageId: string,
    feedback: Omit<ProfileFeedbackRequest, "conversation_id" | "message_id">,
  ) {
    const conversation = conversations.activeConversation;
    if (!conversation) {
      throw new Error("请先选择一个对话。");
    }
    const response = await profileApi.feedback({
      conversation_id: conversation.id,
      message_id: messageId,
      ...feedback,
    });
    return response.created_memory_candidates.length;
  }

  function toggleConversationDocument(document: DocumentSummary, selected: boolean) {
    const conversation = conversations.activeConversation;
    setDocumentSelectionError("");
    documents.setSelectedId(document.id);

    if (!conversation) {
      if (selected) {
        conversations.createConversation([document]);
      }
      return;
    }

    const nextDocumentIds = selected
      ? [...conversation.documentIds, document.id]
      : conversation.documentIds.filter((documentId) => documentId !== document.id);
    const uniqueDocumentIds = Array.from(new Set(nextDocumentIds));
    if (uniqueDocumentIds.length > 3) {
      setDocumentSelectionError("最多选择 3 个文档。");
      return;
    }

    const nextDocuments = uniqueDocumentIds
      .map((documentId) =>
        documents.documents.find((currentDocument) => currentDocument.id === documentId),
      )
      .filter((currentDocument): currentDocument is DocumentSummary =>
        Boolean(currentDocument),
      );
    conversations.setConversationDocuments(conversation.id, nextDocuments);
  }

  return (
    <AppShell status={runtimeStatus}>
      <div className="workspace-grid">
        <aside className="left-rail" aria-label="文档工作区">
          <UploadDropzone
            disabled={upload.isUploading}
            onUpload={(file) => void upload.upload(file)}
          />
          {upload.error ? <p className="error-text">{upload.error}</p> : null}
          {documents.error ? <p className="error-text">{documents.error}</p> : null}
          {documents.processNotice ? (
            <p className="warning-text">{documents.processNotice}</p>
          ) : null}
          {documentSelectionError ? (
            <p className="error-text">{documentSelectionError}</p>
          ) : null}

          <section className="panel documents-panel">
            <div className="panel-heading">
              <div>
                <p className="section-label">文档</p>
                <h2>文档队列</h2>
              </div>
              <Button type="button" variant="ghost" onClick={() => void documents.refresh()}>
                刷新
              </Button>
            </div>
            <div className="documents-scroll">
              <DocumentList
                documents={documents.documents}
                focusedId={documents.selectedId}
                selectedIds={selectedDocumentIds}
                isLoading={documents.isLoading}
                onFocus={documents.setSelectedId}
                onToggle={toggleConversationDocument}
              />
            </div>
          </section>

          <section className="panel actions-panel">
            <Button
              type="button"
              variant="primary"
              disabled={!documents.selectedId || documents.isProcessing}
              onClick={() => void documents.processSelected()}
            >
              {documents.isProcessing ? "解析中..." : "解析 / 更新知识库"}
            </Button>
            <Button type="button" variant="secondary" onClick={() => void documents.clearRuntime()}>
              清空后端缓存
            </Button>
          </section>
        </aside>

        <section className="main-column" aria-label="问答工作区">
          <ChatPanel
            conversation={conversations.activeConversation}
            selectedDocumentIds={selectedDocumentIds}
            selectedDocumentNames={selectedDocumentNames}
            boundDocumentIssue={boundDocumentIssue}
            isAsking={chat.isAsking}
            statusMessage={chat.statusMessage}
            isProcessing={documents.isProcessing}
            isWarmingUp={selectedDocumentIds.includes(documents.warmingDocumentId)}
            warmupError={documents.warmupError}
            ragRetrievalStatus={ragStatus?.retrieval}
            customAgentsMd={customAgentsMd}
            isSavingCustomAgents={isSavingCustomAgents}
            customAgentsError={customAgentsError}
            customAgentsNotice={customAgentsNotice}
            onAsk={ask}
            onClearHistory={() => {
              if (conversations.activeConversation) {
                conversations.clearConversationMessages(conversations.activeConversation.id);
              }
            }}
            onLevelChange={(level) => {
              if (conversations.activeConversation) {
                conversations.setConversationLevel(
                  conversations.activeConversation.id,
                  level,
                );
              }
            }}
            onChatModeChange={(chatMode) => {
              if (conversations.activeConversation) {
                conversations.setConversationChatMode(
                  conversations.activeConversation.id,
                  chatMode,
                );
              }
            }}
            onUseProfileChange={(useProfile) => {
              if (conversations.activeConversation) {
                conversations.setConversationUseProfile(
                  conversations.activeConversation.id,
                  useProfile,
                );
              }
            }}
            onUseMemoryChange={(useMemory) => {
              if (conversations.activeConversation) {
                conversations.setConversationUseMemory(
                  conversations.activeConversation.id,
                  useMemory,
                );
              }
            }}
            onCustomAgentsChange={(value) => {
              setCustomAgentsMd(value);
              setCustomAgentsError("");
              setCustomAgentsNotice("");
            }}
            onSaveCustomAgents={saveCustomAgents}
            onMessageFeedback={sendMessageFeedback}
          />
          {chat.error ? <p className="error-text">{chat.error}</p> : null}
        </section>

        <aside className="right-rail" aria-label="学习与对话记录">
          <QuizPanel conversation={conversations.activeConversation} />
          <section className="panel conversation-panel">
            <ConversationHistoryPanel
              conversations={conversations.conversations}
              activeConversationId={conversations.activeConversationId}
              onCreate={createConversation}
              onSelect={selectConversation}
              onDelete={conversations.deleteConversation}
              onClearAll={conversations.clearConversations}
            />
          </section>
        </aside>
      </div>
    </AppShell>
  );
}

function getBoundDocumentIssue(
  conversation: Conversation | null,
  documents: DocumentSummary[],
) {
  if (!conversation || conversation.documentIds.length === 0) return "";
  if (conversation.documentIds.length > 3) return "最多选择 3 个文档。";

  const missingDocuments: string[] = [];
  const notReadyDocuments: string[] = [];

  conversation.documentIds.forEach((documentId, index) => {
    const document = documents.find((item) => item.id === documentId);
    if (!document) {
      missingDocuments.push(conversation.documentNames[index] || documentId);
      return;
    }
    if (document.status !== "ready_for_chat") {
      notReadyDocuments.push(document.name);
    }
  });

  if (missingDocuments.length) {
    return `当前对话绑定的文档不存在：${formatDocumentNames(missingDocuments)}`;
  }
  if (notReadyDocuments.length) {
    return `当前对话绑定的文档未就绪：${formatDocumentNames(notReadyDocuments)}`;
  }
  return "";
}

function formatDocumentNames(names: string[]) {
  const visibleNames = names.slice(0, 2).join("、");
  return names.length > 2 ? `${visibleNames} 等 ${names.length} 个` : visibleNames;
}
