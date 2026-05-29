"use client";

import { useRef, useState } from "react";
import { api } from "@/lib/api";
import { ApiError } from "@/lib/errors";
import type { ChatRequest, ImageAssetPublic } from "@/types/api";
import type { AnswerLevel, ChatMessage, ChatMode, RelatedImage } from "@/types/chat";

type UseChatOptions = {
  appendMessage: (conversationId: string, message: ChatMessage) => void;
  updateMessage: (
    conversationId: string,
    messageId: string,
    update: (message: ChatMessage) => ChatMessage,
  ) => void;
};

type AskOptions = {
  question: string;
  conversationId: string;
  documentIds: string[];
  level: AnswerLevel;
  chatMode: ChatMode;
};

function statusForElapsed(seconds: number, chatMode: ChatMode) {
  if (chatMode === "fast_text") {
    return seconds >= 30
      ? "仍在使用文本索引快速检索，请稍候..."
      : "正在使用文本索引快速检索。";
  }
  return seconds >= 30
    ? "仍在进行多模态理解，可能较慢，请稍候..."
    : "正在进行多模态理解，可能较慢。";
}

function vlmEnhancedForChatMode(chatMode: ChatMode): ChatRequest["vlm_enhanced"] {
  return chatMode === "fast_text" ? false : "auto";
}

function mapRelatedImages(
  images: ImageAssetPublic[] | undefined,
): RelatedImage[] | undefined {
  if (!images?.length) return undefined;

  const relatedImages = images
    .filter((image) => image.image_id && image.document_id && image.url)
    .map((image) => ({
      imageId: image.image_id,
      documentId: image.document_id,
      documentName: image.document_name || image.document_id,
      url: image.url,
      ...(image.caption ? { caption: image.caption } : {}),
      ...(typeof image.page === "number" ? { page: image.page } : {}),
      ...(image.bbox !== undefined ? { bbox: image.bbox } : {}),
      ...(image.source_type ? { sourceType: image.source_type } : {}),
      ...(image.relevance_reason
        ? { relevanceReason: image.relevance_reason }
        : {}),
    }));

  return relatedImages.length ? relatedImages : undefined;
}

export function useChat({ appendMessage, updateMessage }: UseChatOptions) {
  const [isAsking, setIsAsking] = useState(false);
  const isAskingRef = useRef(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [error, setError] = useState<string>("");

  async function ask({
    question,
    conversationId,
    documentIds,
    level,
    chatMode,
  }: AskOptions) {
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) return false;
    if (!conversationId) {
      setError("请先新建对话");
      return false;
    }
    if (!documentIds.length) {
      setError("请至少选择一个可提问文档");
      return false;
    }
    if (isAskingRef.current) return false;

    isAskingRef.current = true;
    setIsAsking(true);
    setStatusMessage(statusForElapsed(0, chatMode));
    setError("");

    const mode = "hybrid";
    const userMessageId = crypto.randomUUID();
    const startedAt = Date.now();
    const statusTimer = window.setInterval(() => {
      const elapsedSeconds = Math.floor((Date.now() - startedAt) / 1000);
      setStatusMessage(statusForElapsed(elapsedSeconds, chatMode));
    }, 1000);

    appendMessage(conversationId, {
      id: userMessageId,
      role: "user",
      content: trimmedQuestion,
      createdAt: new Date().toISOString(),
      documentId: documentIds[0],
      documentIds,
      mode,
      chatMode,
      level,
      status: "sent",
    });

    try {
      const vlmEnhanced = vlmEnhancedForChatMode(chatMode);
      const payload: ChatRequest = {
        question: trimmedQuestion,
        document_id: documentIds[0],
        document_ids: documentIds,
        level,
        mode,
        vlm_enhanced: vlmEnhanced,
      };
      console.debug("chat payload", payload);
      const response = await api.chat(payload);
      if (!response.answer.trim()) {
        updateMessage(conversationId, userMessageId, (message) => ({
          ...message,
          error: "后端返回空回答",
          status: "failed",
        }));
        setError("后端返回空回答，原问题已恢复到输入框。");
        return false;
      }

      const relatedImages = mapRelatedImages(response.related_images);
      const inlineImageRefs = response.inline_image_refs?.filter(Boolean);
      appendMessage(conversationId, {
        id: crypto.randomUUID(),
        role: "assistant",
        content: response.answer,
        createdAt: new Date().toISOString(),
        documentId: documentIds[0],
        documentIds,
        mode,
        chatMode,
        level,
        ...(relatedImages ? { relatedImages } : {}),
        ...(inlineImageRefs?.length ? { inlineImageRefs } : {}),
        status: "sent",
      });
      return true;
    } catch (nextError) {
      const message =
        nextError instanceof Error ? nextError.message : "问答生成失败";
      updateMessage(conversationId, userMessageId, (currentMessage) => ({
        ...currentMessage,
        error: message,
        status: "failed",
      }));
      if (
        nextError instanceof ApiError &&
        nextError.status === 409 &&
        nextError.code === "knowledge_base_not_ready"
      ) {
        setError("请先解析/更新知识库，原问题已恢复到输入框。");
      } else {
        setError(`${message}，原问题已恢复到输入框。`);
      }
      return false;
    } finally {
      window.clearInterval(statusTimer);
      isAskingRef.current = false;
      setIsAsking(false);
      setStatusMessage("");
    }
  }

  function clearError() {
    setError("");
  }

  return { isAsking, statusMessage, error, ask, clearError };
}
