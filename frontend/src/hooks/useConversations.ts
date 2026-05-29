"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  createConversationTitle,
  loadConversations,
  saveConversations,
  trimConversationMessages,
  trimConversations,
} from "@/lib/conversationStorage";
import type { AnswerLevel, ChatMessage, ChatMode, Conversation } from "@/types/chat";
import type { DocumentSummary } from "@/types/document";

const DEFAULT_LEVEL: AnswerLevel = "undergraduate";
const DEFAULT_CHAT_MODE: ChatMode = "multimodal";
const DEFAULT_RAG_MODE = "hybrid";

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState("");
  const [hasLoaded, setHasLoaded] = useState(false);

  useEffect(() => {
    const loadedConversations = loadConversations();
    setConversations(loadedConversations);
    setActiveConversationId(loadedConversations[0]?.id || "");
    setHasLoaded(true);
  }, []);

  useEffect(() => {
    if (!hasLoaded) return;
    saveConversations(conversations);
  }, [conversations, hasLoaded]);

  const activeConversation = useMemo(
    () =>
      conversations.find((conversation) => conversation.id === activeConversationId) ||
      null,
    [activeConversationId, conversations],
  );

  const createConversation = useCallback((documents: DocumentSummary[] = []) => {
    const now = new Date().toISOString();
    const nextConversation: Conversation = {
      id: crypto.randomUUID(),
      title: "新对话",
      documentIds: documents.map((document) => document.id),
      documentNames: documents.map((document) => document.name),
      mode: DEFAULT_RAG_MODE,
      chatMode: DEFAULT_CHAT_MODE,
      level: DEFAULT_LEVEL,
      messages: [],
      createdAt: now,
      updatedAt: now,
    };

    setConversations((current) => trimConversations([nextConversation, ...current]));
    setActiveConversationId(nextConversation.id);
    return nextConversation.id;
  }, []);

  const selectConversation = useCallback((conversationId: string) => {
    setActiveConversationId(conversationId);
  }, []);

  const deleteConversation = useCallback((conversationId: string) => {
    setConversations((current) => {
      const nextConversations = current.filter(
        (conversation) => conversation.id !== conversationId,
      );
      setActiveConversationId((currentActiveId) => {
        if (currentActiveId !== conversationId) return currentActiveId;
        return nextConversations[0]?.id || "";
      });
      return nextConversations;
    });
  }, []);

  const clearConversations = useCallback(() => {
    setConversations([]);
    setActiveConversationId("");
  }, []);

  const updateConversation = useCallback(
    (
      conversationId: string,
      update: (conversation: Conversation) => Conversation,
    ) => {
      setConversations((current) =>
        trimConversations(
          current.map((conversation) =>
            conversation.id === conversationId ? update(conversation) : conversation,
          ),
        ),
      );
    },
    [],
  );

  const bindDocumentToConversation = useCallback(
    (conversationId: string, document: DocumentSummary) => {
      updateConversation(conversationId, (conversation) => {
        if (conversation.documentIds.includes(document.id)) return conversation;

        return {
          ...conversation,
          documentIds: [...conversation.documentIds, document.id],
          documentNames: [...conversation.documentNames, document.name],
          updatedAt: new Date().toISOString(),
        };
      });
    },
    [updateConversation],
  );

  const setConversationDocuments = useCallback(
    (conversationId: string, documents: DocumentSummary[]) => {
      updateConversation(conversationId, (conversation) => ({
        ...conversation,
        documentIds: documents.map((document) => document.id),
        documentNames: documents.map((document) => document.name),
        updatedAt: new Date().toISOString(),
      }));
    },
    [updateConversation],
  );

  const appendMessage = useCallback(
    (conversationId: string, message: ChatMessage) => {
      updateConversation(conversationId, (conversation) => {
        const messages = trimConversationMessages([...conversation.messages, message]);
        const title =
          conversation.title === "新对话" && message.role === "user"
            ? createConversationTitle(message.content)
            : conversation.title;

        return {
          ...conversation,
          title,
          messages,
          updatedAt: new Date().toISOString(),
        };
      });
    },
    [updateConversation],
  );

  const updateMessage = useCallback(
    (
      conversationId: string,
      messageId: string,
      update: (message: ChatMessage) => ChatMessage,
    ) => {
      updateConversation(conversationId, (conversation) => ({
        ...conversation,
        messages: trimConversationMessages(
          conversation.messages.map((message) =>
            message.id === messageId ? update(message) : message,
          ),
        ),
        updatedAt: new Date().toISOString(),
      }));
    },
    [updateConversation],
  );

  const clearConversationMessages = useCallback(
    (conversationId: string) => {
      updateConversation(conversationId, (conversation) => ({
        ...conversation,
        messages: [],
        updatedAt: new Date().toISOString(),
      }));
    },
    [updateConversation],
  );

  const setConversationLevel = useCallback(
    (conversationId: string, level: AnswerLevel) => {
      updateConversation(conversationId, (conversation) => ({
        ...conversation,
        level,
        updatedAt: new Date().toISOString(),
      }));
    },
    [updateConversation],
  );

  const setConversationChatMode = useCallback(
    (conversationId: string, chatMode: ChatMode) => {
      updateConversation(conversationId, (conversation) => ({
        ...conversation,
        chatMode,
        updatedAt: new Date().toISOString(),
      }));
    },
    [updateConversation],
  );

  return {
    conversations,
    activeConversation,
    activeConversationId,
    createConversation,
    selectConversation,
    deleteConversation,
    clearConversations,
    bindDocumentToConversation,
    setConversationDocuments,
    appendMessage,
    updateMessage,
    clearConversationMessages,
    setConversationLevel,
    setConversationChatMode,
  };
}
