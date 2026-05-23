"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import type { AnswerLevel, ChatMessage } from "@/types/chat";

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isAsking, setIsAsking] = useState(false);
  const [error, setError] = useState<string>("");

  async function ask(question: string, documentId: string, level: AnswerLevel) {
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) return;

    setIsAsking(true);
    setError("");
    setMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), role: "user", content: trimmedQuestion },
    ]);

    try {
      const response = await api.chat({
        question: trimmedQuestion,
        document_id: documentId || undefined,
        level,
        mode: "hybrid",
      });
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: response.answer,
          sources: response.sources,
        },
      ]);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "问答生成失败");
    } finally {
      setIsAsking(false);
    }
  }

  return { messages, isAsking, error, ask };
}
