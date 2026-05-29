"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { DocumentSummary, SourceItem } from "@/types/document";

const SELECTED_DOCUMENT_KEY = "rag-selected-document:v1";

export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedId, setSelectedIdState] = useState<string>("");
  const [sources, setSources] = useState<SourceItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const [warmingDocumentId, setWarmingDocumentId] = useState("");
  const [warmedDocuments, setWarmedDocuments] = useState<Record<string, boolean>>({});
  const [warmupError, setWarmupError] = useState("");
  const [processNotice, setProcessNotice] = useState("");
  const [error, setError] = useState<string>("");

  const refresh = useCallback(async () => {
    setIsLoading(true);
    setError("");
    setProcessNotice("");
    try {
      const nextDocuments = await api.documents();
      setDocuments(nextDocuments);
      setSelectedIdState((current) => {
        const storedSelectedId = readSelectedDocumentId();
        const candidate = current || storedSelectedId;
        if (candidate) {
          const nextSelectedId = nextDocuments.some(
            (document) => document.id === candidate,
          )
            ? candidate
            : "";
          persistSelectedDocumentId(nextSelectedId);
          return nextSelectedId;
        }

        const nextSelectedId = nextDocuments[0]?.id || "";
        persistSelectedDocumentId(nextSelectedId);
        return nextSelectedId;
      });
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "文档列表加载失败");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const processSelected = useCallback(async () => {
    if (!selectedId) return;
    setIsProcessing(true);
    setError("");
    setProcessNotice("");
    try {
      const result = await api.process(selectedId);
      const limitSkipCount = result.skipped_by_reason?.limit_skip || 0;
      setSources(result.sources);
      setDocuments((current) =>
        current.map((document) =>
          document.id === result.document.id ? result.document : document,
        ),
      );
      setWarmedDocuments((current) => ({
        ...current,
        [result.document.id]: true,
      }));
      if (limitSkipCount > 0) {
        setProcessNotice("部分多模态内容因数量上限被跳过；文档仍可提问。");
      }
    } catch (nextError) {
      setProcessNotice("");
      setError(nextError instanceof Error ? nextError.message : "文档解析失败");
    } finally {
      setIsProcessing(false);
    }
  }, [selectedId]);

  const clearRuntime = useCallback(async () => {
    setError("");
    setProcessNotice("");
    try {
      await api.clearRuntime();
      setSources([]);
      setWarmedDocuments({});
      setWarmupError("");
      await refresh();
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "缓存清理失败");
    }
  }, [refresh]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const setSelectedId = useCallback((documentId: string) => {
    setSelectedIdState(documentId);
    persistSelectedDocumentId(documentId);
  }, []);

  const selectedDocument = documents.find((document) => document.id === selectedId);

  useEffect(() => {
    if (
      !selectedId ||
      !selectedDocument ||
      selectedDocument.status !== "ready_for_chat" ||
      warmedDocuments[selectedId]
    ) {
      return;
    }

    let cancelled = false;
    setWarmupError("");
    setWarmingDocumentId(selectedId);
    api
      .warmup(selectedId)
      .then(() => {
        if (cancelled) return;
        setWarmedDocuments((current) => ({
          ...current,
          [selectedId]: true,
        }));
      })
      .catch((nextError: unknown) => {
        if (cancelled) return;
        console.warn("[documents] warmup failed:", nextError);
        setWarmupError(
          nextError instanceof Error ? nextError.message : "知识库预热失败",
        );
      })
      .finally(() => {
        if (cancelled) return;
        setWarmingDocumentId((current) => (current === selectedId ? "" : current));
      });

    return () => {
      cancelled = true;
    };
  }, [documents, selectedDocument, selectedId, warmedDocuments]);

  return {
    documents,
    selectedId,
    selectedDocument,
    sources,
    isLoading,
    isProcessing,
    warmingDocumentId,
    warmupError,
    processNotice,
    error,
    setSelectedId,
    refresh,
    processSelected,
    clearRuntime,
  };
}

function readSelectedDocumentId() {
  if (typeof window === "undefined") return "";

  try {
    return window.localStorage.getItem(SELECTED_DOCUMENT_KEY) || "";
  } catch {
    return "";
  }
}

function persistSelectedDocumentId(documentId: string) {
  if (typeof window === "undefined") return;

  try {
    if (documentId) {
      window.localStorage.setItem(SELECTED_DOCUMENT_KEY, documentId);
    } else {
      window.localStorage.removeItem(SELECTED_DOCUMENT_KEY);
    }
  } catch {
    // Selection persistence is best effort; document loading should continue.
  }
}
