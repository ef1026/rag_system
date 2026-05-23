"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { DocumentSummary, SourceItem } from "@/types/document";

export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [sources, setSources] = useState<SourceItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string>("");

  const refresh = useCallback(async () => {
    setIsLoading(true);
    setError("");
    try {
      const nextDocuments = await api.documents();
      setDocuments(nextDocuments);
      setSelectedId((current) => current || nextDocuments[0]?.id || "");
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
    try {
      const result = await api.process(selectedId);
      setSources(result.sources);
      setDocuments((current) =>
        current.map((document) =>
          document.id === result.document.id ? result.document : document,
        ),
      );
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "文档解析失败");
    } finally {
      setIsProcessing(false);
    }
  }, [selectedId]);

  const clearRuntime = useCallback(async () => {
    setError("");
    try {
      await api.clearRuntime();
      setSources([]);
      await refresh();
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "缓存清理失败");
    }
  }, [refresh]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return {
    documents,
    selectedId,
    selectedDocument: documents.find((document) => document.id === selectedId),
    sources,
    isLoading,
    isProcessing,
    error,
    setSelectedId,
    refresh,
    processSelected,
    clearRuntime,
  };
}
