"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import type { DocumentSummary } from "@/types/document";

export function useUpload(onUploaded: (document: DocumentSummary) => void) {
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string>("");

  async function upload(file: File) {
    setIsUploading(true);
    setError("");
    try {
      const result = await api.upload(file);
      onUploaded(result.document);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "上传失败");
    } finally {
      setIsUploading(false);
    }
  }

  return { upload, isUploading, error };
}
