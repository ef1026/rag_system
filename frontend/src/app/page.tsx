"use client";

import { useEffect, useState } from "react";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { DocumentList } from "@/components/document/DocumentList";
import { UploadDropzone } from "@/components/document/UploadDropzone";
import { AppShell } from "@/components/layout/AppShell";
import { SourceList } from "@/components/sources/SourceList";
import { Button } from "@/components/ui/Button";
import { api } from "@/lib/api";
import { useChat } from "@/hooks/useChat";
import { useDocuments } from "@/hooks/useDocuments";
import { useUpload } from "@/hooks/useUpload";
import type { ApiHealth } from "@/types/api";

export default function Home() {
  const documents = useDocuments();
  const chat = useChat();
  const upload = useUpload((document) => {
    void documents.refresh();
    documents.setSelectedId(document.id);
  });
  const [health, setHealth] = useState<ApiHealth | null>(null);
  const [healthError, setHealthError] = useState("");

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((error: unknown) => {
        setHealthError(error instanceof Error ? error.message : "后端连接失败");
      });
  }, []);

  return (
    <AppShell>
      <section className="runtime-strip">
        <span className={health?.ok ? "runtime-dot ready" : "runtime-dot"} />
        <span>
          {health
            ? `后端已连接 · MinerU ${health.mineru_backend}/${health.mineru_device}`
            : healthError || "正在连接后端..."}
        </span>
        {health && !health.has_api_key ? (
          <strong>未检测到 Qwen API key</strong>
        ) : null}
      </section>

      <div className="workspace-grid">
        <aside className="left-rail">
          <UploadDropzone
            disabled={upload.isUploading}
            onUpload={(file) => void upload.upload(file)}
          />
          {upload.error ? <p className="error-text">{upload.error}</p> : null}
          {documents.error ? <p className="error-text">{documents.error}</p> : null}

          <section className="panel">
            <div className="panel-heading">
              <div>
                <p className="section-label">Documents</p>
                <h2>文档队列</h2>
              </div>
              <Button type="button" variant="ghost" onClick={() => void documents.refresh()}>
                刷新
              </Button>
            </div>
            <DocumentList
              documents={documents.documents}
              selectedId={documents.selectedId}
              isLoading={documents.isLoading}
              onSelect={documents.setSelectedId}
            />
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

        <section className="main-column">
          <ChatPanel
            messages={chat.messages}
            selectedDocumentId={documents.selectedId}
            isAsking={chat.isAsking}
            onAsk={chat.ask}
          />
          {chat.error ? <p className="error-text">{chat.error}</p> : null}
        </section>

        <aside className="right-rail panel">
          <div className="panel-heading">
            <div>
              <p className="section-label">Sources</p>
              <h2>引用片段</h2>
            </div>
          </div>
          <SourceList sources={documents.sources} />
        </aside>
      </div>
    </AppShell>
  );
}
