"use client";

import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { Button } from "@/components/ui/Button";
import type { DocumentSummary } from "@/types/document";
import { parseTags, TagEditor } from "./TagEditor";
import type { Folder, ManagedFile, ManagedFileRegister } from "./types";

type RegisterDocumentDialogProps = {
  isOpen: boolean;
  documents: DocumentSummary[];
  managedFiles: ManagedFile[];
  folders: Folder[];
  isBusy: boolean;
  onClose: () => void;
  onRegister: (payload: ManagedFileRegister) => Promise<void>;
};

export function RegisterDocumentDialog({
  isOpen,
  documents,
  managedFiles,
  folders,
  isBusy,
  onClose,
  onRegister,
}: RegisterDocumentDialogProps) {
  const registeredIds = useMemo(
    () => new Set(managedFiles.map((file) => file.document_id)),
    [managedFiles],
  );
  const unregisteredDocuments = useMemo(
    () => documents.filter((document) => !registeredIds.has(document.id)),
    [documents, registeredIds],
  );
  const [selectedDocumentId, setSelectedDocumentId] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [folderId, setFolderId] = useState("");
  const [course, setCourse] = useState("");
  const [tagsText, setTagsText] = useState("");
  const [description, setDescription] = useState("");
  const [notes, setNotes] = useState("");

  const selectedDocument = unregisteredDocuments.find(
    (document) => document.id === selectedDocumentId,
  );

  useEffect(() => {
    if (!isOpen) return;
    const firstDocument = unregisteredDocuments[0];
    setSelectedDocumentId(firstDocument?.id || "");
    setDisplayName(firstDocument ? defaultDisplayName(firstDocument.name) : "");
    setFolderId("");
    setCourse("");
    setTagsText("");
    setDescription("");
    setNotes("");
  }, [isOpen, unregisteredDocuments]);

  function selectDocument(documentId: string) {
    const document = unregisteredDocuments.find((item) => item.id === documentId);
    setSelectedDocumentId(documentId);
    setDisplayName(document ? defaultDisplayName(document.name) : "");
  }

  async function registerDocument(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedDocument) return;
    await onRegister({
      document_id: selectedDocument.id,
      original_filename: selectedDocument.name,
      display_name: displayName || defaultDisplayName(selectedDocument.name),
      folder_id: folderId || null,
      tags: parseTags(tagsText),
      course: toNullable(course),
      description: toNullable(description),
      notes: toNullable(notes),
    });
  }

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop">
      <section className="panel register-dialog" role="dialog" aria-modal="true">
        <div className="panel-heading">
          <div>
            <p className="section-label">登记文档</p>
            <h2>已有 RAG 文档</h2>
          </div>
          <Button type="button" variant="ghost" onClick={onClose}>
            关闭
          </Button>
        </div>

        {!unregisteredDocuments.length ? (
          <p className="empty-state">当前所有文档都已登记。</p>
        ) : (
          <form className="register-form" onSubmit={registerDocument}>
            <label className="form-field">
              <span>文档</span>
              <select
                value={selectedDocumentId}
                onChange={(event) => selectDocument(event.target.value)}
              >
                {unregisteredDocuments.map((document) => (
                  <option value={document.id} key={document.id}>
                    {document.name}
                  </option>
                ))}
              </select>
            </label>

            <label className="form-field">
              <span>显示名称</span>
              <input
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
              />
            </label>

            <label className="form-field">
              <span>文件夹</span>
              <select
                value={folderId}
                onChange={(event) => setFolderId(event.target.value)}
              >
                <option value="">不放入文件夹</option>
                {folders.map((folder) => (
                  <option value={folder.id} key={folder.id}>
                    {folder.name}
                  </option>
                ))}
              </select>
            </label>

            <label className="form-field">
              <span>课程 / 主题</span>
              <input
                value={course}
                onChange={(event) => setCourse(event.target.value)}
              />
            </label>

            <TagEditor value={tagsText} onChange={setTagsText} />

            <label className="form-field">
              <span>描述</span>
              <textarea
                value={description}
                onChange={(event) => setDescription(event.target.value)}
              />
            </label>

            <label className="form-field">
              <span>备注</span>
              <textarea value={notes} onChange={(event) => setNotes(event.target.value)} />
            </label>

            <Button type="submit" variant="primary" disabled={isBusy || !selectedDocument}>
              登记
            </Button>
          </form>
        )}
      </section>
    </div>
  );
}

function defaultDisplayName(filename: string) {
  const index = filename.lastIndexOf(".");
  return index > 0 ? filename.slice(0, index) : filename;
}

function toNullable(value: string) {
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}
