"use client";

import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { Button } from "@/components/ui/Button";
import type { Folder } from "./types";

type FolderTreeProps = {
  folders: Folder[];
  selectedFolderId: string;
  isBusy: boolean;
  onSelectFolder: (folderId: string) => void;
  onCreateFolder: (name: string) => Promise<void>;
  onRenameFolder: (folderId: string, name: string) => Promise<void>;
  onDeleteFolder: (folderId: string) => Promise<void>;
};

export function FolderTree({
  folders,
  selectedFolderId,
  isBusy,
  onSelectFolder,
  onCreateFolder,
  onRenameFolder,
  onDeleteFolder,
}: FolderTreeProps) {
  const [newFolderName, setNewFolderName] = useState("");
  const [renameValue, setRenameValue] = useState("");
  const selectedFolder = folders.find((folder) => folder.id === selectedFolderId);

  useEffect(() => {
    setRenameValue(selectedFolder?.name || "");
  }, [selectedFolder?.name]);

  async function createFolder(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const name = newFolderName.trim();
    if (!name) return;
    await onCreateFolder(name);
    setNewFolderName("");
  }

  async function renameFolder(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedFolder) return;
    const name = renameValue.trim();
    if (!name || name === selectedFolder.name) return;
    await onRenameFolder(selectedFolder.id, name);
  }

  return (
    <aside className="panel library-sidebar">
      <div className="panel-heading">
        <div>
          <p className="section-label">文件夹</p>
          <h2>资料库</h2>
        </div>
      </div>

      <button
        type="button"
        className={`folder-row ${selectedFolderId ? "" : "selected"}`}
        onClick={() => onSelectFolder("")}
      >
        全部文件
      </button>

      <div className="folder-list">
        {folders.map((folder) => (
          <button
            type="button"
            className={`folder-row ${folder.id === selectedFolderId ? "selected" : ""}`}
            key={folder.id}
            onClick={() => onSelectFolder(folder.id)}
          >
            {folder.name}
          </button>
        ))}
      </div>

      <form className="folder-form" onSubmit={createFolder}>
        <input
          value={newFolderName}
          placeholder="新文件夹"
          onChange={(event) => setNewFolderName(event.target.value)}
        />
        <Button type="submit" variant="secondary" disabled={isBusy}>
          添加
        </Button>
      </form>

      {selectedFolder ? (
        <div className="folder-actions">
          <form className="folder-form" onSubmit={renameFolder}>
            <input
              value={renameValue}
              onChange={(event) => setRenameValue(event.target.value)}
            />
            <Button type="submit" variant="secondary" disabled={isBusy}>
              重命名
            </Button>
          </form>
          <Button
            type="button"
            variant="ghost"
            disabled={isBusy}
            onClick={() => onDeleteFolder(selectedFolder.id)}
          >
            删除空文件夹
          </Button>
        </div>
      ) : null}
    </aside>
  );
}
