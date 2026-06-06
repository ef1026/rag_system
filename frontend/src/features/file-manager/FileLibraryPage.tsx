"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/Button";
import { api as ragApi } from "@/lib/api";
import type { DocumentSummary } from "@/types/document";
import { fileManagerApi } from "./api";
import { FileDetailPanel } from "./FileDetailPanel";
import { FileTable } from "./FileTable";
import { FolderTree } from "./FolderTree";
import { RegisterDocumentDialog } from "./RegisterDocumentDialog";
import type {
  Folder,
  ManagedFile,
  ManagedFileFilters,
  ManagedFilePatch,
  ManagedFileRegister,
  TagSummary,
} from "./types";

type ArchiveFilter = "active" | "all" | "archived";
type PinFilter = "all" | "pinned" | "unpinned";

export function FileLibraryPage() {
  const [folders, setFolders] = useState<Folder[]>([]);
  const [files, setFiles] = useState<ManagedFile[]>([]);
  const [allFiles, setAllFiles] = useState<ManagedFile[]>([]);
  const [tags, setTags] = useState<TagSummary[]>([]);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedFolderId, setSelectedFolderId] = useState("");
  const [selectedFileId, setSelectedFileId] = useState("");
  const [tagFilter, setTagFilter] = useState("");
  const [courseFilter, setCourseFilter] = useState("");
  const [archiveFilter, setArchiveFilter] = useState<ArchiveFilter>("active");
  const [pinFilter, setPinFilter] = useState<PinFilter>("all");
  const [query, setQuery] = useState("");
  const [isFilesLoading, setIsFilesLoading] = useState(true);
  const [isBusy, setIsBusy] = useState(false);
  const [isRegisterOpen, setIsRegisterOpen] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const filters = useMemo<ManagedFileFilters>(() => {
    const nextFilters: ManagedFileFilters = {};
    if (selectedFolderId) nextFilters.folder_id = selectedFolderId;
    if (tagFilter) nextFilters.tag = tagFilter;
    if (courseFilter) nextFilters.course = courseFilter;
    if (archiveFilter === "active") nextFilters.archived = false;
    if (archiveFilter === "archived") nextFilters.archived = true;
    if (pinFilter === "pinned") nextFilters.pinned = true;
    if (pinFilter === "unpinned") nextFilters.pinned = false;
    if (query.trim()) nextFilters.q = query.trim();
    return nextFilters;
  }, [archiveFilter, courseFilter, pinFilter, query, selectedFolderId, tagFilter]);

  const refreshFiles = useCallback(async () => {
    setIsFilesLoading(true);
    setError("");
    try {
      const nextFiles = await fileManagerApi.files(filters);
      setFiles(nextFiles);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "文件加载失败。");
    } finally {
      setIsFilesLoading(false);
    }
  }, [filters]);

  const refreshMetadata = useCallback(async () => {
    setError("");
    try {
      const [nextFolders, nextTags, nextDocuments, nextAllFiles] = await Promise.all([
        fileManagerApi.folders(),
        fileManagerApi.tags(),
        ragApi.documents(),
        fileManagerApi.files(),
      ]);
      setFolders(nextFolders);
      setTags(nextTags);
      setDocuments(nextDocuments);
      setAllFiles(nextAllFiles);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "资料库元数据加载失败。");
    }
  }, []);

  useEffect(() => {
    void refreshMetadata();
  }, [refreshMetadata]);

  useEffect(() => {
    void refreshFiles();
  }, [refreshFiles]);

  useEffect(() => {
    if (selectedFileId && files.some((file) => file.id === selectedFileId)) return;
    setSelectedFileId(files[0]?.id || "");
  }, [files, selectedFileId]);

  const selectedFile =
    files.find((file) => file.id === selectedFileId) ||
    allFiles.find((file) => file.id === selectedFileId) ||
    null;

  const courseOptions = useMemo(
    () =>
      Array.from(
        new Set(
          allFiles
            .map((file) => file.course?.trim())
            .filter((course): course is string => Boolean(course)),
        ),
      ).sort((a, b) => a.localeCompare(b)),
    [allFiles],
  );

  async function reloadLibrary() {
    await Promise.all([refreshMetadata(), refreshFiles()]);
  }

  async function runLibraryAction(
    action: () => Promise<void>,
    successMessage: string,
  ) {
    setIsBusy(true);
    setError("");
    setNotice("");
    try {
      await action();
      await reloadLibrary();
      setNotice(successMessage);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "资料库操作失败。");
    } finally {
      setIsBusy(false);
    }
  }

  function clearFilters() {
    setSelectedFolderId("");
    setTagFilter("");
    setCourseFilter("");
    setArchiveFilter("active");
    setPinFilter("all");
    setQuery("");
  }

  return (
    <div className="library-page">
      <FolderTree
        folders={folders}
        selectedFolderId={selectedFolderId}
        isBusy={isBusy}
        onSelectFolder={setSelectedFolderId}
        onCreateFolder={(name) =>
          runLibraryAction(
            () => fileManagerApi.createFolder({ name }).then(() => undefined),
            "文件夹已创建。",
          )
        }
        onRenameFolder={(folderId, name) =>
          runLibraryAction(
            () => fileManagerApi.updateFolder(folderId, { name }).then(() => undefined),
            "文件夹已重命名。",
          )
        }
        onDeleteFolder={(folderId) =>
          runLibraryAction(
            () => fileManagerApi.deleteFolder(folderId).then(() => undefined),
            "文件夹已删除。",
          )
        }
      />

      <main className="panel library-main">
        <div className="panel-heading library-main-heading">
          <div>
            <p className="section-label">文件</p>
            <h2>托管文档</h2>
          </div>
          <div className="library-actions">
            <Button
              type="button"
              variant="secondary"
              disabled={isBusy}
              onClick={() =>
                runLibraryAction(
                  () => fileManagerApi.syncExistingDocuments().then(() => undefined),
                  "已有文档已同步。",
                )
              }
            >
              同步文档
            </Button>
            <Button
              type="button"
              variant="primary"
              disabled={isBusy}
              onClick={() => setIsRegisterOpen(true)}
            >
              登记
            </Button>
          </div>
        </div>

        {error ? <p className="error-text">{error}</p> : null}
        {notice ? <p className="chat-status">{notice}</p> : null}

        <div className="library-filters">
          <label className="form-field">
            <span>搜索</span>
            <input
              value={query}
              placeholder="名称、备注、标签"
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>
          <label className="form-field">
            <span>标签</span>
            <select value={tagFilter} onChange={(event) => setTagFilter(event.target.value)}>
              <option value="">全部标签</option>
              {tags.map((tag) => (
                <option value={tag.name} key={tag.name}>
                  {tag.name} ({tag.count})
                </option>
              ))}
            </select>
          </label>
          <label className="form-field">
            <span>课程</span>
            <select
              value={courseFilter}
              onChange={(event) => setCourseFilter(event.target.value)}
            >
              <option value="">全部课程</option>
              {courseOptions.map((course) => (
                <option value={course} key={course}>
                  {course}
                </option>
              ))}
            </select>
          </label>
          <label className="form-field">
            <span>归档</span>
            <select
              value={archiveFilter}
              onChange={(event) =>
                setArchiveFilter(event.target.value as ArchiveFilter)
              }
            >
              <option value="active">未归档</option>
              <option value="all">全部</option>
              <option value="archived">已归档</option>
            </select>
          </label>
          <label className="form-field">
            <span>置顶</span>
            <select
              value={pinFilter}
              onChange={(event) => setPinFilter(event.target.value as PinFilter)}
            >
              <option value="all">全部</option>
              <option value="pinned">已置顶</option>
              <option value="unpinned">未置顶</option>
            </select>
          </label>
          <Button type="button" variant="ghost" onClick={clearFilters}>
            清除筛选
          </Button>
        </div>

        <FileTable
          files={files}
          folders={folders}
          selectedFileId={selectedFileId}
          isLoading={isFilesLoading}
          onSelectFile={setSelectedFileId}
        />
      </main>

      <FileDetailPanel
        file={selectedFile}
        folders={folders}
        isBusy={isBusy}
        onSave={(fileId: string, payload: ManagedFilePatch) =>
          runLibraryAction(
            () => fileManagerApi.updateFile(fileId, payload).then(() => undefined),
            "文件元数据已保存。",
          )
        }
        onDelete={(fileId: string) =>
          runLibraryAction(
            async () => {
              await fileManagerApi.deleteFile(fileId);
              setSelectedFileId("");
            },
            "文件元数据已删除。",
          )
        }
      />

      <RegisterDocumentDialog
        isOpen={isRegisterOpen}
        documents={documents}
        managedFiles={allFiles}
        folders={folders}
        isBusy={isBusy}
        onClose={() => setIsRegisterOpen(false)}
        onRegister={(payload: ManagedFileRegister) =>
          runLibraryAction(
            async () => {
              await fileManagerApi.register(payload);
              setIsRegisterOpen(false);
            },
            "文档已登记。",
          )
        }
      />
    </div>
  );
}
