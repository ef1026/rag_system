import type { Folder, ManagedFile } from "./types";

type FileTableProps = {
  files: ManagedFile[];
  folders: Folder[];
  selectedFileId: string;
  isLoading: boolean;
  onSelectFile: (fileId: string) => void;
};

export function FileTable({
  files,
  folders,
  selectedFileId,
  isLoading,
  onSelectFile,
}: FileTableProps) {
  const folderNames = new Map(folders.map((folder) => [folder.id, folder.name]));

  if (isLoading) {
    return <p className="empty-state">正在加载托管文件...</p>;
  }

  if (!files.length) {
    return <p className="empty-state">当前筛选条件下没有文件。</p>;
  }

  return (
    <div className="file-table-wrap">
      <table className="file-table">
        <thead>
          <tr>
            <th>名称</th>
            <th>课程</th>
            <th>文件夹</th>
            <th>标签</th>
            <th>状态</th>
          </tr>
        </thead>
        <tbody>
          {files.map((file) => (
            <tr
              className={file.id === selectedFileId ? "selected" : ""}
              key={file.id}
              onClick={() => onSelectFile(file.id)}
            >
              <td>
                <button type="button" className="file-name-button">
                  <strong>{file.display_name}</strong>
                  <span>{file.document_id}</span>
                </button>
                <div className="file-flags">
                  {file.pinned ? <span>置顶</span> : null}
                  {file.archived ? <span>已归档</span> : null}
                </div>
              </td>
              <td>{file.course || "-"}</td>
              <td>{file.folder_id ? folderNames.get(file.folder_id) || "缺失" : "-"}</td>
              <td>
                <div className="tag-row compact">
                  {file.tags.length ? (
                    file.tags.map((tag) => (
                      <span className="tag-chip" key={tag}>
                        {tag}
                      </span>
                    ))
                  ) : (
                    <span className="muted">-</span>
                  )}
                </div>
              </td>
              <td>
                <span className={`document-status status-${file.status || "unknown"}`}>
                  {statusLabel(file.status)}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function statusLabel(status: string | null | undefined) {
  if (status === "uploaded") return "待解析";
  if (status === "parsed") return "已解析";
  if (status === "indexing") return "索引中";
  if (status === "ready_for_chat") return "可提问";
  if (status === "partial_success") return "部分成功";
  if (status === "failed") return "失败";
  return "未知";
}
