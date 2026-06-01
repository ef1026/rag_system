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
    return <p className="empty-state">Loading managed files...</p>;
  }

  if (!files.length) {
    return <p className="empty-state">No files match the current filters.</p>;
  }

  return (
    <div className="file-table-wrap">
      <table className="file-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Course</th>
            <th>Folder</th>
            <th>Tags</th>
            <th>Status</th>
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
                  {file.pinned ? <span>Pinned</span> : null}
                  {file.archived ? <span>Archived</span> : null}
                </div>
              </td>
              <td>{file.course || "-"}</td>
              <td>{file.folder_id ? folderNames.get(file.folder_id) || "Missing" : "-"}</td>
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
                  {file.status || "unknown"}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
