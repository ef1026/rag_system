"use client";

import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { Button } from "@/components/ui/Button";
import { parseTags, TagEditor } from "./TagEditor";
import type { Folder, ManagedFile, ManagedFilePatch } from "./types";

type FileDetailPanelProps = {
  file: ManagedFile | null;
  folders: Folder[];
  isBusy: boolean;
  onSave: (fileId: string, payload: ManagedFilePatch) => Promise<void>;
  onDelete: (fileId: string) => Promise<void>;
};

type DetailForm = {
  displayName: string;
  folderId: string;
  course: string;
  tagsText: string;
  description: string;
  notes: string;
  pinned: boolean;
  archived: boolean;
};

const emptyForm: DetailForm = {
  displayName: "",
  folderId: "",
  course: "",
  tagsText: "",
  description: "",
  notes: "",
  pinned: false,
  archived: false,
};

export function FileDetailPanel({
  file,
  folders,
  isBusy,
  onSave,
  onDelete,
}: FileDetailPanelProps) {
  const [form, setForm] = useState<DetailForm>(emptyForm);

  useEffect(() => {
    if (!file) {
      setForm(emptyForm);
      return;
    }
    setForm({
      displayName: file.display_name,
      folderId: file.folder_id || "",
      course: file.course || "",
      tagsText: file.tags.join(", "),
      description: file.description || "",
      notes: file.notes || "",
      pinned: file.pinned,
      archived: file.archived,
    });
  }, [file]);

  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;
    await onSave(file.id, {
      display_name: form.displayName,
      folder_id: form.folderId || null,
      course: toNullable(form.course),
      tags: parseTags(form.tagsText),
      description: toNullable(form.description),
      notes: toNullable(form.notes),
      pinned: form.pinned,
      archived: form.archived,
    });
  }

  if (!file) {
    return (
      <aside className="panel library-detail">
        <p className="empty-state">Select a managed file to edit metadata.</p>
      </aside>
    );
  }

  return (
    <aside className="panel library-detail">
      <div className="panel-heading">
        <div>
          <p className="section-label">File Details</p>
          <h2>Edit metadata</h2>
        </div>
      </div>

      <form className="detail-form" onSubmit={save}>
        <label className="form-field">
          <span>Display name</span>
          <input
            value={form.displayName}
            onChange={(event) =>
              setForm((current) => ({ ...current, displayName: event.target.value }))
            }
          />
        </label>

        <label className="form-field">
          <span>Folder</span>
          <select
            value={form.folderId}
            onChange={(event) =>
              setForm((current) => ({ ...current, folderId: event.target.value }))
            }
          >
            <option value="">No folder</option>
            {folders.map((folder) => (
              <option value={folder.id} key={folder.id}>
                {folder.name}
              </option>
            ))}
          </select>
        </label>

        <label className="form-field">
          <span>Course / subject</span>
          <input
            value={form.course}
            onChange={(event) =>
              setForm((current) => ({ ...current, course: event.target.value }))
            }
          />
        </label>

        <TagEditor
          value={form.tagsText}
          onChange={(value) => setForm((current) => ({ ...current, tagsText: value }))}
        />

        <label className="form-field">
          <span>Description</span>
          <textarea
            value={form.description}
            onChange={(event) =>
              setForm((current) => ({
                ...current,
                description: event.target.value,
              }))
            }
          />
        </label>

        <label className="form-field">
          <span>Notes</span>
          <textarea
            value={form.notes}
            onChange={(event) =>
              setForm((current) => ({ ...current, notes: event.target.value }))
            }
          />
        </label>

        <div className="checkbox-row">
          <label>
            <input
              type="checkbox"
              checked={form.pinned}
              onChange={(event) =>
                setForm((current) => ({
                  ...current,
                  pinned: event.target.checked,
                }))
              }
            />
            Pinned
          </label>
          <label>
            <input
              type="checkbox"
              checked={form.archived}
              onChange={(event) =>
                setForm((current) => ({
                  ...current,
                  archived: event.target.checked,
                }))
              }
            />
            Archived
          </label>
        </div>

        <div className="detail-actions">
          <Button type="submit" variant="primary" disabled={isBusy}>
            Save file
          </Button>
          <Button
            type="button"
            variant="ghost"
            disabled={isBusy}
            onClick={() => onDelete(file.id)}
          >
            Delete metadata
          </Button>
        </div>
      </form>
    </aside>
  );
}

function toNullable(value: string) {
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}
