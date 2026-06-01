export type Folder = {
  id: string;
  profile_id: string;
  parent_id: string | null;
  name: string;
  sort_order: number;
  created_at: string;
  updated_at: string;
};

export type ManagedFile = {
  id: string;
  profile_id: string;
  document_id: string;
  original_filename: string | null;
  display_name: string;
  folder_id: string | null;
  tags: string[];
  course: string | null;
  description: string | null;
  notes: string | null;
  pinned: boolean;
  archived: boolean;
  status: string | null;
  source_type: string;
  created_at: string;
  updated_at: string;
};

export type ManagedFileRegister = {
  document_id: string;
  original_filename?: string | null;
  display_name?: string | null;
  folder_id?: string | null;
  tags?: string[] | null;
  course?: string | null;
  description?: string | null;
  notes?: string | null;
};

export type ManagedFilePatch = {
  display_name?: string;
  folder_id?: string | null;
  tags?: string[];
  course?: string | null;
  description?: string | null;
  notes?: string | null;
  pinned?: boolean;
  archived?: boolean;
};

export type ManagedFileFilters = {
  folder_id?: string;
  tag?: string;
  course?: string;
  archived?: boolean;
  pinned?: boolean;
  q?: string;
};

export type RegisterManagedFileResponse = {
  file: ManagedFile;
  created: boolean;
};

export type SyncExistingDocumentsResponse = {
  created_count: number;
  existing_count: number;
  files: ManagedFile[];
};

export type TagSummary = {
  name: string;
  count: number;
};
