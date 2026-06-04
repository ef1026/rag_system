export type UserMemory = {
  id: string;
  profile_id: string;
  memory_type: string;
  key: string | null;
  value: string;
  confidence: number;
  source_conversation_id: string | null;
  evidence: string | null;
  status: "candidate" | "active" | "dismissed" | "expired" | "deleted" | string;
  sensitivity: string;
  scope_type: "global" | "course" | "document" | "conversation";
  scope_id: string | null;
  evidence_message_ids: string[];
  last_seen_at: string | null;
  last_confirmed_at: string | null;
  expires_at: string | null;
  auto_apply: boolean;
  created_at: string;
  updated_at: string;
};

export type UserMemoryPatch = Partial<{
  memory_type: string;
  key: string | null;
  value: string;
  confidence: number;
  evidence: string | null;
  status: string;
  sensitivity: string;
  scope_type: "global" | "course" | "document" | "conversation";
  scope_id: string | null;
  evidence_message_ids: string[];
  expires_at: string | null;
  auto_apply: boolean;
}>;

export type MemoryExtractResponse = {
  created_count: number;
  memories: UserMemory[];
};
