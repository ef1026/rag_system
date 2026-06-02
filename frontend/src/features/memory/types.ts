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
  last_seen_at: string | null;
  created_at: string;
  updated_at: string;
};

export type MemoryExtractResponse = {
  created_count: number;
  memories: UserMemory[];
};
