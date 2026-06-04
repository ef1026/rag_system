import type { UserMemory } from "@/features/memory/types";

export type UserProfile = {
  id: string;
  display_name: string;
  role: string | null;
  education_level: string | null;
  major: string | null;
  learning_goals: string[];
  preferred_language: string | null;
  answer_style: string | null;
  math_level: string | null;
  coding_level: string | null;
  default_depth: string | null;
  citation_preference: string | null;
  agents_md: string | null;
  created_at: string;
  updated_at: string;
};

export type UserProfileInput = {
  display_name: string;
  role: string | null;
  education_level: string | null;
  major: string | null;
  learning_goals: string[];
  preferred_language: string | null;
  answer_style: string | null;
  math_level: string | null;
  coding_level: string | null;
  default_depth: string | null;
  citation_preference: string | null;
  agents_md: string | null;
};

export type ProfilePromptContextResponse = {
  profile_id: string;
  prompt_context: string;
  selected_level: string;
  effective_agents_md: string;
  agents_md_editable: boolean;
  builtin_agents_md: Record<string, string | undefined>;
  used_profile_fields: string[];
  used_memories: UserMemory[];
  retrieval_hints: string[];
  warnings: string[];
};

export type PersonalizationPreviewRequest = {
  question?: string;
  document_ids?: string[];
  level: string;
  conversation_id?: string;
  use_profile?: boolean;
  use_memory?: boolean;
};

export type PersonalizationPreviewResponse = ProfilePromptContextResponse;

export type ProfileFeedbackRequest = {
  conversation_id: string;
  message_id: string;
  rating?: "helpful" | "not_helpful";
  difficulty?: "too_easy" | "too_hard";
  style_feedback?: "needs_examples" | "needs_derivation" | "more_concise";
  note?: string;
};

export type ProfileFeedbackResponse = {
  ok: boolean;
  created_memory_candidates: UserMemory[];
};
