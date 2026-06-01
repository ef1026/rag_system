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
};
