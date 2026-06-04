"use client";

import { useCallback, useEffect, useState } from "react";
import type { FormEvent } from "react";
import { Button } from "@/components/ui/Button";
import { MemoryReviewPanel } from "@/features/memory/MemoryReviewPanel";
import type { UserMemory } from "@/features/memory/types";
import type { AnswerLevel } from "@/types/chat";
import { profileApi } from "./api";
import { ProfilePromptPreview } from "./ProfilePromptPreview";
import type {
  PersonalizationPreviewResponse,
  UserProfile,
  UserProfileInput,
} from "./types";

type TextFieldKey = Exclude<
  keyof UserProfileInput,
  "learning_goals" | "agents_md" | "default_depth"
>;

const textFields: Array<{
  key: TextFieldKey;
  label: string;
  multiline?: boolean;
}> = [
  { key: "display_name", label: "Display name" },
  { key: "role", label: "Role" },
  { key: "education_level", label: "Education level" },
  { key: "major", label: "Major / subject area" },
  { key: "preferred_language", label: "Preferred language" },
  { key: "answer_style", label: "Answer style" },
  { key: "math_level", label: "Math level" },
  { key: "coding_level", label: "Coding level" },
  { key: "citation_preference", label: "Citation preference", multiline: true },
];

const depthLevels: Array<{ value: AnswerLevel; label: string }> = [
  { value: "beginner", label: "入门" },
  { value: "undergraduate", label: "本科" },
  { value: "expert", label: "专家" },
  { value: "custom", label: "自定义" },
];

const emptyProfileInput: UserProfileInput = {
  display_name: "",
  role: null,
  education_level: null,
  major: null,
  learning_goals: [],
  preferred_language: null,
  answer_style: null,
  math_level: null,
  coding_level: null,
  default_depth: null,
  citation_preference: null,
  agents_md: null,
};

export function ProfileEditor() {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [form, setForm] = useState<UserProfileInput>(emptyProfileInput);
  const [goalsText, setGoalsText] = useState("");
  const [customAgentsText, setCustomAgentsText] = useState("");
  const [builtinAgents, setBuiltinAgents] = useState<Record<string, string | undefined>>({});
  const [promptContext, setPromptContext] = useState("");
  const [retrievalHints, setRetrievalHints] = useState<string[]>([]);
  const [usedProfileFields, setUsedProfileFields] = useState<string[]>([]);
  const [usedMemories, setUsedMemories] = useState<UserMemory[]>([]);
  const [previewWarnings, setPreviewWarnings] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const applyPreview = useCallback((nextPrompt: PersonalizationPreviewResponse) => {
    setBuiltinAgents(nextPrompt.builtin_agents_md);
    setPromptContext(nextPrompt.prompt_context);
    setRetrievalHints(nextPrompt.retrieval_hints || []);
    setUsedProfileFields(nextPrompt.used_profile_fields || []);
    setUsedMemories(nextPrompt.used_memories || []);
    setPreviewWarnings(nextPrompt.warnings || []);
  }, []);

  const loadProfile = useCallback(async () => {
    setIsLoading(true);
    setError("");
    try {
      const nextProfile = await profileApi.get();
      const nextPrompt = await profileApi.personalizationPreview({
        level: nextProfile.default_depth || "undergraduate",
        use_profile: true,
        use_memory: true,
      });
      setProfile(nextProfile);
      setForm(toProfileInput(nextProfile));
      setGoalsText(nextProfile.learning_goals.join("\n"));
      setCustomAgentsText(nextProfile.agents_md || "");
      applyPreview(nextPrompt);
    } catch (nextError) {
      setError(
        nextError instanceof Error ? nextError.message : "Profile loading failed.",
      );
    } finally {
      setIsLoading(false);
    }
  }, [applyPreview]);

  useEffect(() => {
    void loadProfile();
  }, [loadProfile]);

  async function saveProfile(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSaving(true);
    setError("");
    setNotice("");
    try {
      const payload: UserProfileInput = {
        ...form,
        agents_md: customAgentsText,
        learning_goals: parseGoals(goalsText),
      };
      const savedProfile = await profileApi.save(payload);
      const nextPrompt = await profileApi.personalizationPreview({
        level: savedProfile.default_depth || "undergraduate",
        use_profile: true,
        use_memory: true,
      });
      setProfile(savedProfile);
      setForm(toProfileInput(savedProfile));
      setGoalsText(savedProfile.learning_goals.join("\n"));
      setCustomAgentsText(savedProfile.agents_md || "");
      applyPreview(nextPrompt);
      setNotice("Profile saved.");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Profile save failed.");
    } finally {
      setIsSaving(false);
    }
  }

  function updateField(key: TextFieldKey, value: string) {
    setForm((current) => ({
      ...current,
      [key]: key === "display_name" ? value : toNullableText(value),
    }) as UserProfileInput);
  }

  async function updateDefaultDepth(level: AnswerLevel) {
    setForm((current) => ({
      ...current,
      default_depth: level,
    }));
    setError("");
    try {
      const nextPrompt = await profileApi.personalizationPreview({
        level,
        use_profile: true,
        use_memory: true,
      });
      applyPreview(nextPrompt);
    } catch (nextError) {
      setError(
        nextError instanceof Error ? nextError.message : "Profile preview failed.",
      );
    }
  }

  if (isLoading && !profile) {
    return <p className="empty-state">Loading profile...</p>;
  }

  const selectedDepth = normalizeDepth(form.default_depth);
  const agentsEditable = selectedDepth === "custom";
  const displayedAgentsText = agentsEditable
    ? customAgentsText
    : builtinAgents[selectedDepth] || "";

  return (
    <div className="profile-page">
      <form className="panel profile-editor" onSubmit={saveProfile}>
        <div className="panel-heading">
          <div>
            <p className="section-label">User Profile</p>
            <h2>Learning preferences</h2>
          </div>
          <Button type="submit" variant="primary" disabled={isSaving}>
            {isSaving ? "Saving..." : "Save"}
          </Button>
        </div>

        {error ? <p className="error-text">{error}</p> : null}
        {notice ? <p className="chat-status">{notice}</p> : null}

        <div className="profile-form-grid">
          {textFields.map((field) => (
            <label className="form-field" key={field.key}>
              <span>{field.label}</span>
              {field.multiline ? (
                <textarea
                  value={form[field.key] ?? ""}
                  onChange={(event) => updateField(field.key, event.target.value)}
                />
              ) : (
                <input
                  value={form[field.key] ?? ""}
                  onChange={(event) => updateField(field.key, event.target.value)}
                />
              )}
            </label>
          ))}
          <div className="form-field profile-depth-field">
            <span>Default depth</span>
            <div className="segmented-control profile-depth-control" aria-label="Default depth">
              {depthLevels.map((item) => (
                <button
                  type="button"
                  key={item.value}
                  className={selectedDepth === item.value ? "active" : ""}
                  onClick={() => void updateDefaultDepth(item.value)}
                >
                  {depthLabel(item.value)}
                </button>
              ))}
            </div>
          </div>
          <label className="form-field profile-goals-field">
            <span>Learning goals</span>
            <textarea
              value={goalsText}
              onChange={(event) => setGoalsText(event.target.value)}
            />
          </label>
          <label className="form-field profile-agents-field">
            <span>AGENTS.md</span>
            <textarea
              value={displayedAgentsText}
              readOnly={!agentsEditable}
              onChange={(event) =>
                setCustomAgentsText(event.target.value)
              }
            />
            <small className="field-help">
              {agentsEditable
                ? "当前为自定义深度，保存后 Chat 的自定义模式会使用这段提示词。"
                : "当前为内置深度，AGENTS.md 仅用于预览；选择自定义后可编辑。"}
            </small>
          </label>
        </div>
      </form>

      <ProfilePromptPreview
        promptContext={promptContext}
        retrievalHints={retrievalHints}
        usedProfileFields={usedProfileFields}
        usedMemories={usedMemories}
        warnings={previewWarnings}
        isLoading={isLoading || isSaving}
      />
      <MemoryReviewPanel />
    </div>
  );
}

function toProfileInput(profile: UserProfile): UserProfileInput {
  return {
    display_name: profile.display_name,
    role: profile.role,
    education_level: profile.education_level,
    major: profile.major,
    learning_goals: profile.learning_goals,
    preferred_language: profile.preferred_language,
    answer_style: profile.answer_style,
    math_level: profile.math_level,
    coding_level: profile.coding_level,
    default_depth: profile.default_depth,
    citation_preference: profile.citation_preference,
    agents_md: profile.agents_md,
  };
}

function normalizeDepth(value: string | null | undefined): AnswerLevel {
  return value === "beginner" ||
    value === "undergraduate" ||
    value === "expert" ||
    value === "custom"
    ? value
    : "undergraduate";
}

function depthLabel(value: AnswerLevel) {
  if (value === "beginner") return "入门";
  if (value === "expert") return "专家";
  if (value === "custom") return "自定义";
  return "本科";
}

function parseGoals(value: string) {
  const seen = new Set<string>();
  const goals: string[] = [];
  value
    .split(/\r?\n|,/)
    .map((item) => item.trim())
    .filter(Boolean)
    .forEach((item) => {
      const key = item.toLowerCase();
      if (!seen.has(key)) {
        goals.push(item);
        seen.add(key);
      }
    });
  return goals;
}

function toNullableText(value: string) {
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}
