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
  { key: "display_name", label: "显示名称" },
  { key: "role", label: "身份 / 角色" },
  { key: "education_level", label: "教育阶段" },
  { key: "major", label: "专业 / 研究方向" },
  { key: "preferred_language", label: "偏好语言" },
  { key: "answer_style", label: "回答风格" },
  { key: "math_level", label: "数学基础" },
  { key: "coding_level", label: "编程基础" },
  { key: "citation_preference", label: "引用偏好", multiline: true },
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
      setError(nextError instanceof Error ? nextError.message : "画像加载失败。");
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
      setNotice("画像已保存。");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "画像保存失败。");
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
      setError(nextError instanceof Error ? nextError.message : "画像预览失败。");
    }
  }

  if (isLoading && !profile) {
    return <p className="empty-state">正在加载画像...</p>;
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
            <p className="section-label">用户画像</p>
            <h2>学习偏好</h2>
          </div>
          <Button type="submit" variant="primary" disabled={isSaving}>
            {isSaving ? "保存中..." : "保存"}
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
            <span>默认回答深度</span>
            <div className="segmented-control profile-depth-control" aria-label="默认回答深度">
              {depthLevels.map((item) => (
                <button
                  type="button"
                  key={item.value}
                  className={selectedDepth === item.value ? "active" : ""}
                  onClick={() => void updateDefaultDepth(item.value)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>
          <label className="form-field profile-goals-field">
            <span>学习目标</span>
            <textarea
              value={goalsText}
              onChange={(event) => setGoalsText(event.target.value)}
            />
          </label>
          <label className="form-field profile-agents-field">
            <span>自定义提示词（AGENTS.md）</span>
            <textarea
              value={displayedAgentsText}
              readOnly={!agentsEditable}
              onChange={(event) => setCustomAgentsText(event.target.value)}
            />
            <small className="field-help">
              {agentsEditable
                ? "当前为自定义深度，保存后问答页的自定义模式会使用这段提示词。"
                : "当前为内置深度，这里只用于预览；选择自定义后可以编辑。"}
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
