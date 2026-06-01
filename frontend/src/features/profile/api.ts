import { request } from "@/lib/api";
import type { AnswerLevel } from "@/types/chat";
import type {
  ProfilePromptContextResponse,
  UserProfile,
  UserProfileInput,
} from "./types";

export const profileApi = {
  get: () => request<UserProfile>("/api/profile"),
  save: (payload: UserProfileInput) =>
    request<UserProfile>("/api/profile", {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
  patch: (payload: Partial<UserProfileInput>) =>
    request<UserProfile>("/api/profile", {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  promptContext: (level?: AnswerLevel | string | null) =>
    request<ProfilePromptContextResponse>(
      `/api/profile/prompt-context${
        level ? `?level=${encodeURIComponent(level)}` : ""
      }`,
    ),
};
