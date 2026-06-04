import { request } from "@/lib/api";
import type { AnswerLevel } from "@/types/chat";
import type {
  PersonalizationPreviewRequest,
  PersonalizationPreviewResponse,
  ProfileFeedbackRequest,
  ProfileFeedbackResponse,
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
  personalizationPreview: (payload: PersonalizationPreviewRequest) =>
    request<PersonalizationPreviewResponse>("/api/profile/personalization-preview", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  feedback: (payload: ProfileFeedbackRequest) =>
    request<ProfileFeedbackResponse>("/api/profile/feedback", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
