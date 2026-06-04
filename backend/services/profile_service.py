from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from backend.schemas import (
    ProfilePromptContextResponse,
    UserProfile,
    UserProfilePatch,
    UserProfilePut,
)
from backend.prompts.templates import (
    ANSWER_LEVEL_LABELS,
    BUILTIN_AGENTS_MD,
    CUSTOM_AGENTS_MD_DEFAULT,
    agents_md_for_level,
    normalize_answer_level,
)
from backend.storage.sqlite import metadata_connection

DEFAULT_PROFILE_ID = "default"
DEFAULT_AGENTS_MD = CUSTOM_AGENTS_MD_DEFAULT
AGENTS_MD_MAX_CHARS = 4000
PROFILE_TEXT_MAX_CHARS = 240
CITATION_PREFERENCE_MAX_CHARS = 600

PROFILE_COLUMNS = (
    "id",
    "display_name",
    "role",
    "education_level",
    "major",
    "learning_goals_json",
    "preferred_language",
    "answer_style",
    "math_level",
    "coding_level",
    "default_depth",
    "citation_preference",
    "agents_md",
    "created_at",
    "updated_at",
)


def get_profile() -> UserProfile:
    profile = _read_profile(DEFAULT_PROFILE_ID)
    if profile:
        return profile
    return _create_default_profile()


def put_profile(payload: UserProfilePut) -> UserProfile:
    existing = _read_profile(DEFAULT_PROFILE_ID)
    created_at = existing.created_at if existing else _utc_now()
    data = _normalize_profile_data(_model_dump(payload))
    data["id"] = DEFAULT_PROFILE_ID
    data["created_at"] = created_at
    data["updated_at"] = _utc_now()
    _write_profile(data)
    return _read_required_profile(DEFAULT_PROFILE_ID)


def patch_profile(payload: UserProfilePatch) -> UserProfile:
    current = get_profile()
    updates = _normalize_profile_data(_model_dump(payload, exclude_unset=True))
    data = _model_dump(current)
    data.update(updates)
    data["updated_at"] = _utc_now()
    _write_profile(data)
    return _read_required_profile(DEFAULT_PROFILE_ID)


def get_prompt_context(level: str | None = None) -> ProfilePromptContextResponse:
    profile = get_profile()
    selected_level = normalize_answer_level(level or profile.default_depth or "undergraduate")
    effective_agents_md = agents_md_for_level(selected_level, profile.agents_md)
    return ProfilePromptContextResponse(
        profile_id=profile.id,
        prompt_context=build_prompt_context(profile, selected_level),
        selected_level=selected_level,
        effective_agents_md=effective_agents_md,
        agents_md_editable=selected_level == "custom",
        builtin_agents_md={key: value.strip() for key, value in BUILTIN_AGENTS_MD.items()},
    )


def build_prompt_context(profile: UserProfile, level: str | None = None) -> str:
    selected_level = normalize_answer_level(level or profile.default_depth or "undergraduate")
    parts: list[str] = []
    identity = _join_sentence_parts(
        [
            profile.role,
            profile.education_level,
            f"majoring in {profile.major}" if profile.major else None,
        ]
    )
    if identity:
        parts.append(f"The user is {identity}.")

    if profile.learning_goals:
        parts.append(f"Their learning goals are: {', '.join(profile.learning_goals)}.")
    if profile.preferred_language:
        parts.append(f"Prefer explanations in {profile.preferred_language}.")
    if profile.answer_style:
        parts.append(f"Use a {profile.answer_style} answer style.")
    if selected_level:
        parts.append(f"Use {selected_level}-level depth for this response.")
    if profile.math_level:
        parts.append(f"Assume {profile.math_level} math background.")
    if profile.coding_level:
        parts.append(f"Assume {profile.coding_level} coding background.")
    if profile.citation_preference:
        parts.append(f"Citation preference: {profile.citation_preference}.")

    profile_context = (
        " ".join(parts)
        if parts
        else (
            "The user has not configured a detailed profile yet. Use clear, "
            "step-by-step explanations and cite sources when available."
        )
    )
    agents_md = agents_md_for_level(selected_level, profile.agents_md)
    level_label = ANSWER_LEVEL_LABELS.get(selected_level, selected_level)
    return (
        f"# Profile context\n\n{profile_context}\n\n"
        f"# Effective AGENTS.md ({level_label})\n\n{agents_md}"
    )


def _create_default_profile() -> UserProfile:
    now = _utc_now()
    data: dict[str, Any] = {
        "id": DEFAULT_PROFILE_ID,
        "display_name": "Default Learner",
        "role": "student",
        "education_level": "undergraduate",
        "major": None,
        "learning_goals": [],
        "preferred_language": "Chinese",
        "answer_style": "step-by-step",
        "math_level": "undergraduate",
        "coding_level": None,
        "default_depth": "undergraduate",
        "citation_preference": "cite when available",
        "agents_md": DEFAULT_AGENTS_MD,
        "created_at": now,
        "updated_at": now,
    }
    _write_profile(data)
    return _read_required_profile(DEFAULT_PROFILE_ID)


def _read_required_profile(profile_id: str) -> UserProfile:
    profile = _read_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=500, detail="Profile could not be loaded.")
    return profile


def _read_profile(profile_id: str) -> UserProfile | None:
    with metadata_connection() as connection:
        row = connection.execute(
            "SELECT * FROM user_profiles WHERE id = ?",
            (profile_id,),
        ).fetchone()
    if row is None:
        return None
    return _profile_from_row(dict(row))


def _write_profile(data: dict[str, Any]) -> None:
    goals = _normalize_goals(data.get("learning_goals"))
    values = {
        "id": data["id"],
        "display_name": _required_text(data.get("display_name"), "display_name"),
        "role": _optional_text(data.get("role"), "role"),
        "education_level": _optional_text(data.get("education_level"), "education_level"),
        "major": _optional_text(data.get("major"), "major"),
        "learning_goals_json": json.dumps(goals, ensure_ascii=False),
        "preferred_language": _optional_text(data.get("preferred_language"), "preferred_language"),
        "answer_style": _optional_text(data.get("answer_style"), "answer_style"),
        "math_level": _optional_text(data.get("math_level"), "math_level"),
        "coding_level": _optional_text(data.get("coding_level"), "coding_level"),
        "default_depth": normalize_answer_level(
            _optional_text(data.get("default_depth"), "default_depth") or "undergraduate"
        ),
        "citation_preference": _optional_text(
            data.get("citation_preference"),
            "citation_preference",
            CITATION_PREFERENCE_MAX_CHARS,
        ),
        "agents_md": _optional_text(
            data.get("agents_md"),
            "agents_md",
            AGENTS_MD_MAX_CHARS,
        ),
        "created_at": data["created_at"],
        "updated_at": data["updated_at"],
    }
    placeholders = ", ".join("?" for _name in PROFILE_COLUMNS)
    assignments = ", ".join(f"{name} = excluded.{name}" for name in PROFILE_COLUMNS[1:])
    with metadata_connection() as connection:
        connection.execute(
            f"""
            INSERT INTO user_profiles ({", ".join(PROFILE_COLUMNS)})
            VALUES ({placeholders})
            ON CONFLICT(id) DO UPDATE SET {assignments}
            """,
            tuple(values[name] for name in PROFILE_COLUMNS),
        )


def _profile_from_row(row: dict[str, Any]) -> UserProfile:
    return UserProfile(
        id=str(row["id"]),
        display_name=str(row["display_name"]),
        role=row.get("role"),
        education_level=row.get("education_level"),
        major=row.get("major"),
        learning_goals=_load_json_list(row.get("learning_goals_json")),
        preferred_language=row.get("preferred_language"),
        answer_style=row.get("answer_style"),
        math_level=row.get("math_level"),
        coding_level=row.get("coding_level"),
        default_depth=row.get("default_depth"),
        citation_preference=row.get("citation_preference"),
        agents_md=row.get("agents_md") or DEFAULT_AGENTS_MD,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _normalize_profile_data(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    if "learning_goals" in normalized:
        normalized["learning_goals"] = _normalize_goals(normalized["learning_goals"])
    return normalized


def _normalize_goals(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    goals: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if len(text) > 120:
            raise HTTPException(status_code=400, detail="learning_goals item is too long.")
        key = text.lower()
        if text and key not in seen:
            goals.append(text)
            seen.add(key)
        if len(goals) >= 20:
            break
    return goals


def _load_json_list(value: Any) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return _normalize_goals(parsed)


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail=f"{field_name} is required.")
    return text


def _optional_text(
    value: Any,
    field_name: str = "value",
    max_chars: int = PROFILE_TEXT_MAX_CHARS,
) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) > max_chars:
        raise HTTPException(status_code=400, detail=f"{field_name} is too long.")
    return text or None


def _join_sentence_parts(parts: list[str | None]) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _model_dump(model: Any, **kwargs: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)
    return model.dict(**kwargs)
