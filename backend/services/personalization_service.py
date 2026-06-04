from __future__ import annotations

from dataclasses import dataclass

from backend.prompts.templates import (
    BUILTIN_AGENTS_MD,
    agents_md_for_level,
    normalize_answer_level,
)
from backend.schemas import (
    PersonalizationPreviewResponse,
    UserMemory,
    UserProfile,
)
from backend.services.memory_service import select_relevant_memories
from backend.services.profile_service import build_prompt_context, get_profile


@dataclass(frozen=True)
class CompiledPersonalization:
    profile_id: str
    generation_prompt: str
    selected_level: str
    effective_agents_md: str
    agents_md_editable: bool
    builtin_agents_md: dict[str, str]
    used_profile_fields: list[str]
    used_memories: list[UserMemory]
    retrieval_hints: list[str]
    warnings: list[str]


def compile_personalization(
    *,
    question: str,
    document_ids: list[str],
    level: str,
    conversation_id: str | None = None,
    use_profile: bool = True,
    use_memory: bool = True,
) -> CompiledPersonalization:
    profile = get_profile()
    selected_level = normalize_answer_level(level or profile.default_depth or "undergraduate")
    effective_agents_md = agents_md_for_level(selected_level, profile.agents_md)
    used_profile_fields = _used_profile_fields(profile) if use_profile else []
    profile_prompt = build_prompt_context(profile, selected_level) if use_profile else ""

    used_memories: list[UserMemory] = []
    memory_prompt = ""
    if use_memory:
        used_memories = select_relevant_memories(
            question=question,
            document_ids=document_ids,
            conversation_id=conversation_id,
            limit=5,
        )
        from backend.prompts.memory_context import build_memory_prompt_context

        memory_prompt = build_memory_prompt_context(used_memories)

    generation_prompt = "\n\n".join(
        part.strip() for part in (profile_prompt, memory_prompt) if part.strip()
    )
    retrieval_hints = _retrieval_hints(selected_level, used_memories)
    warnings = _warnings(use_profile, use_memory, used_memories)

    return CompiledPersonalization(
        profile_id=profile.id,
        generation_prompt=generation_prompt,
        selected_level=selected_level,
        effective_agents_md=effective_agents_md,
        agents_md_editable=selected_level == "custom",
        builtin_agents_md={key: value.strip() for key, value in BUILTIN_AGENTS_MD.items()},
        used_profile_fields=used_profile_fields,
        used_memories=used_memories,
        retrieval_hints=retrieval_hints,
        warnings=warnings,
    )


def personalization_preview_response(
    *,
    question: str,
    document_ids: list[str],
    level: str,
    conversation_id: str | None = None,
    use_profile: bool = True,
    use_memory: bool = True,
) -> PersonalizationPreviewResponse:
    compiled = compile_personalization(
        question=question,
        document_ids=document_ids,
        level=level,
        conversation_id=conversation_id,
        use_profile=use_profile,
        use_memory=use_memory,
    )
    return PersonalizationPreviewResponse(
        profile_id=compiled.profile_id,
        prompt_context=compiled.generation_prompt,
        selected_level=compiled.selected_level,
        effective_agents_md=compiled.effective_agents_md,
        agents_md_editable=compiled.agents_md_editable,
        builtin_agents_md=compiled.builtin_agents_md,
        used_profile_fields=compiled.used_profile_fields,
        used_memories=compiled.used_memories,
        retrieval_hints=compiled.retrieval_hints,
        warnings=compiled.warnings,
    )


def _used_profile_fields(profile: UserProfile) -> list[str]:
    fields: list[str] = []
    for field_name in (
        "role",
        "education_level",
        "major",
        "learning_goals",
        "preferred_language",
        "answer_style",
        "math_level",
        "coding_level",
        "default_depth",
        "citation_preference",
        "agents_md",
    ):
        value = getattr(profile, field_name)
        if isinstance(value, list):
            if value:
                fields.append(field_name)
        elif value:
            fields.append(field_name)
    return fields


def _retrieval_hints(level: str, memories: list[UserMemory]) -> list[str]:
    hints: list[str] = []
    if level == "beginner":
        hints.append("优先召回定义、直观解释、基础例子和低门槛说明。")
    elif level == "expert":
        hints.append("优先召回公式、推导、限制条件、技术细节和边界条件。")
    else:
        hints.append("保持宽召回，重点覆盖关键条件、公式、例题和概念关系。")

    for memory in memories:
        text = f"{memory.memory_type} {memory.key or ''} {memory.value}".lower()
        if any(term in text for term in ("quiz", "practice", "exam", "test", "练习", "测验", "考试")):
            hints.append("用户存在练习/测验偏好时，额外召回考点、易错点和可转化为练习题的材料。")
        if "example" in text or "例" in memory.value:
            hints.append("额外召回具体例子和可操作步骤。")
        if "derivation" in text or "推导" in memory.value:
            hints.append("额外召回推导过程、公式来源和关键假设。")
    return _dedupe(hints)[:6]


def _warnings(
    use_profile: bool,
    use_memory: bool,
    memories: list[UserMemory],
) -> list[str]:
    warnings: list[str] = []
    if not use_profile:
        warnings.append("Profile is disabled for this request.")
    if use_memory and not memories:
        warnings.append("No active scoped memory matched this request.")
    if not use_memory:
        warnings.append("Memory is disabled for this request.")
    return warnings


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = item.strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result
