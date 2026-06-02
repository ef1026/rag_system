from __future__ import annotations

from backend.schemas import UserMemory


def build_memory_prompt_context(memories: list[UserMemory]) -> str:
    active_memories = [
        memory
        for memory in memories
        if memory.status == "active" and memory.value.strip()
    ][:5]
    if not active_memories:
        return ""

    lines = ["[Relevant Learning Memory]"]
    for memory in active_memories:
        key = f"{memory.memory_type}"
        if memory.key:
            key += f":{memory.key}"
        lines.append(f"- {key}: {memory.value.strip()}")

    lines.extend(
        [
            "",
            "[Memory Rules]",
            "- Use memory only to adjust language, depth, format, and learning continuity.",
            "- Do not use memory as factual evidence.",
            "- Ground factual claims only in retrieved document content.",
            "- If memory conflicts with this turn's question, follow this turn's question.",
            "- If memory conflicts with retrieved documents, follow retrieved documents.",
        ]
    )
    return "\n".join(lines)
