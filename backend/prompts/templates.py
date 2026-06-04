from __future__ import annotations

ANSWER_LEVELS = {
    "beginner": (
        "使用日常中文、短句和直观类比；减少术语密度，必要公式后置，"
        "并解释符号含义。"
    ),
    "undergraduate": (
        "使用本科生可理解的中文；可以使用常见术语、公式和推导步骤，"
        "但要解释关键条件。"
    ),
    "expert": (
        "使用精确、紧凑的专业中文；可以直接使用术语、公式、边界条件和严格表述。"
    ),
    "custom": "遵循用户在 Profile / AGENTS.md 中配置的语言难度、风格和输出偏好。",
}


ANSWER_LEVEL_LABELS = {
    "beginner": "入门",
    "undergraduate": "本科",
    "expert": "专家",
    "custom": "自定义",
}


BUILTIN_AGENTS_MD = {
    "beginner": """# AGENTS.md

## 难度：入门
只调整回答语言难度，不改变用户任务。
- 不要对用户输入做任务分类。
- 不要把用户问题改写成总结、出题、讲解或其他模板。
- 用户怎么问，就按原问题直接回答。
- 使用日常中文、短句和直观类比。
- 少用术语；必须使用术语时先解释。
- 必要公式可以出现，但要放慢节奏解释符号和含义。
- 只基于文档内容回答；信息不足时直接说明。
""",
    "undergraduate": """# AGENTS.md

## 难度：本科
只调整回答语言难度，不改变用户任务。
- 不要对用户输入做任务分类。
- 不要把用户问题改写成总结、出题、讲解或其他模板。
- 用户怎么问，就按原问题直接回答。
- 使用本科生可理解的中文。
- 可以使用常见术语、公式和推导步骤，但要解释关键条件。
- 如果用户指定数量、格式或范围，按用户原文要求执行。
- 只基于文档内容回答；信息不足时直接说明。
""",
    "expert": """# AGENTS.md

## 难度：专家
只调整回答语言难度，不改变用户任务。
- 不要对用户输入做任务分类。
- 不要把用户问题改写成总结、出题、讲解或其他模板。
- 用户怎么问，就按原问题直接回答。
- 使用精确、紧凑的专业中文。
- 可以直接使用术语、公式、边界条件和严格表述。
- 如果用户指定数量、格式或范围，按用户原文要求执行。
- 只基于文档内容回答；信息不足时直接说明。
""",
}


CUSTOM_AGENTS_MD_DEFAULT = """# AGENTS.md

只调整回答语言难度和表达风格，不改变用户任务。
- 不要对用户输入做任务分类。
- 不要把用户问题改写成总结、出题、讲解或其他模板。
- 用户怎么问，就按原问题直接回答。
- 如果我指定数量、格式或范围，按我的原文要求执行。
- 只基于文档内容回答；信息不足时直接说明。
"""


def normalize_answer_level(level: str) -> str:
    return level if level in ANSWER_LEVELS else "undergraduate"


def agents_md_for_level(level_key: str, custom_agents_md: str | None = None) -> str:
    if level_key == "custom":
        return (custom_agents_md or CUSTOM_AGENTS_MD_DEFAULT).strip()
    return BUILTIN_AGENTS_MD.get(level_key, BUILTIN_AGENTS_MD["undergraduate"]).strip()


def level_guidance(level_key: str, level_prompt: str) -> str:
    level_label = ANSWER_LEVEL_LABELS.get(level_key, ANSWER_LEVEL_LABELS["undergraduate"])
    return (
        f"当前回答深度：{level_label}。{level_prompt} "
        "注意：回答深度只影响语言复杂度、术语密度和解释颗粒度，"
        "不得改变用户任务、题目数量、输出格式或问题范围。"
    )


def profile_prompt_prefix(profile_prompt: str | None) -> str:
    prompt = (profile_prompt or "").strip()
    if not prompt:
        return ""
    return (
        "【Profile / Memory 个性化上下文】\n"
        "以下内容来自用户维护的 Profile 和人工确认的 Memory。"
        "它只用于调整语言难度、表达风格、输出偏好和学习连续性。"
        "不得用它对用户输入做任务分类，不得覆盖本轮用户原始问题。"
        "如果它与本轮问题冲突，以本轮问题为准；"
        "如果它与检索文档事实冲突，以检索文档事实为准。\n\n"
        f"{prompt}\n"
        "【Profile / Memory 个性化上下文结束】\n\n"
    )


def build_task_system_prompt(profile_prompt: str | None = None) -> str:
    return (
        profile_prompt_prefix(profile_prompt)
        + "你是一名中文 RAG 助教。不要对用户输入做任务分类，"
        "不要根据分类套用任何专用模板。用户怎么问，就按用户原始问题直接回答；"
        "如果用户指定数量、格式或范围，严格按原文执行。"
        "回答只能基于检索到的文档内容；信息不足时直接说明。"
        "难度选项只控制语言复杂度、术语密度和解释颗粒度，不改变用户任务。"
    )


def build_document_prompt(
    question: str,
    document_id: str,
    level_key: str,
    level_prompt: str,
    profile_prompt: str | None = None,
    for_synthesis: bool = False,
) -> str:
    del profile_prompt
    scoped_context = (
        f"当前选中文档是：{document_id}。\n"
        "检索范围：document。\n"
        "你只能基于该文档的检索结果回答。"
        "如果检索结果不足，请说明“当前文档中没有足够信息”，不要引用其他文档。"
    )
    inline_image_rules = (
        "回答必须使用 Markdown。\n"
        "如果上下文提供了图片引用 ID，且某段解释依赖对应图片，"
        "请在该段后插入真实图片 ID，例如 [[image:012345abcdef012345abcdef012345abcdef012345abcdef012345abcdef0123]]。\n"
        "每张图最多引用一次；不要在答案末尾堆全部图片；不要输出本地文件路径；"
        "不要输出字面量 image_id；不要编造 image_id；只能使用上下文明示提供的真实图片 ID。"
    )
    if for_synthesis:
        response_rule = (
            "当前是多文档回答的单文档检索阶段。不要尝试完成最终跨文档回答，"
            "也不要因为用户问题提到多个文档就判定当前文档信息不足。"
            "请提取当前文档中能支持最终回答的事实、概念、公式、例子、图表线索和可引用依据；"
            "如果当前文档确实没有相关内容，再说明信息不足。"
        )
    else:
        response_rule = (
            "按用户原始问题本身要求输出。不要新增用户没有要求的任务结构。"
        )

    return (
        f"{scoped_context}\n\n"
        "用户原始问题如下。不要分类、不要改写、不要替换成内置任务模板：\n"
        f"{question}\n\n"
        f"语言难度：{level_guidance(level_key, level_prompt)}\n\n"
        f"回答规则：{response_rule}\n\n"
        f"图片引用规则：{inline_image_rules}"
    )
