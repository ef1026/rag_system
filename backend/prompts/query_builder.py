from __future__ import annotations


def prompt_template_name(document_count: int) -> str:
    scope = "multi_document" if document_count > 1 else "single_document"
    return f"direct_{scope}"


def build_chat_query(question: str, document_id: str, for_synthesis: bool = False) -> str:
    broad_recall_terms = (
        "宽召回线索：文档主题、章节标题、核心概念、定义、公式、定理、性质、"
        "推导步骤、例题、图表、表格、符号含义、应用条件、限制条件、常见误区、"
        "可用于回答用户问题的事实依据。"
    )
    if for_synthesis:
        return (
            f"用户原始问题：{question}\n"
            f"当前文档：{document_id}\n"
            "检索目标：尽可能宽泛地找出当前文档中能支持回答用户原始问题的相关内容。\n"
            f"{broad_recall_terms}"
        )
    return (
        f"用户原始问题：{question}\n"
        f"当前文档：{document_id}\n"
        "检索目标：先宽泛召回相关材料，再基于用户原始问题组织回答。\n"
        f"{broad_recall_terms}"
    )
