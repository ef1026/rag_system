from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.constants import DocumentState

class DocumentSummary(BaseModel):
    id: str
    name: str
    size: int
    status: DocumentState
    readiness_warnings: list[str] = Field(default_factory=list)


class SourceItem(BaseModel):
    id: str
    type: str = "text"
    page: int | None = None
    page_end: int | None = None
    text: str
    document_id: str | None = None
    document_name: str | None = None
    chunk_id: str | None = None
    content_index: int | None = None
    rank: int | None = None
    score: float | None = None
    score_type: str | None = None
    match_score: float | None = None
    match_method: str | None = None
    citation_mode: str = "retrieval_context"


class UploadResponse(BaseModel):
    document: DocumentSummary


class ProcessResponse(BaseModel):
    document: DocumentSummary
    message: str
    sources: list[SourceItem]
    readiness_warnings: list[str] = Field(default_factory=list)
    text_indexed: bool = False
    multimodal_enabled: bool = False
    multimodal_status: str = "disabled"
    image_status: str = "disabled"
    table_status: str = "disabled"
    equation_status: str = "disabled"
    formula_status: str = "disabled"
    skipped_multimodal_items_count: int = 0
    skipped_by_reason: dict[str, int] = Field(default_factory=dict)
    multimodal_warnings_count: int = 0
    warnings_summary: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    document_id: str | None = None
    document_ids: list[str] | None = None
    level: str = "undergraduate"
    mode: str = "hybrid"
    vlm_enhanced: bool | Literal["auto"] = "auto"
    conversation_id: str | None = None
    client_user_message_id: str | None = None
    use_profile: bool = True
    use_memory: bool = False


class ChatDocumentUsed(BaseModel):
    document_id: str
    name: str
    status: DocumentState


class ChatPartialFailure(BaseModel):
    document_id: str
    name: str | None = None
    error: str


class ImageAssetPublic(BaseModel):
    image_id: str
    document_id: str
    document_name: str
    url: str
    filename: str | None = Field(default=None, exclude=True)
    caption: str | None = None
    page: int | None = None
    bbox: Any | None = None
    source_type: str | None = None
    relevance_reason: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    related_images: list[ImageAssetPublic] = Field(default_factory=list)
    inline_image_refs: list[str] = Field(default_factory=list)
    documents_used: list[ChatDocumentUsed] = Field(default_factory=list)
    partial_failures: list[ChatPartialFailure] = Field(default_factory=list)
    conversation_id: str | None = None
    user_message_id: str | None = None
    assistant_message_id: str | None = None


class CacheResponse(BaseModel):
    ok: bool


class WarmupResponse(BaseModel):
    ok: bool
    document: DocumentSummary
    storage_dir: str
    warmup_seconds: float


class HealthResponse(BaseModel):
    ok: bool
    has_api_key: bool
    mineru_backend: str
    mineru_device: str


class RAGRetrievalStatus(BaseModel):
    default_mode: str
    rerank_requested: bool
    rerank_enabled: bool
    rerank_model: str | None
    rerank_provider: str | None
    rerank_model_loaded: bool
    rerank_last_error: str | None = None
    reason: str | None = None
    rerank_top_n: int | None = None


class RAGMultimodalStatus(BaseModel):
    enabled: bool
    image_processing: bool
    table_processing: bool
    equation_processing: bool
    formula_processing: bool
    vlm_model: str | None = None


class RAGEmbeddingStatus(BaseModel):
    model: str


class RAGStatusResponse(BaseModel):
    parser: str
    parser_output_dir: str
    retrieval: RAGRetrievalStatus
    multimodal: RAGMultimodalStatus
    embedding: RAGEmbeddingStatus
    embedding_model: str
    vector_store: str
    graph_store: str
    retrieval_mode: str
    rerank_requested: bool
    rerank_enabled: bool
    rerank_model: str | None
    rerank_binding: str | None
    multimodal_enabled: bool
    citation_status: str
    sources_status: str


class UserProfile(BaseModel):
    id: str
    display_name: str
    role: str | None = None
    education_level: str | None = None
    major: str | None = None
    learning_goals: list[str] = Field(default_factory=list)
    preferred_language: str | None = None
    answer_style: str | None = None
    math_level: str | None = None
    coding_level: str | None = None
    default_depth: str | None = None
    citation_preference: str | None = None
    agents_md: str | None = None
    created_at: str
    updated_at: str


class UserProfilePut(BaseModel):
    display_name: str = Field(min_length=1)
    role: str | None = None
    education_level: str | None = None
    major: str | None = None
    learning_goals: list[str] = Field(default_factory=list)
    preferred_language: str | None = None
    answer_style: str | None = None
    math_level: str | None = None
    coding_level: str | None = None
    default_depth: str | None = None
    citation_preference: str | None = None
    agents_md: str | None = None


class UserProfilePatch(BaseModel):
    display_name: str | None = Field(default=None, min_length=1)
    role: str | None = None
    education_level: str | None = None
    major: str | None = None
    learning_goals: list[str] | None = None
    preferred_language: str | None = None
    answer_style: str | None = None
    math_level: str | None = None
    coding_level: str | None = None
    default_depth: str | None = None
    citation_preference: str | None = None
    agents_md: str | None = None


class ProfilePromptContextResponse(BaseModel):
    profile_id: str
    prompt_context: str
    selected_level: str
    effective_agents_md: str
    agents_md_editable: bool
    builtin_agents_md: dict[str, str] = Field(default_factory=dict)
    used_profile_fields: list[str] = Field(default_factory=list)
    used_memories: list["UserMemory"] = Field(default_factory=list)
    retrieval_hints: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PersonalizationPreviewRequest(BaseModel):
    question: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    level: str = "undergraduate"
    conversation_id: str | None = None
    use_profile: bool = True
    use_memory: bool = True


class PersonalizationPreviewResponse(BaseModel):
    profile_id: str
    prompt_context: str
    selected_level: str
    effective_agents_md: str
    agents_md_editable: bool
    builtin_agents_md: dict[str, str] = Field(default_factory=dict)
    used_profile_fields: list[str] = Field(default_factory=list)
    used_memories: list["UserMemory"] = Field(default_factory=list)
    retrieval_hints: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Folder(BaseModel):
    id: str
    profile_id: str
    parent_id: str | None = None
    name: str
    sort_order: int = 0
    created_at: str
    updated_at: str


class FolderCreate(BaseModel):
    name: str = Field(min_length=1)
    parent_id: str | None = None
    sort_order: int = 0


class FolderPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1)
    parent_id: str | None = None
    sort_order: int | None = None


class ManagedFile(BaseModel):
    id: str
    profile_id: str
    document_id: str
    original_filename: str | None = None
    display_name: str
    folder_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    course: str | None = None
    description: str | None = None
    notes: str | None = None
    pinned: bool = False
    archived: bool = False
    status: str | None = None
    source_type: str = "existing_document"
    created_at: str
    updated_at: str


class ManagedFileRegister(BaseModel):
    document_id: str = Field(min_length=1)
    original_filename: str | None = None
    display_name: str | None = None
    folder_id: str | None = None
    tags: list[str] | None = None
    course: str | None = None
    description: str | None = None
    notes: str | None = None


class ManagedFilePatch(BaseModel):
    display_name: str | None = Field(default=None, min_length=1)
    folder_id: str | None = None
    tags: list[str] | None = None
    course: str | None = None
    description: str | None = None
    notes: str | None = None
    pinned: bool | None = None
    archived: bool | None = None


class RegisterManagedFileResponse(BaseModel):
    file: ManagedFile
    created: bool


class SyncExistingDocumentsResponse(BaseModel):
    created_count: int
    existing_count: int
    files: list[ManagedFile]


class TagSummary(BaseModel):
    name: str
    count: int


class Conversation(BaseModel):
    id: str
    profile_id: str
    title: str
    document_ids: list[str] = Field(default_factory=list)
    level: str = "undergraduate"
    mode: str = "hybrid"
    chat_mode: str = "multimodal"
    file_context: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class ConversationCreate(BaseModel):
    id: str | None = None
    title: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    level: str = "undergraduate"
    mode: str = "hybrid"
    chat_mode: str = "multimodal"
    file_context: dict[str, Any] = Field(default_factory=dict)


class ConversationPatch(BaseModel):
    title: str | None = None
    document_ids: list[str] | None = None
    level: str | None = None
    mode: str | None = None
    chat_mode: str | None = None
    file_context: dict[str, Any] | None = None


class ConversationMessage(BaseModel):
    id: str
    conversation_id: str
    role: Literal["user", "assistant"]
    content: str
    status: str = "sent"
    document_ids: list[str] = Field(default_factory=list)
    level: str | None = None
    mode: str | None = None
    chat_mode: str | None = None
    sources: list[SourceItem] = Field(default_factory=list)
    related_images: list[ImageAssetPublic] = Field(default_factory=list)
    inline_image_refs: list[str] = Field(default_factory=list)
    error: str | None = None
    created_at: str


class ConversationMessageImport(BaseModel):
    id: str | None = None
    role: Literal["user", "assistant"]
    content: str
    status: str = "sent"
    document_ids: list[str] = Field(default_factory=list)
    level: str | None = None
    mode: str | None = None
    chat_mode: str | None = None
    sources: list[SourceItem] = Field(default_factory=list)
    related_images: list[ImageAssetPublic] = Field(default_factory=list)
    inline_image_refs: list[str] = Field(default_factory=list)
    error: str | None = None
    created_at: str | None = None


class ConversationImportItem(BaseModel):
    id: str
    title: str
    document_ids: list[str] = Field(default_factory=list)
    level: str = "undergraduate"
    mode: str = "hybrid"
    chat_mode: str = "multimodal"
    file_context: dict[str, Any] = Field(default_factory=dict)
    messages: list[ConversationMessageImport] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None


class ConversationImportRequest(BaseModel):
    conversations: list[ConversationImportItem] = Field(default_factory=list)


class ConversationImportResponse(BaseModel):
    imported_count: int
    message_count: int
    conversations: list[Conversation] = Field(default_factory=list)


class UserMemory(BaseModel):
    id: str
    profile_id: str
    memory_type: str
    key: str | None = None
    value: str
    confidence: float = 0.5
    source_conversation_id: str | None = None
    evidence: str | None = None
    status: str = "candidate"
    sensitivity: str = "normal"
    scope_type: Literal["global", "course", "document", "conversation"] = "global"
    scope_id: str | None = None
    evidence_message_ids: list[str] = Field(default_factory=list)
    last_seen_at: str | None = None
    last_confirmed_at: str | None = None
    expires_at: str | None = None
    auto_apply: bool = True
    created_at: str
    updated_at: str


class UserMemoryPatch(BaseModel):
    memory_type: str | None = None
    key: str | None = None
    value: str | None = None
    confidence: float | None = None
    evidence: str | None = None
    status: str | None = None
    sensitivity: str | None = None
    scope_type: Literal["global", "course", "document", "conversation"] | None = None
    scope_id: str | None = None
    evidence_message_ids: list[str] | None = None
    last_confirmed_at: str | None = None
    expires_at: str | None = None
    auto_apply: bool | None = None


class MemoryCandidateSource(BaseModel):
    id: str
    source_type: Literal["conversation", "wrong_question"]
    title: str
    preview: str
    created_at: str
    updated_at: str | None = None
    message_count: int = 0
    document_ids: list[str] = Field(default_factory=list)
    reviewed_at: str | None = None


class MemoryExtractRequest(BaseModel):
    conversation_id: str | None = None
    conversation_ids: list[str] = Field(default_factory=list)
    wrong_question_ids: list[str] = Field(default_factory=list)
    limit: int = 50


class MemoryExtractResponse(BaseModel):
    created_count: int
    memories: list[UserMemory] = Field(default_factory=list)


class ProfileFeedbackRequest(BaseModel):
    conversation_id: str = Field(min_length=1)
    message_id: str = Field(min_length=1)
    rating: Literal["helpful", "not_helpful"] | None = None
    difficulty: Literal["too_easy", "too_hard"] | None = None
    style_feedback: Literal[
        "needs_examples",
        "needs_derivation",
        "more_concise",
    ] | None = None
    note: str | None = None


class ProfileFeedbackResponse(BaseModel):
    ok: bool = True
    created_memory_candidates: list[UserMemory] = Field(default_factory=list)


class QuizChoice(BaseModel):
    id: str
    text: str


class QuizQuestion(BaseModel):
    id: str
    prompt: str
    choices: list[QuizChoice] = Field(default_factory=list)
    correct_choice_id: str
    explanation: str
    source_message_ids: list[str] = Field(default_factory=list)
    related_images: list[ImageAssetPublic] = Field(default_factory=list)


class QuizSession(BaseModel):
    id: str
    profile_id: str
    conversation_id: str
    title: str
    questions: list[QuizQuestion] = Field(default_factory=list)
    created_at: str


class QuizGenerateRequest(BaseModel):
    conversation_id: str | None = Field(default=None, min_length=1)
    document_ids: list[str] = Field(default_factory=list)
    count: int = 3


class QuizSubmitAnswer(BaseModel):
    question_id: str = Field(min_length=1)
    selected_choice_id: str = Field(min_length=1)


class QuizSubmitRequest(BaseModel):
    answers: list[QuizSubmitAnswer] = Field(default_factory=list)


class QuizQuestionResult(BaseModel):
    question: QuizQuestion
    selected_choice_id: str | None = None
    is_correct: bool
    correct_choice_id: str
    explanation: str


class QuizSubmitResponse(BaseModel):
    session_id: str
    correct_count: int
    total_count: int
    results: list[QuizQuestionResult] = Field(default_factory=list)


class WrongQuestion(BaseModel):
    id: str
    profile_id: str
    quiz_session_id: str
    conversation_id: str
    question_id: str
    prompt: str
    choices: list[QuizChoice] = Field(default_factory=list)
    selected_choice_id: str
    correct_choice_id: str
    explanation: str
    source_message_ids: list[str] = Field(default_factory=list)
    related_images: list[ImageAssetPublic] = Field(default_factory=list)
    created_at: str
    reviewed_at: str | None = None


for _model in (ProfilePromptContextResponse, PersonalizationPreviewResponse):
    if hasattr(_model, "model_rebuild"):
        _model.model_rebuild()
    elif hasattr(_model, "update_forward_refs"):
        _model.update_forward_refs()
