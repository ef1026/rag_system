from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    smoke_dir = REPO_ROOT / ".cache" / "smoke-memory-e2e"
    shutil.rmtree(smoke_dir, ignore_errors=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)

    import backend.storage.sqlite as sqlite_storage

    sqlite_storage.APP_DATA_DIR = smoke_dir
    sqlite_storage.METADATA_DB_PATH = smoke_dir / "metadata.sqlite3"

    from backend.schemas import (
        ConversationCreate,
        ConversationImportItem,
        ConversationImportRequest,
        ConversationMessageImport,
        MemoryExtractRequest,
        ProfileFeedbackRequest,
        UserMemoryPatch,
    )
    from backend.services.conversation_service import (
        append_message,
        clear_messages,
        create_conversation,
        delete_conversation,
        import_conversations,
        list_messages,
    )
    from backend.services.memory_service import (
        accept_memory,
        delete_memory,
        dismiss_memory,
        extract_memories,
        list_memory_candidate_sources,
        list_memories,
        patch_memory,
        select_relevant_memories,
    )
    from backend.services.feedback_service import record_profile_feedback
    from backend.storage.sqlite import metadata_connection

    try:
        conversation = create_conversation(
            ConversationCreate(
                id="conv_e2e",
                title="Smoke conversation",
                document_ids=["doc_smoke"],
                level="undergraduate",
                mode="hybrid",
                chat_mode="multimodal",
            )
        )

        first_user = append_message(
            conversation_id=conversation.id,
            message_id="client_user_1",
            role="user",
            content="Please make a quiz with answers for this document.",
            document_ids=["doc_smoke"],
            level="undergraduate",
            mode="hybrid",
            chat_mode="multimodal",
        )
        append_message(
            conversation_id=conversation.id,
            message_id="client_user_2",
            role="user",
            content="Give me another practice quiz and include explanations.",
            document_ids=["doc_smoke"],
            level="undergraduate",
            mode="hybrid",
            chat_mode="multimodal",
        )
        assistant = append_message(
            conversation_id=conversation.id,
            role="assistant",
            content="Here is a practice set with answer explanations.",
            document_ids=["doc_smoke"],
            level="undergraduate",
            mode="hybrid",
            chat_mode="multimodal",
        )

        with metadata_connection() as connection:
            connection.execute(
                """
                INSERT INTO wrong_questions (
                    id, profile_id, quiz_session_id, conversation_id, question_id,
                    prompt, choices_json, selected_choice_id, correct_choice_id,
                    explanation, source_message_ids_json, related_images_json,
                    created_at, reviewed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    "wrong_e2e",
                    conversation.profile_id,
                    "quiz_e2e",
                    conversation.id,
                    "qq_e2e",
                    "Which condition makes the sampled signal alias?",
                    json.dumps(
                        [
                            {"id": "A", "text": "Sampling below Nyquist"},
                            {"id": "B", "text": "Sampling exactly at zero"},
                        ],
                        ensure_ascii=False,
                    ),
                    "B",
                    "A",
                    "Aliasing appears when the sampling rate is below Nyquist.",
                    json.dumps([assistant.id], ensure_ascii=False),
                    "[]",
                    assistant.created_at,
                ),
            )

        candidate_sources = list_memory_candidate_sources()
        assert any(
            source.source_type == "conversation" and source.id == conversation.id
            for source in candidate_sources
        )
        assert any(
            source.source_type == "wrong_question" and source.id == "wrong_e2e"
            for source in candidate_sources
        )

        selected_extract = extract_memories(
            MemoryExtractRequest(
                conversation_ids=[conversation.id],
                wrong_question_ids=["wrong_e2e"],
                limit=20,
            )
        )
        assert any(
            memory.memory_type == "chat_record"
            for memory in selected_extract.memories
        )
        assert any(
            memory.memory_type == "wrong_question"
            for memory in selected_extract.memories
        )

        assert first_user.id == "client_user_1"
        before_import_count = len(list_messages(conversation.id))

        import_conversations(
            ConversationImportRequest(
                conversations=[
                    ConversationImportItem(
                        id=conversation.id,
                        title=conversation.title,
                        document_ids=conversation.document_ids,
                        level=conversation.level,
                        mode=conversation.mode,
                        chat_mode=conversation.chat_mode,
                        messages=[
                            ConversationMessageImport(
                                id=first_user.id,
                                role=first_user.role,
                                content=first_user.content,
                                status=first_user.status,
                                document_ids=first_user.document_ids,
                                level=first_user.level,
                                mode=first_user.mode,
                                chat_mode=first_user.chat_mode,
                                created_at=first_user.created_at,
                            ),
                            ConversationMessageImport(
                                id=assistant.id,
                                role=assistant.role,
                                content=assistant.content,
                                status=assistant.status,
                                document_ids=assistant.document_ids,
                                level=assistant.level,
                                mode=assistant.mode,
                                chat_mode=assistant.chat_mode,
                                created_at=assistant.created_at,
                            ),
                        ],
                    )
                ]
            )
        )
        after_import_count = len(list_messages(conversation.id))
        assert after_import_count == before_import_count

        extract_result = extract_memories(
            MemoryExtractRequest(conversation_id=conversation.id, limit=5)
        )
        assert extract_result.created_count >= 1
        assert any(memory.scope_type in {"document", "conversation"} for memory in extract_result.memories)

        feedback = record_profile_feedback(
            ProfileFeedbackRequest(
                conversation_id=conversation.id,
                message_id=assistant.id,
                difficulty="too_hard",
                style_feedback="needs_examples",
            )
        )
        assert feedback.created_memory_candidates
        candidate = feedback.created_memory_candidates[0]
        assert candidate.scope_type == "document"
        assert candidate.scope_id == "doc_smoke"
        assert candidate.evidence_message_ids

        active = accept_memory(candidate.id)
        assert active.status == "active"
        scoped_matches = select_relevant_memories(
            question="Please add examples.",
            document_ids=["doc_smoke"],
        )
        assert any(memory.id == active.id for memory in scoped_matches)
        scoped_misses = select_relevant_memories(
            question="Please add examples.",
            document_ids=["other_doc"],
        )
        assert all(memory.id != active.id for memory in scoped_misses)

        updated = patch_memory(
            active.id,
            UserMemoryPatch(
                confidence=0.99,
                scope_type="document",
                scope_id="doc_smoke",
            ),
        )
        assert updated.confidence == 0.99
        assert updated.scope_type == "document"

        dismissed = dismiss_memory(updated.id)
        assert dismissed.status == "dismissed"

        delete_memory(dismissed.id)
        for memory in list_memories():
            delete_memory(memory.id)
        assert not list_memories()

        clear_messages(conversation.id)
        assert not list_messages(conversation.id)
        delete_conversation(conversation.id)
    finally:
        shutil.rmtree(smoke_dir, ignore_errors=True)

    print("smoke_memory_e2e ok")


if __name__ == "__main__":
    main()
