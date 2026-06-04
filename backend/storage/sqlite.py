from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from backend.config import APP_DATA_DIR, METADATA_DB_PATH

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS user_profiles (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    role TEXT,
    education_level TEXT,
    major TEXT,
    learning_goals_json TEXT,
    preferred_language TEXT,
    answer_style TEXT,
    math_level TEXT,
    coding_level TEXT,
    default_depth TEXT,
    citation_preference TEXT,
    agents_md TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS folders (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    parent_id TEXT,
    name TEXT NOT NULL,
    sort_order INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS managed_files (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    original_filename TEXT,
    display_name TEXT NOT NULL,
    folder_id TEXT,
    tags_json TEXT,
    course TEXT,
    description TEXT,
    notes TEXT,
    pinned INTEGER DEFAULT 0,
    archived INTEGER DEFAULT 0,
    status TEXT,
    source_type TEXT DEFAULT 'existing_document',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(profile_id, document_id)
);

CREATE INDEX IF NOT EXISTS idx_folders_profile_parent
    ON folders(profile_id, parent_id);

CREATE INDEX IF NOT EXISTS idx_managed_files_profile_folder
    ON managed_files(profile_id, folder_id);

CREATE INDEX IF NOT EXISTS idx_managed_files_profile_document
    ON managed_files(profile_id, document_id);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    title TEXT NOT NULL,
    document_ids_json TEXT NOT NULL DEFAULT '[]',
    level TEXT NOT NULL DEFAULT 'undergraduate',
    mode TEXT NOT NULL DEFAULT 'hybrid',
    chat_mode TEXT NOT NULL DEFAULT 'multimodal',
    file_context_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'sent',
    document_ids_json TEXT NOT NULL DEFAULT '[]',
    level TEXT,
    mode TEXT,
    chat_mode TEXT,
    sources_json TEXT NOT NULL DEFAULT '[]',
    related_images_json TEXT NOT NULL DEFAULT '[]',
    inline_image_refs_json TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_profile_updated
    ON conversations(profile_id, updated_at);

CREATE INDEX IF NOT EXISTS idx_conversation_messages_conversation_created
    ON conversation_messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS user_memories (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    key TEXT,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    source_conversation_id TEXT,
    evidence TEXT,
    status TEXT DEFAULT 'candidate',
    sensitivity TEXT DEFAULT 'normal',
    scope_type TEXT DEFAULT 'global',
    scope_id TEXT,
    evidence_message_ids_json TEXT NOT NULL DEFAULT '[]',
    last_seen_at TEXT,
    last_confirmed_at TEXT,
    expires_at TEXT,
    auto_apply INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_user_memories_profile_status
    ON user_memories(profile_id, status);

CREATE TABLE IF NOT EXISTS memory_events (
    id TEXT PRIMARY KEY,
    memory_id TEXT,
    profile_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    source_conversation_id TEXT,
    evidence TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_events_profile_created
    ON memory_events(profile_id, created_at);

CREATE TABLE IF NOT EXISTS learning_feedback (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    rating TEXT,
    difficulty TEXT,
    style_feedback TEXT,
    note TEXT,
    created_memory_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_learning_feedback_profile_created
    ON learning_feedback(profile_id, created_at);
"""


def init_metadata_db() -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(METADATA_DB_PATH) as connection:
        connection.executescript(SCHEMA_SQL)
        _ensure_column(connection, "user_profiles", "agents_md", "TEXT")
        _ensure_column(connection, "user_memories", "scope_type", "TEXT DEFAULT 'global'")
        _ensure_column(connection, "user_memories", "scope_id", "TEXT")
        _ensure_column(
            connection,
            "user_memories",
            "evidence_message_ids_json",
            "TEXT NOT NULL DEFAULT '[]'",
        )
        _ensure_column(connection, "user_memories", "last_confirmed_at", "TEXT")
        _ensure_column(connection, "user_memories", "expires_at", "TEXT")
        _ensure_column(connection, "user_memories", "auto_apply", "INTEGER DEFAULT 1")
        connection.commit()


def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_type: str,
) -> None:
    existing_columns = {
        row[1] for row in connection.execute(f"PRAGMA table_info({table_name})")
    }
    if column_name not in existing_columns:
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
        )


@contextmanager
def metadata_connection() -> Iterator[sqlite3.Connection]:
    init_metadata_db()
    connection = sqlite3.connect(METADATA_DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
