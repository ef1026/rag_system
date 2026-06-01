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
"""


def init_metadata_db() -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(METADATA_DB_PATH) as connection:
        connection.executescript(SCHEMA_SQL)
        _ensure_column(connection, "user_profiles", "agents_md", "TEXT")
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
