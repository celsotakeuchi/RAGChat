"""Operational store: interaction log, feedback and token usage (separate from the KB database)."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from ragmvp.core.models import Request, Response, Usage

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    message_id TEXT PRIMARY KEY,
    session_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    user_id TEXT,
    kb_id TEXT,
    kb_version TEXT,
    lang TEXT,
    question TEXT,
    answer TEXT,
    status TEXT,
    flags TEXT,
    retrieved TEXT,
    cited TEXT,
    model TEXT,
    prompt_version TEXT,
    latency_ms INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    raw_answer TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    user_id TEXT,
    rating INTEGER NOT NULL,
    comment TEXT
);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    day TEXT NOT NULL DEFAULT (date('now')),
    month TEXT NOT NULL DEFAULT (strftime('%Y-%m', 'now')),
    user_id TEXT NOT NULL,
    model TEXT,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_user_day ON usage (user_id, day);
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    department TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'agent',
    active INTEGER NOT NULL DEFAULT 1,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,                -- user | anonymous
    user_id TEXT NOT NULL,
    department TEXT NOT NULL DEFAULT '',
    audience TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    ended_at TEXT,
    last_seen_at TEXT,
    ip_hash TEXT,
    user_agent TEXT
);
CREATE INDEX IF NOT EXISTS sessions_ip ON sessions (ip_hash, created_at);
"""

INTERACTION_MIGRATIONS = {
    "raw_answer": "TEXT",
    "department": "TEXT",
    "audience": "TEXT",
}


class OpsStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            columns = {r["name"] for r in self._conn.execute("PRAGMA table_info(interactions)")}
            for name, kind in INTERACTION_MIGRATIONS.items():
                if name not in columns:
                    self._conn.execute(f"ALTER TABLE interactions ADD COLUMN {name} {kind}")

    def run(self, sql: str, params: tuple = ()) -> int:
        """Execute one write statement; returns the affected row count."""
        with self._lock, self._conn:
            return self._conn.execute(sql, params).rowcount

    def rows(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def log_interaction(self, request: Request, response: Response, kb_id: str, kb_version: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO interactions (message_id, session_id, user_id, kb_id, kb_version, lang, question, "
                "answer, status, flags, retrieved, cited, model, prompt_version, latency_ms, prompt_tokens, "
                "completion_tokens, raw_answer, department, audience) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    response.message_id,
                    request.session_id,
                    request.user_id,
                    kb_id,
                    kb_version,
                    request.lang,
                    request.text,
                    response.answer,
                    response.status,
                    json.dumps(response.flags),
                    json.dumps(response.retrieved),
                    json.dumps(response.cited),
                    response.model,
                    response.prompt_version,
                    response.latency_ms,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    response.raw_answer or None,
                    request.department,
                    request.audience,
                ),
            )

    def interaction_exists(self, message_id: str) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM interactions WHERE message_id=?", (message_id,)).fetchone()
        return row is not None

    def interaction_session(self, message_id: str) -> str | None:
        rows = self.rows("SELECT session_id FROM interactions WHERE message_id=?", (message_id,))
        return rows[0]["session_id"] if rows else None

    def add_feedback(self, message_id: str, user_id: str, rating: int, comment: str | None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO feedback (message_id, user_id, rating, comment) VALUES (?,?,?,?)",
                (message_id, user_id, rating, comment),
            )

    def record_usage(self, user_id: str, model: str, usage: Usage) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO usage (user_id, model, prompt_tokens, completion_tokens) VALUES (?,?,?,?)",
                (user_id, model, usage.prompt_tokens, usage.completion_tokens),
            )

    def tokens_used_today(self, user_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0) t FROM usage "
                "WHERE user_id=? AND day=date('now')",
                (user_id,),
            ).fetchone()
        return int(row["t"])

    def tokens_used_today_by_prefix(self, prefix: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0) t FROM usage "
                "WHERE user_id LIKE ? AND day=date('now')",
                (prefix + "%",),
            ).fetchone()
        return int(row["t"])

    def tokens_used_this_month(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0) t FROM usage "
                "WHERE month=strftime('%Y-%m', 'now')"
            ).fetchone()
        return int(row["t"])

    def close(self) -> None:
        self._conn.close()
