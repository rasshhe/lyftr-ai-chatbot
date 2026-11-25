import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class SQLiteStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.init_schema()

    def init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages (session_id, id);
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_prompts (
                    session_id TEXT PRIMARY KEY,
                    system_prompt TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def ping(self) -> bool:
        cur = self._conn.cursor()
        cur.execute("SELECT 1;")
        row = cur.fetchone()
        return row is not None and row[0] == 1

    def set_session_prompt(self, session_id: str, system_prompt: str) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO session_prompts (session_id, system_prompt, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    system_prompt = excluded.system_prompt,
                    updated_at = excluded.updated_at;
                """,
                (session_id, system_prompt, now_iso),
            )

    def get_session_prompt(self, session_id: str, fallback_prompt: str) -> str:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT system_prompt FROM session_prompts WHERE session_id = ?;",
            (session_id,),
        )
        row = cur.fetchone()
        if row and row["system_prompt"]:
            return row["system_prompt"]
        return fallback_prompt

    def get_conversation_messages(self, session_id: str) -> List[Dict[str, str]]:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC;
            """,
            (session_id,),
        )
        rows = cur.fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def append_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_reply: str,
        assistant_metadata: Dict[str, Any],
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(assistant_metadata)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata, created_at)
                VALUES (?, 'user', ?, NULL, ?);
                """,
                (session_id, user_message, now_iso),
            )
            self._conn.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata, created_at)
                VALUES (?, 'assistant', ?, ?, ?);
                """,
                (session_id, assistant_reply, meta_json, now_iso),
            )

    def get_session_history(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT role, content, metadata
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC;
            """,
            (session_id,),
        )
        rows = cur.fetchall()
        if not rows:
            return None

        history: List[Dict[str, Any]] = []
        for r in rows:
            item: Dict[str, Any] = {
                "role": r["role"],
                "content": r["content"],
            }
            if r["role"] == "assistant" and r["metadata"]:
                try:
                    parsed_meta = json.loads(r["metadata"])
                    item["metadata"] = {
                        "resolution_source": parsed_meta.get("resolution_source"),
                        "model": parsed_meta.get("model"),
                        "latency_ms": parsed_meta.get("latency_ms"),
                        "input_tokens": parsed_meta.get("input_tokens", 0),
                        "output_tokens": parsed_meta.get("output_tokens", 0),
                        "key_id": parsed_meta.get("key_id"),
                        "fallback_attempts": parsed_meta.get("fallback_attempts", 0),
                    }
                except json.JSONDecodeError:
                    item["metadata"] = {}
            history.append(item)
        return history

    def close(self) -> None:
        self._conn.close()
