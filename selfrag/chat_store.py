"""
chat_store.py
-------------
Lightweight SQLite persistence for Self-RAG chat threads.

This is intentionally separate from LangGraph's own checkpointer
(`MemorySaver` / `SqliteSaver`). LangGraph's checkpointer persists
*graph execution state* so a run can be resumed/inspected — it does
not give us a convenient way to redisplay "here's the whole
conversation, with the Self-RAG debug details for each answer, and
here's which PDFs/YouTube videos were loaded" in a chat UI. This
module owns exactly that: threads, messages (with attached metadata),
and per-thread sources.

Everything here is plain `sqlite3` — no ORM — so it has no extra
dependencies beyond the standard library, and survives Streamlit
reruns/hot-reloads and full process restarts because it's backed by a
file on disk (`CHAT_DB_PATH`, default `self_rag_chat_history.sqlite3`).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from . import config

DB_PATH = config.CHAT_DB_PATH

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id   TEXT PRIMARY KEY,
    title       TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    metadata    TEXT,
    created_at  REAL NOT NULL,
    FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
);

CREATE TABLE IF NOT EXISTS thread_sources (
    thread_id     TEXT NOT NULL,
    source_name   TEXT NOT NULL,
    documents     TEXT NOT NULL,
    added_at      REAL NOT NULL,
    PRIMARY KEY (thread_id, source_name)
);

CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);
"""


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        with _lock:
            _conn.executescript(_SCHEMA)
            _conn.commit()
    return _conn


def init_db() -> None:
    """Idempotent — safe to call on every app start/rerun."""
    _get_conn()


# ---------------------------------------------------------------
# Threads
# ---------------------------------------------------------------

def thread_exists(thread_id: str) -> bool:
    conn = _get_conn()
    with _lock:
        row = conn.execute(
            "SELECT 1 FROM threads WHERE thread_id = ?", (thread_id,)
        ).fetchone()
    return row is not None


def create_thread(thread_id: str, title: Optional[str] = None) -> None:
    now = time.time()
    conn = _get_conn()
    with _lock:
        conn.execute(
            "INSERT OR IGNORE INTO threads (thread_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (thread_id, title, now, now),
        )
        conn.commit()


def touch_thread(thread_id: str, title: Optional[str] = None) -> None:
    """Update `updated_at` (bumps thread to top of the list) and,
    optionally, set the title if it isn't set yet."""
    conn = _get_conn()
    with _lock:
        if title:
            conn.execute(
                "UPDATE threads SET updated_at = ?, "
                "title = COALESCE(NULLIF(title, ''), ?) "
                "WHERE thread_id = ?",
                (time.time(), title, thread_id),
            )
        else:
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                (time.time(), thread_id),
            )
        conn.commit()


def list_threads(limit: int = 100) -> List[Dict[str, Any]]:
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT thread_id, title, created_at, updated_at FROM threads "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_thread(thread_id: str) -> None:
    conn = _get_conn()
    with _lock:
        conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM thread_sources WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
        conn.commit()


# ---------------------------------------------------------------
# Messages
# ---------------------------------------------------------------

def _sanitize_for_storage(obj: Any) -> Any:
    """Recursively convert LangChain `Document` objects (and anything
    else json.dumps chokes on) into plain, JSON-serializable structures.

    Node outputs from the graph (e.g. `relevant_docs`, `docs`) contain
    raw `Document` objects, which aren't JSON-serializable on their
    own — this makes it safe to pass a full graph result straight to
    `add_message(..., metadata=result)` without every caller having to
    remember to convert Documents first.
    """
    if isinstance(obj, Document):
        return {"page_content": obj.page_content, "metadata": obj.metadata}
    if isinstance(obj, dict):
        return {k: _sanitize_for_storage(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_storage(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def add_message(
    thread_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    conn = _get_conn()
    with _lock:
        conn.execute(
            "INSERT INTO messages (thread_id, role, content, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                thread_id,
                role,
                content,
                json.dumps(_sanitize_for_storage(metadata)) if metadata is not None else None,
                time.time(),
            ),
        )
        conn.commit()


def get_messages(thread_id: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT role, content, metadata, created_at FROM messages "
            "WHERE thread_id = ? ORDER BY id ASC",
            (thread_id,),
        ).fetchall()

    out = []
    for r in rows:
        d = dict(r)
        d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else None
        out.append(d)
    return out


# ---------------------------------------------------------------
# Sources (Documents) per thread
# ---------------------------------------------------------------

def _serialize_documents(docs: List[Document]) -> str:
    return json.dumps(
        [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
    )


def _deserialize_documents(raw: str) -> List[Document]:
    items = json.loads(raw)
    return [
        Document(page_content=item["page_content"], metadata=item.get("metadata") or {})
        for item in items
    ]


def add_source(thread_id: str, source_name: str, documents: List[Document]) -> None:
    conn = _get_conn()
    with _lock:
        conn.execute(
            "INSERT OR REPLACE INTO thread_sources "
            "(thread_id, source_name, documents, added_at) VALUES (?, ?, ?, ?)",
            (thread_id, source_name, _serialize_documents(documents), time.time()),
        )
        conn.commit()


def get_sources(thread_id: str) -> Dict[str, List[Document]]:
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT source_name, documents FROM thread_sources WHERE thread_id = ?",
            (thread_id,),
        ).fetchall()
    return {r["source_name"]: _deserialize_documents(r["documents"]) for r in rows}


def clear_sources(thread_id: str) -> None:
    conn = _get_conn()
    with _lock:
        conn.execute("DELETE FROM thread_sources WHERE thread_id = ?", (thread_id,))
        conn.commit()
