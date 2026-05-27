import sqlite3
import json
import os
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Dict, Any, List

DB_PATH = os.getenv("DB_PATH", "./retrieval.db")


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_sessions (
                query_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                original_query TEXT NOT NULL,
                current_query TEXT NOT NULL,
                status TEXT NOT NULL,
                context TEXT NOT NULL DEFAULT '[]',
                history TEXT NOT NULL DEFAULT '[]',
                chunks TEXT,
                answer TEXT,
                sources TEXT,
                clarification_question TEXT,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_tenant 
            ON retrieval_sessions(tenant_id, created_at)
        """)
        conn.commit()


def create_session(query_id: str, tenant_id: str, query: str) -> None:
    with get_db() as conn:
        conn.execute("""
            INSERT INTO retrieval_sessions 
            (query_id, tenant_id, original_query, current_query, status, context)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            query_id, tenant_id, query, query,
            "pending_validation",
            json.dumps({"history": []})
        ))
        conn.commit()


def get_session(query_id: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM retrieval_sessions WHERE query_id = ?", 
            (query_id,)
        ).fetchone()
        if not row:
            return None
        return dict(row)


def update_session(query_id: str, **fields) -> None:
    allowed = {
        "current_query", "status", "context", "chunks", 
        "answer", "sources", "clarification_question", "error_message"
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return

    updates["updated_at"] = datetime.utcnow().isoformat()
    set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
    values = list(updates.values()) + [query_id]

    with get_db() as conn:
        conn.execute(
            f"UPDATE retrieval_sessions SET {set_clause} WHERE query_id = ?",
            values
        )
        conn.commit()


def list_sessions(tenant_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM retrieval_sessions WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ?",
            (tenant_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]