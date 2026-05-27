import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

DB_PATH = os.getenv("DB_PATH", "./ingestion.db")


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
            CREATE TABLE IF NOT EXISTS ingestion_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                status TEXT NOT NULL,
                strategy TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                file_count INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detected_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed INTEGER DEFAULT 0,
                UNIQUE(tenant_id, file_path)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_tenant 
            ON ingestion_jobs(tenant_id, started_at)
        """)
        conn.commit()


def get_latest_job(tenant_id: str):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE tenant_id = ? ORDER BY started_at DESC LIMIT 1",
            (tenant_id,)
        ).fetchone()
        return dict(row) if row else None


def create_job(tenant_id: str, strategy: str):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO ingestion_jobs (tenant_id, status, strategy) VALUES (?, ?, ?)",
            (tenant_id, "in_progress", strategy)
        )
        conn.commit()


def update_job_status(tenant_id: str, status: str, error: str = None, chunks: int = None):
    with get_db() as conn:
        if status in ("completed", "failed"):
            conn.execute("""
                UPDATE ingestion_jobs 
                SET status = ?, error_message = ?, chunk_count = ?, completed_at = ? 
                WHERE tenant_id = ? AND status = 'in_progress'
            """, (status, error, chunks, datetime.utcnow(), tenant_id))
        else:
            conn.execute("""
                UPDATE ingestion_jobs SET status = ? 
                WHERE tenant_id = ? AND status = 'in_progress'
            """, (status, tenant_id))
        conn.commit()


def update_file_count(tenant_id: str, count: int):
    with get_db() as conn:
        conn.execute("""
            UPDATE ingestion_jobs SET file_count = ? 
            WHERE tenant_id = ? AND status = 'in_progress'
        """, (count, tenant_id))
        conn.commit()


def add_detected_file(tenant_id: str, filename: str, file_path: str):
    with get_db() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO detected_files (tenant_id, filename, file_path) 
            VALUES (?, ?, ?)
        """, (tenant_id, filename, file_path))
        conn.commit()