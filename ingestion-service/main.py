import os
import shutil
from datetime import datetime, UTC
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

from .database import (
    init_db, get_latest_job, create_job, update_job_status,
    update_file_count, add_detected_file
)
from .parser import get_document_type, parse_pdf, parse_docx, parse_txt, parse_image
from .chunker import chunk_text
from .ocr_engine import ocr_image
from .file_watcher import start_watcher
from .embedding_client import send_chunks
from shared.models import ChunkPayload, ChunkMetadata, ChunkingStrategy, DocumentType
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

app = FastAPI(title="RAG Ingestion Service")
file_observer = None

import os
print(f"[DEBUG] OLLAMA_HOST = {os.getenv('OLLAMA_HOST')}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global file_observer
    init_db()
    file_observer = start_watcher()
    yield
    if file_observer:
        file_observer.stop()
        file_observer.join()

app.router.lifespan_context = lifespan


@app.post("/ingest/{tenant_id}")
async def trigger_ingestion(
    tenant_id: str,
    background_tasks: BackgroundTasks,
    strategy: ChunkingStrategy = Query(default=ChunkingStrategy.RECURSIVE),
    custom_tags: Optional[str] = None
):
    latest = get_latest_job(tenant_id)
    if latest and latest["status"] == "in_progress":
        raise HTTPException(
            status_code=409,
            detail=f"Ingestion already in progress for tenant {tenant_id}. "
                   f"Started at {latest['started_at']}."
        )

    create_job(tenant_id, strategy.value)
    background_tasks.add_task(run_ingestion, tenant_id, strategy.value, custom_tags)

    return {
        "tenant_id": tenant_id,
        "status": "started",
        "strategy": strategy.value,
        "message": "Ingestion job started in background."
    }


@app.get("/ingest/{tenant_id}/status")
async def get_status(tenant_id: str):
    latest = get_latest_job(tenant_id)
    if not latest:
        raise HTTPException(status_code=404, detail="No ingestion jobs found")
    return {
        "tenant_id": tenant_id,
        "status": latest["status"],
        "started_at": latest["started_at"],
        "completed_at": latest["completed_at"],
        "error_message": latest["error_message"],
        "file_count": latest["file_count"],
        "chunk_count": latest["chunk_count"]
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ingestion"}


async def run_ingestion(tenant_id: str, strategy: str, custom_tags: Optional[str] = None):
    data_dir = os.getenv("DATA_DIR", "/data")
    raw_dir = os.path.join(data_dir, tenant_id, "raw")
    processed_dir = os.path.join(data_dir, tenant_id, "processed")

    if not os.path.exists(raw_dir):
        update_job_status(tenant_id, "failed", error=f"Directory not found: {raw_dir}")
        return

    os.makedirs(processed_dir, exist_ok=True)

    files = [f for f in os.listdir(raw_dir) if not f.startswith(".")]
    update_file_count(tenant_id, len(files))

    all_chunks = []
    errors = []

    for filename in files:
        file_path = os.path.join(raw_dir, filename)
        try:
            doc_type = get_document_type(filename)
            print(f"Document type is {doc_type}")
            if doc_type == DocumentType.PDF:
                pages = parse_pdf(file_path, tenant_id)
            elif doc_type == DocumentType.DOCX:
                pages = parse_docx(file_path, tenant_id)
            elif doc_type == DocumentType.TXT:
                pages = parse_txt(file_path, tenant_id)
            elif doc_type == DocumentType.IMAGE:
                pages = parse_image(file_path, tenant_id)
            else:
                continue

            for page in pages:
                page_text = page["text"] or ""
                image_paths = []

                for img_info in page.get("images", []):
                    ocr_text = ocr_image(img_info["path"])
                    if ocr_text:
                        page_text += f"\n\n[Image Text]:\n{ocr_text}"
                        image_paths.append(img_info["path"])

                if not page_text.strip():
                    continue
                print(f"Page text is {page_text}\n\n")
                print(f"Image path is {image_paths}")
                chunks = chunk_text(page_text, strategy)
                normalized_paths = [
                    str(Path(p).as_posix())
                    for p in image_paths
                ]
                for idx, chunk_text_content in enumerate(chunks):
                    metadata = ChunkMetadata(
                        tenant_id=tenant_id,
                        filename=filename,
                        document_type=doc_type,
                        page_number=page.get("page_number"),
                        chunk_index=idx,
                        total_chunks=len(chunks),
                        timestamp=datetime.now(UTC),
                        custom_tags={"tag": custom_tags} if custom_tags else None,
                        source_image_paths=normalized_paths if normalized_paths else None,
                    )
                    all_chunks.append(
                        ChunkPayload(text=chunk_text_content, metadata=metadata)
                    )

            shutil.move(file_path, os.path.join(processed_dir, filename))

        except Exception as e:
            err = f"Failed to process {filename}: {str(e)}"
            print(err)
            errors.append(err)

    if not all_chunks:
        update_job_status(
            tenant_id,
            "completed" if not errors else "failed",
            error="; ".join(errors) if errors else None,
            chunks=0
        )
        return

    batch_size = 128
    total_stored = 0
    try:
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            result = await send_chunks(tenant_id, batch)
            total_stored += result.get("stored", 0)

        final_status = "completed" if not errors else "failed"
        update_job_status(
            tenant_id,
            final_status,
            error="; ".join(errors) if errors else None,
            chunks=total_stored
        )
    except Exception as e:
        update_job_status(tenant_id, "failed", error=f"Embedding service error: {str(e)}")
        print(f"Fatal embedding error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)