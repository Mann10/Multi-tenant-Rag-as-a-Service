import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
import uuid
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager

from .database import init_db, create_session, get_session, update_session
from .orchestrator import orchestrator
from shared.models import (
    SubmitQueryRequest, ClarifyQueryRequest, QueryResponse, QueryStatus
)

app = FastAPI(title="RAG Retrieval Service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app.router.lifespan_context = lifespan


@app.post("/query/{tenant_id}", response_model=QueryResponse)
async def submit_query(tenant_id: str, request: SubmitQueryRequest, background_tasks: BackgroundTasks):
    query_id = str(uuid.uuid4())

    create_session(query_id, tenant_id, request.query)

    state = {
        "query_id": query_id,
        "tenant_id": tenant_id,
        "original_query": request.query,
        "current_query": request.query,
        "chat_history": [("human", request.query)],  # ← seed history
        "status": "pending_validation",
        "chunks": None,
        "answer": None,
        "sources": None,
        "clarification_question": None,
        "error_message": None,
        "validation_reason": None
    }

    background_tasks.add_task(run_pipeline, query_id, state)

    return QueryResponse(
        query_id=query_id,
        status=QueryStatus.PENDING_VALIDATION,
        answer=None,
        clarification_question=None,
        sources=None,
        error_message=None
    )


@app.post("/query/{query_id}/clarify", response_model=QueryResponse)
async def clarify_query(query_id: str, request: ClarifyQueryRequest, background_tasks: BackgroundTasks):
    session = get_session(query_id)
    if not session:
        raise HTTPException(status_code=404, detail="Query session not found")

    if session["status"] != QueryStatus.NEEDS_CLARIFICATION.value:
        raise HTTPException(status_code=400, detail="Query does not need clarification")

    #clarified = f"{session['original_query']}\n\nClarification: {request.clarification}"
    prior_history = json.loads(session.get("history") or "[]")
    updated_history = prior_history + [
        ("assistant", session["clarification_question"])
    ]
    update_session(
        query_id,
        current_query=request.clarification,
        status="pending_validation",
        clarification_question=None,
        history=json.dumps(updated_history)  # ← persist immediately
    )
    state = {
        "query_id": query_id,
        "tenant_id": session["tenant_id"],
        "original_query": session["original_query"],
        "current_query": request.clarification,
        "chat_history": updated_history,
        "status": "pending_validation",
        "chunks": None,
        "answer": None,
        "sources": None,
        "clarification_question": None,
        "error_message": None,
        "validation_reason": None
    }
    
    background_tasks.add_task(run_pipeline, query_id, state)

    return QueryResponse(
        query_id=query_id,
        status=QueryStatus.PENDING_VALIDATION,
        answer=None,
        clarification_question=None,
        sources=None,
        error_message=None
    )


@app.get("/query/{query_id}", response_model=QueryResponse)
async def get_query_status(query_id: str):
    session = get_session(query_id)
    if not session:
        raise HTTPException(status_code=404, detail="Query session not found")

    return QueryResponse(
        query_id=query_id,
        status=session["status"],
        answer=session["answer"],
        clarification_question=session["clarification_question"],
        sources=json.loads(session["sources"]) if session["sources"] else None,
        error_message=session["error_message"]
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "retrieval"}


async def run_pipeline(query_id: str, state: dict):
    try:
        final_state = await orchestrator.ainvoke(state)

        updates = {
            "status": final_state.get("status"),
            "answer": final_state.get("answer"),
            "sources": json.dumps(final_state.get("sources")) if final_state.get("sources") else None,
            "chunks": json.dumps(final_state.get("chunks")) if final_state.get("chunks") else None,
            "clarification_question": final_state.get("clarification_question"),
            "error_message": final_state.get("error_message"),
            "history": json.dumps(final_state.get("chat_history", [])),  # ← own column
            "context": json.dumps({
                "validation_reason": final_state.get("validation_reason"),
            })
        }
        update_session(query_id, **updates)

    except Exception as e:
        update_session(
            query_id,
            status="failed",
            error_message=str(e)
        )
        print(f"[Pipeline] Failed for {query_id}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)