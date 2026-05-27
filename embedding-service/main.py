from fastapi import FastAPI, HTTPException
from langsmith import traceable
from shared.models import EmbedRequest
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from .embedder import get_embeddings
from .pinecone_store import upsert_vectors,query_namespace

app = FastAPI(title="RAG Embedding Service")

class EmbedTextsRequest(BaseModel):
    texts: List[str]
    input_type: str = "query"


class QueryRequest(BaseModel):
    tenant_id: str
    vector: List[float]
    top_k: int = 5
    filter: Optional[Dict[str, Any]] = None


@app.post("/embed")
@traceable(run_type="chain", name="embed_and_upsert")
async def embed_documents(request: EmbedRequest):
    if not request.chunks:
        return {"stored": 0, "tenant_id": request.tenant_id}
    print(f"Number of chunks are : {len(request.chunks)}")
    texts = [chunk.text for chunk in request.chunks]

    try:
        embeddings = get_embeddings(texts)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {str(e)}")

    try:
        stored = upsert_vectors(request.tenant_id, request.chunks, embeddings)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Pinecone upsert failed: {str(e)}")

    return {"stored": stored, "tenant_id": request.tenant_id}

@app.post("/embed-texts")
@traceable(run_type="chain", name="embed_texts_only")
async def embed_texts(request: EmbedTextsRequest):
    if not request.texts:
        return {"embeddings": []}

    try:
        embeddings = get_embeddings(request.texts, input_type=request.input_type)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {str(e)}")

    return {"embeddings": embeddings}

@app.post("/query")
@traceable(run_type="chain", name="pinecone_query")
async def query_vectors(request: QueryRequest):
    try:
        matches = query_namespace(
            tenant_id=request.tenant_id,
            vector=request.vector,
            top_k=request.top_k,
            filter=request.filter
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Pinecone query failed: {str(e)}")

    return {
        "tenant_id": request.tenant_id,
        "matches": matches,
        "count": len(matches)
    }

@app.get("/health")
async def health():
    return {"status": "ok", "service": "embedding"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)