import os
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict, Any, Optional

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8001")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def get_query_embeddings(texts: List[str]) -> List[List[float]]:
    payload = {
        "texts": texts,
        "input_type": "query"
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{EMBEDDING_SERVICE_URL}/embed-texts",
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def query_pinecone(tenant_id: str, vector: List[float], top_k: int = 5,
                         filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    payload = {
        "tenant_id": tenant_id,
        "vector": vector,
        "top_k": top_k,
        "filter": filter
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{EMBEDDING_SERVICE_URL}/query",
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        return data["matches"]