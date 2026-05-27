import os
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List
from shared.models import ChunkPayload

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8001")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def send_chunks(tenant_id: str, chunks: List[ChunkPayload]):
    payload = {
        "tenant_id": tenant_id,
        "chunks": [chunk.model_dump(mode="json") for chunk in chunks]
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{EMBEDDING_SERVICE_URL}/embed",
            json=payload
        )
        response.raise_for_status()
        return response.json()