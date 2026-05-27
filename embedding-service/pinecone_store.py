import os
import json
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
from typing import List
from shared.models import ChunkPayload
from typing import List, Optional, Dict, Any
import uuid

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "multi-tenant-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

pc = Pinecone(api_key=PINECONE_API_KEY)

existing = [idx.name for idx in pc.list_indexes()]
if PINECONE_INDEX not in existing:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )

index = pc.Index(PINECONE_INDEX)


def upsert_vectors(tenant_id: str, chunks: List[ChunkPayload], embeddings: List[List[float]]) -> int:
    vectors = []

    for chunk, embedding in zip(chunks, embeddings):
        meta = chunk.metadata

        pinecone_meta = {
            "tenant_id": meta.tenant_id,
            "filename": meta.filename,
            "document_type": str(meta.document_type.value),
            "page_number": meta.page_number or 0,
            "chunk_index": meta.chunk_index,
            "total_chunks": meta.total_chunks,
            "timestamp": meta.timestamp.isoformat() if isinstance(meta.timestamp, datetime) else str(meta.timestamp),
            "text": chunk.text[:30000],
        }

        try:
            if meta.custom_tags:
                pinecone_meta["custom_tags"] = json.dumps(meta.custom_tags)

            if meta.source_image_paths:
                pinecone_meta["source_image_paths"] = json.dumps(meta.source_image_paths)

            if meta.ocr_confidence:
                pinecone_meta["ocr_confidence"] = meta.ocr_confidence

            vector_id = (
                f"{meta.tenant_id}_"
                f"{meta.filename}_"
                f"p{meta.page_number}_"
                f"c{meta.chunk_index}_"
                f"{uuid.uuid4().hex}"
            )
            
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": pinecone_meta
            })
        except Exception as e:
            print(f"Error is {e}")

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size], namespace=tenant_id)

    return len(vectors)

def query_namespace(tenant_id: str, vector: List[float], top_k: int = 5,
                    filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    results = index.query(
        namespace=tenant_id,
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter
    )
    return [
        {
            "id": match.id,
            "score": match.score,
            "text": match.metadata.get("text", ""),
            "source": match.metadata.get("filename", ""),
            "page": match.metadata.get("page_number", 0),
            "document_type": match.metadata.get("document_type", ""),
            "chunk_index": match.metadata.get("chunk_index", 0),
            "total_chunks": match.metadata.get("total_chunks", 0),
            "timestamp": match.metadata.get("timestamp", ""),
            "custom_tags": match.metadata.get("custom_tags", None),
            "source_image_paths": match.metadata.get("source_image_paths", None),
            "ocr_confidence": match.metadata.get("ocr_confidence", None),
        }
        for match in results.matches
    ]