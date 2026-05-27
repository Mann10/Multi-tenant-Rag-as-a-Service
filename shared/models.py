from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ChunkingStrategy(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    IMAGE = "image"


class ChunkMetadata(BaseModel):
    tenant_id: str
    filename: str
    document_type: DocumentType
    page_number: Optional[int] = None
    chunk_index: int
    total_chunks: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    custom_tags: Optional[Dict[str, Any]] = None
    source_image_paths: Optional[List[str]] = None
    ocr_confidence: Optional[float] = None


class ChunkPayload(BaseModel):
    text: str
    metadata: ChunkMetadata


class EmbedRequest(BaseModel):
    tenant_id: str
    chunks: List[ChunkPayload]


class IngestionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    
class QueryStatus(str, Enum):
    PENDING_VALIDATION = "pending_validation"
    NEEDS_CLARIFICATION = "needs_clarification"
    VALIDATED = "validated"
    RETRIEVING = "retrieving"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryContext(BaseModel):
    query_id: str
    tenant_id: str
    original_query: str
    current_query: str
    status: QueryStatus
    clarification_question: Optional[str] = None
    chunks: Optional[List[Dict[str, Any]]] = None
    answer: Optional[str] = None
    sources: Optional[List[str]] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SubmitQueryRequest(BaseModel):
    query: str


class ClarifyQueryRequest(BaseModel):
    clarification: str


class QueryResponse(BaseModel):
    query_id: str
    status: QueryStatus
    answer: Optional[str] = None
    clarification_question: Optional[str] = None
    sources: Optional[List[str]] = None
    error_message: Optional[str] = None