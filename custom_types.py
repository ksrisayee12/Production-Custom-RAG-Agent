"""
custom_types.py — Structured Pydantic models for the RAG platform.
Covers document metadata, chunk metadata, chunking config/results,
retrieval, ingestion status, and query responses.
"""

from __future__ import annotations

import enum
from typing import Optional

import pydantic


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class ChunkingStrategy(str, enum.Enum):
    FIXED = "fixed"
    SENTENCE = "sentence"
    RECURSIVE = "recursive"
    STRUCTURE_AWARE = "structure_aware"
    SEMANTIC = "semantic"
    HYBRID_SEMANTIC = "hybrid_semantic"


class IngestionStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class StructureType(str, enum.Enum):
    HEADING = "heading"
    SUBHEADING = "subheading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    NUMBERED_LIST = "numbered_list"
    TABLE = "table"
    CODE_BLOCK = "code_block"
    DEFINITION = "definition"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration models
# ─────────────────────────────────────────────────────────────────────────────


class ChunkingConfig(pydantic.BaseModel):
    """Configuration for the chunking engine."""

    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID_SEMANTIC
    chunk_size: int = 1000
    chunk_overlap: int = 150
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    semantic_threshold: float = 0.35   # cosine-distance threshold for boundaries
    enable_context_enrichment: bool = True
    embedding_batch_size: int = 50


class RetrievalConfig(pydantic.BaseModel):
    """Configuration for the retrieval pipeline."""

    top_k: int = 5
    similarity_threshold: float = 0.0
    enable_neighbor_expansion: bool = True
    max_context_tokens: int = 8000
    dedup_threshold: float = 0.92     # cosine-similarity above which a chunk is a dup


# ─────────────────────────────────────────────────────────────────────────────
# Document-level models
# ─────────────────────────────────────────────────────────────────────────────


class DocumentMetadata(pydantic.BaseModel):
    """Metadata about an ingested document."""

    document_id: str
    source: str                          # user-facing source identifier
    file_name: str
    file_size_bytes: Optional[int] = None
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    chunking_strategy: Optional[ChunkingStrategy] = None
    status: IngestionStatus = IngestionStatus.UPLOADED
    uploaded_at: Optional[str] = None   # ISO-8601 string
    indexed_at: Optional[str] = None
    error_message: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Chunk-level models
# ─────────────────────────────────────────────────────────────────────────────


class ChunkMetadata(pydantic.BaseModel):
    """Full metadata stored in Qdrant payload for every chunk."""

    document_id: str
    source: str
    chunk_id: str
    chunk_index: int
    text: str
    page_number: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    parent_context: Optional[str] = None
    token_count: Optional[int] = None
    chunking_method: str = ChunkingStrategy.HYBRID_SEMANTIC.value
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    document_version: Optional[str] = None
    structure_type: Optional[str] = None
    semantic_score: Optional[float] = None  # boundary similarity at creation time


class ChunkingResult(pydantic.BaseModel):
    """Output of the chunking engine for one document."""

    document_id: str
    source: str
    strategy: ChunkingStrategy
    chunks: list[ChunkMetadata]
    total_chunks: int
    avg_chunk_tokens: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Legacy Inngest-compatible models (kept for backward compat)
# ─────────────────────────────────────────────────────────────────────────────


class RAGChunkAndSrc(pydantic.BaseModel):
    """Payload passed between load and upsert Inngest steps."""

    chunks: list[str]
    # FIX: Pydantic v2 Schema Compatibility - Optional[str] = None ensures strict Pydantic 2.14+ validation compliance
    source_id: Optional[str] = None
    chunking_result: Optional[ChunkingResult] = None


class RAGUpsertResult(pydantic.BaseModel):
    ingested: int
    document_id: Optional[str] = None
    source: Optional[str] = None
    chunk_ids: list[str] = pydantic.Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval models
# ─────────────────────────────────────────────────────────────────────────────


class RetrievedChunk(pydantic.BaseModel):
    """A single chunk returned from vector search, with full metadata."""

    chunk_id: str
    text: str
    score: float
    source: str
    document_id: Optional[str] = None
    chunk_index: Optional[int] = None
    page_number: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    parent_context: Optional[str] = None
    token_count: Optional[int] = None
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    is_neighbor: bool = False           # True when retrieved as neighbor, not by score


class RetrievalResult(pydantic.BaseModel):
    """Full result from the retrieval pipeline."""

    retrieved: list[RetrievedChunk]
    sources: list[str]
    total_retrieved: int
    context_token_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Assembled context model
# ─────────────────────────────────────────────────────────────────────────────


class AssembledContext(pydantic.BaseModel):
    """Context passed to the LLM for answer generation."""

    context_text: str
    chunks_used: list[RetrievedChunk]
    sources: list[str]
    token_count: int
    truncated: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Legacy search model (kept for Inngest step backward compat)
# ─────────────────────────────────────────────────────────────────────────────


class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]
    retrieved_chunks: list[RetrievedChunk] = pydantic.Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Query response
# ─────────────────────────────────────────────────────────────────────────────


class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int
    retrieved_chunks: list[RetrievedChunk] = pydantic.Field(default_factory=list)
    context_token_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Analytics model
# ─────────────────────────────────────────────────────────────────────────────


class AnalyticsSnapshot(pydantic.BaseModel):
    total_documents: int = 0
    total_chunks: int = 0
    avg_chunk_tokens: float = 0.0
    strategy_distribution: dict[str, int] = pydantic.Field(default_factory=dict)
    documents: list[DocumentMetadata] = pydantic.Field(default_factory=list)
