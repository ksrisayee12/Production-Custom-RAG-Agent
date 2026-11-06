"""
vector_db.py — Qdrant storage, retrieval, and document management.

Improvements over the original:
- Full ChunkMetadata payload (not just {source, text})
- Safe document reprocessing (delete stale chunks by source before re-upserting)
- Similarity score exposure
- Neighboring chunk retrieval
- Source filtering preserved
- Analytics queries
- Graceful error handling
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
    ScrollRequest,
    # FIX: Qdrant Payload Indexing - import PayloadSchemaType for explicit keyword index creation
    PayloadSchemaType,
)

from custom_types import (
    AnalyticsSnapshot,
    ChunkMetadata,
    ChunkingStrategy,
    DocumentMetadata,
    IngestionStatus,
    RetrievedChunk,
    RetrievalConfig,
    RetrievalResult,
)

load_dotenv()
logger = logging.getLogger(__name__)


class QdrantStorage:

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection: str = "docs",
        dim: int = 3072,
    ):
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection = collection
        self.dim = dim

        if not self.url or not self.api_key:
            raise ValueError("QDRANT_URL or QDRANT_API_KEY missing in .env")

        self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=30)

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '%s' (dim=%d)", self.collection, dim)

        # FIX: Qdrant Payload Indexing - ensure explicit keyword payload index on 'source' field
        try:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="source",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Ensured keyword payload index for 'source' in collection '%s'", self.collection)
        except Exception as exc:
            # FIX: Qdrant Payload Indexing - wrap in safe try...except to handle duplicate/existing index gracefully
            logger.debug("Payload index on 'source' already exists or notice received: %s", exc)

    # ─────────────────────────────────────────────────────────────────────────
    # Upsert — legacy simple interface (backward compat)
    # ─────────────────────────────────────────────────────────────────────────

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        """Legacy upsert: accepts raw ids/vectors/payloads."""
        if not ids:
            return
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        logger.debug("Upserted %d points", len(points))

    # ─────────────────────────────────────────────────────────────────────────
    # Upsert — rich metadata interface
    # ─────────────────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[ChunkMetadata],
        vectors: list[list[float]],
    ) -> None:
        """
        Upsert ChunkMetadata objects with their corresponding vectors.
        Validates embedding dimensions.
        Raises ValueError on dimension mismatch.
        """
        if not chunks:
            return

        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks/vectors length mismatch: {len(chunks)} vs {len(vectors)}"
            )

        # Validate dimensions
        for i, vec in enumerate(vectors):
            if len(vec) != self.dim:
                raise ValueError(
                    f"Vector at index {i} has dim {len(vec)}, expected {self.dim}"
                )

        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=vectors[i],
                payload=chunk.model_dump(),
            )
            for i, chunk in enumerate(chunks)
        ]

        self.client.upsert(collection_name=self.collection, points=points)
        logger.info("Upserted %d chunks for source '%s'", len(points), chunks[0].source)

    # ─────────────────────────────────────────────────────────────────────────
    # Document management
    # ─────────────────────────────────────────────────────────────────────────

    def delete_by_source(self, source_id: str) -> int:
        """
        Delete all points associated with a source_id.
        Returns number of deleted points.
        Used for safe document reprocessing.
        """
        existing_ids = self._get_ids_by_source(source_id)
        if not existing_ids:
            return 0

        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=existing_ids),
        )
        logger.info(
            "Deleted %d stale chunks for source '%s'", len(existing_ids), source_id
        )
        return len(existing_ids)

    def _get_ids_by_source(self, source_id: str) -> list[str]:
        """Scroll through all points matching source_id and collect their IDs."""
        ids: list[str] = []
        offset = None
        # FIX: Qdrant Query Failures - match indexed keyword 'source'
        query_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_id))]
        )
        while True:
            result, next_offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=query_filter,
                with_payload=False,
                with_vectors=False,
                limit=100,
                offset=offset,
            )
            for point in result:
                ids.append(str(point.id))
            if next_offset is None:
                break
            offset = next_offset
        return ids

    # ─────────────────────────────────────────────────────────────────────────
    # Search — rich retrieval
    # ─────────────────────────────────────────────────────────────────────────

    def search_chunks(
        self,
        query_vector: list[float],
        config: Optional[RetrievalConfig] = None,
        source_id: str | None = None,
    ) -> RetrievalResult:
        """
        Full semantic search returning RetrievedChunk objects with scores,
        optional source filtering, similarity threshold, and neighbor expansion.
        """
        if config is None:
            config = RetrievalConfig()

        query_filter = None
        if source_id:
            # FIX: Qdrant Query Failures - universal search filter with MatchValue on keyword-indexed source
            query_filter = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_id))]
            )

        # FIX: Qdrant Query Failures - query_points execution and safe point response iteration
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            with_payload=True,
            limit=config.top_k,
        )

        primary_chunks: list[RetrievedChunk] = []

        for point in response.points:
            payload = getattr(point, "payload", {}) or {}
            score = float(getattr(point, "score", 0.0))

            if score < config.similarity_threshold:
                continue

            chunk = _payload_to_retrieved_chunk(payload, score, is_neighbor=False)
            if chunk:
                primary_chunks.append(chunk)

        # Neighbor expansion
        neighbor_chunks: list[RetrievedChunk] = []
        if config.enable_neighbor_expansion:
            neighbor_chunks = self._retrieve_neighbors(
                primary_chunks, config, query_filter
            )

        all_chunks = primary_chunks + neighbor_chunks
        sources = list(dict.fromkeys(c.source for c in all_chunks))
        context_tokens = sum(c.token_count or 0 for c in all_chunks)

        return RetrievalResult(
            retrieved=all_chunks,
            sources=sources,
            total_retrieved=len(all_chunks),
            context_token_count=context_tokens,
        )

    def _retrieve_neighbors(
        self,
        primary_chunks: list[RetrievedChunk],
        config: RetrievalConfig,
        query_filter,
    ) -> list[RetrievedChunk]:
        """
        For chunks with prev/next IDs, retrieve neighbors if context budget allows.
        Only retrieves neighbors for the top-3 primary chunks to avoid context bloat.
        """
        neighbor_ids: set[str] = set()
        primary_ids = {c.chunk_id for c in primary_chunks}

        for chunk in primary_chunks[:3]:   # limit to top-3 primary chunks
            if chunk.previous_chunk_id:
                neighbor_ids.add(chunk.previous_chunk_id)
            if chunk.next_chunk_id:
                neighbor_ids.add(chunk.next_chunk_id)

        # Exclude already-retrieved primaries
        neighbor_ids -= primary_ids

        if not neighbor_ids:
            return []

        neighbors: list[RetrievedChunk] = []
        for nid in list(neighbor_ids)[:6]:  # hard cap
            try:
                results = self.client.retrieve(
                    collection_name=self.collection,
                    ids=[nid],
                    with_payload=True,
                )
                for point in results:
                    payload = getattr(point, "payload", {}) or {}
                    chunk = _payload_to_retrieved_chunk(payload, score=0.0, is_neighbor=True)
                    if chunk:
                        neighbors.append(chunk)
            except Exception as exc:
                logger.warning("Failed to retrieve neighbor '%s': %s", nid, exc)

        return neighbors

    # ─────────────────────────────────────────────────────────────────────────
    # Legacy search (backward compat with original RAGSearchResult)
    # ─────────────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        source_id: str | None = None,
    ):
        """Legacy search returning (contexts, sources) tuple."""
        config = RetrievalConfig(top_k=top_k, enable_neighbor_expansion=False)
        result = self.search_chunks(query_vector, config, source_id=source_id)

        contexts = [c.text for c in result.retrieved]
        sources = result.sources
        return contexts, sources

    # ─────────────────────────────────────────────────────────────────────────
    # Analytics
    # ─────────────────────────────────────────────────────────────────────────

    def get_analytics(self) -> AnalyticsSnapshot:
        """
        Collect actual statistics from the Qdrant collection.
        Returns AnalyticsSnapshot with real measured values.
        """
        try:
            info = self.client.get_collection(self.collection)
            total_chunks = info.points_count or 0
        except Exception:
            total_chunks = 0

        if total_chunks == 0:
            return AnalyticsSnapshot(total_chunks=0)

        # Sample up to 500 points to compute stats
        sample_points, _ = self.client.scroll(
            collection_name=self.collection,
            with_payload=True,
            with_vectors=False,
            limit=500,
        )

        source_map: dict[str, DocumentMetadata] = {}
        total_tokens = 0
        token_count_samples = 0
        strategy_dist: dict[str, int] = {}

        for point in sample_points:
            payload = getattr(point, "payload", {}) or {}
            source = payload.get("source", "unknown")
            doc_id = payload.get("document_id", source)
            file_name = payload.get("file_name", source)
            token_count = payload.get("token_count")
            strategy = payload.get("chunking_method", ChunkingStrategy.SENTENCE.value)

            if token_count:
                total_tokens += int(token_count)
                token_count_samples += 1

            strategy_dist[strategy] = strategy_dist.get(strategy, 0) + 1

            if source not in source_map:
                source_map[source] = DocumentMetadata(
                    document_id=doc_id,
                    source=source,
                    file_name=file_name,
                    status=IngestionStatus.INDEXED,
                )
            # Accumulate chunk count
            doc_meta = source_map[source]
            doc_meta.chunk_count = (doc_meta.chunk_count or 0) + 1
            if payload.get("chunking_method"):
                try:
                    doc_meta.chunking_strategy = ChunkingStrategy(payload["chunking_method"])
                except ValueError:
                    pass

        avg_tokens = (
            total_tokens / token_count_samples if token_count_samples > 0 else 0.0
        )

        return AnalyticsSnapshot(
            total_documents=len(source_map),
            total_chunks=total_chunks,
            avg_chunk_tokens=round(avg_tokens, 1),
            strategy_distribution=strategy_dist,
            documents=list(source_map.values()),
        )

    def get_chunks_for_source(self, source_id: str) -> list[ChunkMetadata]:
        """
        Retrieve all chunks for a given source (for Chunk Explorer).
        """
        query_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_id))]
        )
        all_chunks: list[ChunkMetadata] = []
        offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                limit=100,
                offset=offset,
            )
            for point in points:
                payload = getattr(point, "payload", {}) or {}
                try:
                    chunk = ChunkMetadata(**payload)
                    if chunk.chunk_index is None:
                        chunk.chunk_index = len(all_chunks)
                    all_chunks.append(chunk)
                except Exception:
                    # If payload doesn't match (legacy), build minimal
                    text = payload.get("text", "")
                    source = payload.get("source", "")
                    if text:
                        idx = payload.get("chunk_index")
                        if idx is None:
                            idx = len(all_chunks)
                        all_chunks.append(
                            ChunkMetadata(
                                document_id=payload.get("document_id", source),
                                source=source,
                                chunk_id=str(point.id),
                                chunk_index=idx,
                                text=text,
                            )
                        )

            if next_offset is None:
                break
            offset = next_offset

        # Normalize legacy chunks where chunk_index was omitted or all zero
        if len(all_chunks) > 1 and all(c.chunk_index == 0 for c in all_chunks):
            for i, c in enumerate(all_chunks):
                c.chunk_index = i

        # Sort by chunk_index
        all_chunks.sort(key=lambda c: c.chunk_index or 0)
        return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _payload_to_retrieved_chunk(
    payload: dict,
    score: float,
    is_neighbor: bool = False,
) -> Optional[RetrievedChunk]:
    """Convert a Qdrant payload dict to a RetrievedChunk."""
    text = payload.get("text", "")
    source = payload.get("source", "")
    chunk_id = payload.get("chunk_id", "")

    if not text or not source:
        return None

    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        source=source,
        document_id=payload.get("document_id"),
        chunk_index=payload.get("chunk_index"),
        page_number=payload.get("page_number"),
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
        section=payload.get("section"),
        subsection=payload.get("subsection"),
        parent_context=payload.get("parent_context"),
        token_count=payload.get("token_count"),
        previous_chunk_id=payload.get("previous_chunk_id"),
        next_chunk_id=payload.get("next_chunk_id"),
        is_neighbor=is_neighbor,
    )
