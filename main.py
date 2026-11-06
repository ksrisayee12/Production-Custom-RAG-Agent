"""
main.py — FastAPI application with Inngest orchestration.

Preserved architecture:
- FastAPI app
- Inngest client (app_id="Prod_RAG_App")
- rag/ingest_pdf event → rag_ingest_pdf function
- rag/query_pdf_ai event → rag_query_pdf_ai function

Improvements:
- Full semantic chunking pipeline via chunking_engine
- Rich ChunkMetadata stored in Qdrant
- Safe document reprocessing (delete stale chunks first)
- Improved query pipeline with context assembly
- Grounded answer generation
- Structured logging
"""

from __future__ import annotations

import datetime
import logging
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
import google.generativeai as genai
import inngest
import inngest.fast_api

from chunking_engine import PageContent, chunk_document
from context_builder import assemble_context, build_prompt
from custom_types import (
    AssembledContext,
    ChunkingConfig,
    ChunkingResult,
    ChunkingStrategy,
    RAGChunkAndSrc,
    RAGQueryResult,
    RAGSearchResult,
    RAGUpsertResult,
    RetrievalConfig,
    RetrievedChunk,
    RetrievalResult,
)
from data_loader import embed_text, load_pdf_pages
from vector_db import QdrantStorage

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag.main")

# FIX: API Key Fallback - Support both GEMINI_API_KEY and GOOGLE_API_KEY
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

app = FastAPI(title="RAG Document Intelligence Platform")

inngest_client = inngest.Inngest(
    app_id="Prod_RAG_App",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion function
# ─────────────────────────────────────────────────────────────────────────────


# FIX: Inngest Function Retries - configure retries=3 for resilient ingestion against network hiccups
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    retries=3,
)
async def rag_ingest_pdf(ctx: inngest.Context):
    """
    Step 1 — load_and_chunk: load PDF, run hybrid semantic chunking.
    Step 2 — embed_and_upsert: generate embeddings, delete stale, store in Qdrant.
    """

    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        raw_source = ctx.event.data.get("source_id", pdf_path)
        # FIX: Source ID Normalization - standardize source_id using Path(raw_source).name
        source_id = Path(raw_source).name
        strategy_name = ctx.event.data.get("strategy", ChunkingStrategy.HYBRID_SEMANTIC.value)

        logger.info("Loading PDF: %s (normalized source_id=%s)", pdf_path, source_id)

        try:
            strategy = ChunkingStrategy(strategy_name)
        except ValueError:
            strategy = ChunkingStrategy.HYBRID_SEMANTIC

        pages: list[PageContent] = load_pdf_pages(pdf_path)

        if not pages:
            raise ValueError(f"No readable pages found in: {pdf_path}")

        config = ChunkingConfig(strategy=strategy)
        result: ChunkingResult = chunk_document(
            pages=pages,
            source_id=source_id,
            config=config,
            embed_fn=embed_text,
        )

        logger.info(
            "Chunking complete: %d chunks from %d pages (strategy=%s)",
            result.total_chunks, len(pages), strategy.value
        )

        # Backward-compat: also pass plain text chunks
        return RAGChunkAndSrc(
            chunks=[c.text for c in result.chunks],
            source_id=source_id,
            chunking_result=result,
        )

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        if isinstance(chunks_and_src, dict):
            chunks_and_src = RAGChunkAndSrc.model_validate(chunks_and_src)
        result = chunks_and_src.chunking_result
        source_id = chunks_and_src.source_id

        if result is None or not result.chunks:
            # Legacy fallback path
            chunks_text = chunks_and_src.chunks
            if not chunks_text:
                return RAGUpsertResult(ingested=0)
            vectors = embed_text(chunks_text, task_type="retrieval_document")
            ids = [
                str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
                for i in range(len(chunks_text))
            ]
            payloads = [{"source": source_id, "text": chunks_text[i]} for i in range(len(chunks_text))]
            store = QdrantStorage()
            store.upsert(ids, vectors, payloads)
            return RAGUpsertResult(ingested=len(chunks_text), source=source_id)

        # Rich metadata path
        chunk_metas = result.chunks
        texts = [c.text for c in chunk_metas]

        logger.info("Generating embeddings for %d chunks...", len(texts))
        vectors = embed_text(texts, task_type="retrieval_document")

        store = QdrantStorage()

        # Safe reprocessing: delete stale chunks
        deleted = store.delete_by_source(source_id)
        if deleted > 0:
            logger.info("Removed %d stale chunks for source '%s'", deleted, source_id)

        store.upsert_chunks(chunk_metas, vectors)

        return RAGUpsertResult(
            ingested=len(chunk_metas),
            document_id=result.document_id,
            source=source_id,
            chunk_ids=[c.chunk_id for c in chunk_metas],
        )

    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc,
    )

    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult,
    )

    return ingested.model_dump() if hasattr(ingested, "model_dump") else ingested


# ─────────────────────────────────────────────────────────────────────────────
# Query function
# ─────────────────────────────────────────────────────────────────────────────


@inngest_client.create_function(
    fn_id="rag_query_pdf_ai",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    """
    Step 1 — embed-and-search: embed query, retrieve chunks with metadata.
    Step 2 — llm-answer: assemble context, generate grounded answer.
    """
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    raw_source = ctx.event.data.get("source_id") or ctx.event.data.get("pdf_path")
    # FIX: Source ID Normalization - standardize source_id query filter using Path(raw_source).name
    source_id = Path(raw_source).name if raw_source else None
    enable_neighbors = bool(ctx.event.data.get("enable_neighbors", True))
    similarity_threshold = float(ctx.event.data.get("similarity_threshold", 0.0))

    def _search(
        question: str,
        top_k: int,
        source_id: str | None,
    ) -> RAGSearchResult:
        query_vec = embed_text([question], task_type="retrieval_query")[0]
        store = QdrantStorage()

        config = RetrievalConfig(
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            enable_neighbor_expansion=enable_neighbors,
        )
        result: RetrievalResult = store.search_chunks(
            query_vec, config=config, source_id=source_id
        )

        return RAGSearchResult(
            contexts=[c.text for c in result.retrieved],
            sources=result.sources,
            retrieved_chunks=result.retrieved,
        )

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k, source_id),
        output_type=RAGSearchResult,
    )

    def _generate_answer(found: RAGSearchResult) -> str:
        if isinstance(found, dict):
            found = RAGSearchResult.model_validate(found)
        config = RetrievalConfig()
        context: AssembledContext = assemble_context(found.retrieved_chunks, config)

        if not context.context_text.strip():
            return (
                "I could not find relevant information in the indexed documents "
                "to answer your question. Please ensure the relevant documents have "
                "been uploaded and indexed."
            )

        prompt = build_prompt(question, context)

        # Select active generative model with graceful fallbacks
        candidate_models = [
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-3.8-flash",
            "gemini-flash-latest",
        ]
        response = None
        last_err = None
        for mod_name in candidate_models:
            try:
                model = genai.GenerativeModel(mod_name)
                gen_config = genai.types.GenerationConfig(
                    max_output_tokens=2048,
                    temperature=0.1,
                )
                response = model.generate_content(prompt, generation_config=gen_config)
                if response:
                    break
            except Exception as e:
                logger.warning("Generation with %s failed: %s, trying next fallback...", mod_name, e)
                last_err = e

        if response is None:
            raise RuntimeError(f"All generative model fallbacks failed. Last error: {last_err}")

        # FIX: Robust multi-part fallback extraction
        try:
            return response.text
        except Exception:
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                part_texts = [p.text for p in response.candidates[0].content.parts if getattr(p, "text", None)]
                if part_texts:
                    return "".join(part_texts)
            return "Unable to generate an answer based on the provided context."

    answer = await ctx.step.run(
        "llm-answer",
        lambda: _generate_answer(found),
    )

    if isinstance(found, dict):
        found = RAGSearchResult.model_validate(found)

    return RAGQueryResult(
        answer=answer,
        sources=found.sources,
        num_contexts=len(found.contexts),
        retrieved_chunks=found.retrieved_chunks,
        context_token_count=sum(c.token_count or 0 for c in found.retrieved_chunks),
    ).model_dump()


inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
