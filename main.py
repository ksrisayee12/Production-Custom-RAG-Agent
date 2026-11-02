import datetime
import logging
import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI
import google.generativeai as genai
import inngest
import inngest.fast_api

from custom_types import (
    RAGChunkAndSrc,
    RAGQueryResult,
    RAGSearchResult,
    RAGUpsertResult,
)
from data_loader import embed_text, load_and_chunk_pdf
from vector_db import QdrantStorage

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

app = FastAPI()

inngest_client = inngest.Inngest(
    app_id="Prod_RAG_App",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):

  def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
    pdf_path = ctx.event.data["pdf_path"]
    source_id = ctx.event.data.get("source_id", pdf_path)
    chunks = load_and_chunk_pdf(pdf_path)
    return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

  def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
    chunks = chunks_and_src.chunks
    source_id = chunks_and_src.source_id

    vectors = embed_text(chunks, task_type="retrieval_document")

    ids = [
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
        for i in range(len(chunks))
    ]

    payloads = [
        {"source": source_id, "text": chunks[i]} for i in range(len(chunks))
    ]

    QdrantStorage().upsert(ids, vectors, payloads)
    return RAGUpsertResult(ingested=len(chunks))

  chunks_and_src = await ctx.step.run(
      "load and chunk",
      lambda: _load(ctx),
      output_type=RAGChunkAndSrc,
  )

  ingested = await ctx.step.run(
      "embed and upsert",
      lambda: _upsert(chunks_and_src),
      output_type=RAGUpsertResult,
  )

  return ingested.model_dump()


@inngest_client.create_function(
    fn_id="rag_query_pdf_ai",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
  question = ctx.event.data["question"]
  top_k = int(ctx.event.data.get("top_k", 5))

  # >>>>>>>> CHANGED: source_id / pdf_path extract panrom + redundant embedding line remove panniyirukkom >>>>>>>>
  source_id = ctx.event.data.get("source_id") or ctx.event.data.get("pdf_path")

  def _search(
      question: str, top_k: int = 5, source_id: str | None = None
  ) -> RAGSearchResult:
    query_vec = embed_text([question], task_type="retrieval_query")[0]
    store = QdrantStorage()
    contexts, sources = store.search(query_vec, top_k, source_id=source_id)
    return RAGSearchResult(contexts=contexts, sources=sources)

  found = await ctx.step.run(
      "embed-and-search",
      lambda: _search(question, top_k, source_id),
      output_type=RAGSearchResult,
  )
  # <<<<<<<< END CHANGED <<<<<<<<

  context_block = "\n\n".join(f"- {c}" for c in found.contexts)
  user_content = (
      "Use the following context to answer the question.\n\n"
      f"Context:\n{context_block}\n\n"
      f"Question: {question}\n"
      "Answer concisely using the context above."
  )

  def _generate_answer(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    config = genai.types.GenerationConfig(
        max_output_tokens=2048,
        temperature=0.2,
    )
    response = model.generate_content(prompt, generation_config=config)
    try:
      return response.text
    except Exception:
      if response.candidates and response.candidates[0].content.parts:
        return response.candidates[0].content.parts[0].text
      return "Unable to generate an answer based on the provided context."

  answer = await ctx.step.run(
      "llm-answer",
      lambda: _generate_answer(user_content),
  )

  return RAGQueryResult(
      answer=answer,
      sources=found.sources,
      num_contexts=len(found.contexts),
  ).model_dump()


inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])