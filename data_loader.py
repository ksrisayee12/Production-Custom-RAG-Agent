import os
from dotenv import load_dotenv
import google.generativeai as genai
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str) -> list[str]:
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_text(texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
    if not texts:
        return []

    embeddings: list[list[float]] = []
    batch_size = 50

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = genai.embed_content(
            model=EMBED_MODEL,
            content=batch,
            task_type=task_type,
        )

        batch_embeddings = response["embedding"]

        if isinstance(batch_embeddings[0], list):
            embeddings.extend(batch_embeddings)
        else:
            embeddings.append(batch_embeddings)

    return embeddings