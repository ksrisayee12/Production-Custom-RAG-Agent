import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
# >>>>>>>> CHANGED: Filter models import pannirukkom >>>>>>>>
from qdrant_client.models import FieldCondition, Filter, MatchValue
# <<<<<<<< END CHANGED <<<<<<<<

load_dotenv()


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

    if not self.url or not self.api_key:
      raise ValueError("QDRANT_URL or QDRANT_API_KEY missing in .env")

    self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=30)

    if not self.client.collection_exists(self.collection):
      self.client.create_collection(
          collection_name=self.collection,
          vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
      )

  def upsert(
      self, ids: list[str], vectors: list[list[float]], payloads: list[dict]
  ):
    points = [
        PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
        for i in range(len(ids))
    ]
    self.client.upsert(collection_name=self.collection, points=points)

  # >>>>>>>> CHANGED: source_id parameter serthu Qdrant-la payload filter apply pannirukkom >>>>>>>>
  def search(
      self,
      query_vector: list[float],
      top_k: int = 5,
      source_id: str | None = None,
  ):
    query_filter = None
    if source_id:
      query_filter = Filter(
          must=[
              FieldCondition(
                  key="source",
                  match=MatchValue(value=source_id),
              )
          ]
      )

    response = self.client.query_points(
        collection_name=self.collection,
        query=query_vector,
        query_filter=query_filter,  # Specific PDF chunks mattume search aagum
        with_payload=True,
        limit=top_k,
    )
    # <<<<<<<< END CHANGED <<<<<<<<

    contexts = []
    sources = set()

    for r in response.points:
      payload = getattr(r, "payload", None) or {}
      text = payload.get("text", "")
      source = payload.get("source", "")

      if text:
        contexts.append(text)
      if source:
        sources.add(source)

    return contexts, list(sources)