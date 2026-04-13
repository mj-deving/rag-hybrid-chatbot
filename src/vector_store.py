"""Qdrant vector store wrapper — persistent file-based storage."""

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    FilterSelector,
    MatchValue,
)

from src.document_processor import EMBEDDING_DIM, Chunk

COLLECTION_NAME = "documents"
QDRANT_PATH = os.environ.get(
    "QDRANT_PATH", str(Path(__file__).parent.parent / "data" / "qdrant")
)

_client: QdrantClient | None = None


@dataclass
class SearchResult:
    text: str
    source: str
    chunk_index: int
    score: float


@dataclass
class DocumentInfo:
    id: str
    filename: str
    chunks: int
    uploaded: str


def get_client() -> QdrantClient:
    """Get or initialize the Qdrant client with persistent storage."""
    global _client
    if _client is None:
        Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
        _client = QdrantClient(path=QDRANT_PATH)
        if not _client.collection_exists(COLLECTION_NAME):
            _client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
    return _client


def upsert_chunks(chunks: list[Chunk]) -> int:
    """Store chunks with embeddings in Qdrant. Returns count of points stored."""
    client = get_client()

    points = [
        PointStruct(
            id=uuid.uuid4().hex,
            vector=chunk.embedding,
            payload={
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "document_id": chunk.document_id,
            },
        )
        for chunk in chunks
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def search(query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
    """Similarity search against stored vectors."""
    client = get_client()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    return [
        SearchResult(
            text=hit.payload["text"],
            source=hit.payload["source"],
            chunk_index=hit.payload["chunk_index"],
            score=hit.score,
        )
        for hit in results.points
    ]


def list_documents() -> list[DocumentInfo]:
    """List all unique documents in the collection."""
    client = get_client()
    # Scroll through all points and aggregate by document_id
    docs: dict[str, DocumentInfo] = {}

    offset = None
    while True:
        scroll_result = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = scroll_result

        for point in points:
            doc_id = point.payload["document_id"]
            if doc_id not in docs:
                docs[doc_id] = DocumentInfo(
                    id=doc_id,
                    filename=point.payload["source"],
                    chunks=0,
                    uploaded="",
                )
            docs[doc_id].chunks += 1

        if next_offset is None:
            break
        offset = next_offset

    return list(docs.values())


def delete_document(document_id: str) -> int:
    """Delete all chunks belonging to a document. Returns count deleted."""
    client = get_client()

    # Count matching points first
    doc_filter = Filter(
        must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
    )
    count = client.count(collection_name=COLLECTION_NAME, count_filter=doc_filter).count

    if count > 0:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(filter=doc_filter),
        )

    return count
