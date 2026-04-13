"""Tests for vector store module."""

import pytest

from src.document_processor import Chunk, EMBEDDING_DIM
import src.vector_store as vs


def _make_chunks(n: int, doc_id: str = "test123", source: str = "test.md") -> list[Chunk]:
    """Create test chunks with dummy embeddings."""
    return [
        Chunk(
            text=f"Chunk {i} text content",
            source=source,
            chunk_index=i,
            document_id=doc_id,
            embedding=[0.1 * (i + 1)] * EMBEDDING_DIM,
        )
        for i in range(n)
    ]


class TestUpsertAndSearch:
    def test_upsert_stores_chunks(self):
        chunks = _make_chunks(3)
        count = vs.upsert_chunks(chunks)
        assert count == 3

    def test_search_returns_results(self):
        chunks = _make_chunks(3)
        vs.upsert_chunks(chunks)

        query_emb = [0.1] * EMBEDDING_DIM
        results = vs.search(query_emb, top_k=2)
        assert len(results) == 2
        assert results[0].score > 0

    def test_search_empty_collection(self):
        results = vs.search([0.1] * EMBEDDING_DIM, top_k=5)
        assert results == []


class TestListDocuments:
    def test_list_after_upsert(self):
        vs.upsert_chunks(_make_chunks(3, doc_id="doc1", source="a.md"))
        vs.upsert_chunks(_make_chunks(2, doc_id="doc2", source="b.md"))

        docs = vs.list_documents()
        assert len(docs) == 2
        filenames = {d.filename for d in docs}
        assert filenames == {"a.md", "b.md"}

    def test_chunk_counts(self):
        vs.upsert_chunks(_make_chunks(5, doc_id="doc1", source="big.md"))
        docs = vs.list_documents()
        assert docs[0].chunks == 5


class TestDeleteDocument:
    def test_delete_removes_chunks(self):
        vs.upsert_chunks(_make_chunks(3, doc_id="doc1"))
        vs.upsert_chunks(_make_chunks(2, doc_id="doc2"))

        deleted = vs.delete_document("doc1")
        assert deleted == 3

        docs = vs.list_documents()
        assert len(docs) == 1
        assert docs[0].id == "doc2"

    def test_delete_nonexistent_returns_zero(self):
        deleted = vs.delete_document("nonexistent")
        assert deleted == 0
