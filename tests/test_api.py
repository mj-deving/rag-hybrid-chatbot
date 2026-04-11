"""Tests for FastAPI endpoints."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import src.vector_store as vs
from src.api import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_vector_store():
    """Reset Qdrant for each test."""
    vs._client = None
    yield
    vs._client = None


class TestUploadEndpoint:
    def test_upload_markdown(self, tmp_path):
        content = b"# Test\n\nSome content for testing the upload endpoint."
        response = client.post(
            "/upload",
            files={"file": ("test.md", content, "text/markdown")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.md"
        assert data["chunks"] >= 1
        assert data["status"] == "indexed"
        assert "document_id" in data

    def test_upload_unsupported_type(self):
        response = client.post(
            "/upload",
            files={"file": ("test.exe", b"binary", "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_upload_empty_file(self):
        response = client.post(
            "/upload",
            files={"file": ("empty.md", b"", "text/markdown")},
        )
        assert response.status_code == 400


class TestDocumentsEndpoint:
    def test_list_empty(self):
        response = client.get("/documents")
        assert response.status_code == 200
        assert response.json()["documents"] == []

    def test_list_after_upload(self):
        client.post(
            "/upload",
            files={"file": ("doc.md", b"# Hello\n\nContent here.", "text/markdown")},
        )
        response = client.get("/documents")
        docs = response.json()["documents"]
        assert len(docs) == 1
        assert docs[0]["filename"] == "doc.md"


class TestDeleteEndpoint:
    def test_delete_existing(self):
        resp = client.post(
            "/upload",
            files={"file": ("del.md", b"# Delete me\n\nSome text.", "text/markdown")},
        )
        doc_id = resp.json()["document_id"]

        resp = client.delete(f"/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_nonexistent(self):
        resp = client.delete("/documents/nonexistent")
        assert resp.status_code == 404


class TestQueryEndpoint:
    @patch("src.rag_engine.OpenAI")
    def test_query_with_mock_llm(self, mock_openai_cls):
        # Upload a document first
        client.post(
            "/upload",
            files={"file": ("test.md", b"# AI\n\nKI ist toll.", "text/markdown")},
        )

        # Reset singleton so mock is used
        import src.rag_engine as re
        re._client = None

        # Mock OpenRouter/OpenAI response
        mock_choice = MagicMock()
        mock_choice.message.content = "KI ist eine Technologie."
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_openai_cls.return_value.chat.completions.create.return_value = mock_response

        response = client.post("/query", json={"question": "Was ist KI?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "tokens_used" in data


class TestSwaggerDocs:
    def test_docs_accessible(self):
        response = client.get("/docs")
        assert response.status_code == 200
