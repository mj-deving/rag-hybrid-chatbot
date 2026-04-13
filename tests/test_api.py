"""Tests for FastAPI endpoints."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


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
    @patch("src.rag_engine.get_client")
    @patch("src.relevance_checker.get_client")
    @patch("src.query_classifier.get_client")
    def test_query_with_mock_llm(self, mock_cls_client, mock_rel_client, mock_eng_client):
        import json as _json

        client.post(
            "/upload",
            files={"file": ("test.md", b"# AI\n\nKI ist toll.", "text/markdown")},
        )

        def _resp(content, with_usage=False):
            r = MagicMock()
            r.choices[0].message.content = content
            if with_usage:
                r.usage.prompt_tokens = 100
                r.usage.completion_tokens = 50
            else:
                r.usage = None
            return r

        cls_client = MagicMock()
        cls_client.chat.completions.create.return_value = _resp(
            _json.dumps({"route": "standard", "sub_queries": []})
        )
        mock_cls_client.return_value = cls_client

        rel_client = MagicMock()
        rel_client.chat.completions.create.return_value = _resp(
            _json.dumps({"scores": [{"index": 0, "relevant": True, "confidence": 0.9}]})
        )
        mock_rel_client.return_value = rel_client

        eng_client = MagicMock()
        eng_client.chat.completions.create.return_value = _resp(
            "KI ist eine Technologie.", with_usage=True
        )
        mock_eng_client.return_value = eng_client

        response = client.post("/query", json={"question": "Was ist KI?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "tokens_used" in data
        assert "routing" in data
        assert data["routing"]["route"] == "standard"
        assert "retrieval_quality" in data


class TestSwaggerDocs:
    def test_docs_accessible(self):
        response = client.get("/docs")
        assert response.status_code == 200


class TestAuth:
    def test_no_auth_when_key_not_set(self, monkeypatch):
        """Without RAG_API_KEY, all endpoints are open."""
        import src.api as api_mod
        monkeypatch.setattr(api_mod, "API_KEY", "")
        response = client.get("/documents")
        assert response.status_code == 200

    def test_rejects_without_token(self, monkeypatch):
        """With RAG_API_KEY set, requests without token get 401."""
        import src.api as api_mod
        monkeypatch.setattr(api_mod, "API_KEY", "test-secret-key")
        response = client.get("/documents")
        assert response.status_code == 401

    def test_accepts_valid_token(self, monkeypatch):
        """With RAG_API_KEY set, requests with correct token succeed."""
        import src.api as api_mod
        monkeypatch.setattr(api_mod, "API_KEY", "test-secret-key")
        response = client.get(
            "/documents",
            headers={"Authorization": "Bearer test-secret-key"},
        )
        assert response.status_code == 200

    def test_rejects_wrong_token(self, monkeypatch):
        """With RAG_API_KEY set, requests with wrong token get 401."""
        import src.api as api_mod
        monkeypatch.setattr(api_mod, "API_KEY", "test-secret-key")
        response = client.get(
            "/documents",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert response.status_code == 401

    def test_root_accessible_without_auth(self, monkeypatch):
        """Static file serving works even with auth enabled."""
        import src.api as api_mod
        monkeypatch.setattr(api_mod, "API_KEY", "test-secret-key")
        response = client.get("/")
        assert response.status_code == 200
