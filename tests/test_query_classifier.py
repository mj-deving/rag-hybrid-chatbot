"""Tests for adaptive RAG query classifier."""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.query_classifier import classify_query, ClassificationResult


def _mock_llm_response(content: str):
    """Create a mock OpenAI chat completion response."""
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


def _mock_client(content: str) -> MagicMock:
    """Create a mock OpenAI client that returns the given content."""
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_llm_response(content)
    return client


class TestClassifyQuery:
    @patch("src.query_classifier.get_client")
    def test_simple_route(self, mock_get_client):
        mock_get_client.return_value = _mock_client(
            json.dumps({"route": "simple", "sub_queries": []})
        )
        result = classify_query("Was ist RAG?")
        assert result.route == "simple"
        assert result.sub_queries == []

    @patch("src.query_classifier.get_client")
    def test_standard_route(self, mock_get_client):
        mock_get_client.return_value = _mock_client(
            json.dumps({"route": "standard", "sub_queries": []})
        )
        result = classify_query("Was sagt Willison über November 2025?")
        assert result.route == "standard"
        assert result.sub_queries == []

    @patch("src.query_classifier.get_client")
    def test_complex_route_with_sub_queries(self, mock_get_client):
        mock_get_client.return_value = _mock_client(
            json.dumps({
                "route": "complex",
                "sub_queries": [
                    "Was ist Karpathys RAG-Architektur?",
                    "Was ist LightRAG?",
                    "Vergleich der Eignung für den Mittelstand",
                ],
            })
        )
        result = classify_query(
            "Vergleiche die RAG-Architekturen von Karpathy und LightRAG"
        )
        assert result.route == "complex"
        assert len(result.sub_queries) == 3

    @patch("src.query_classifier.get_client")
    def test_empty_sub_queries_for_simple(self, mock_get_client):
        mock_get_client.return_value = _mock_client(
            json.dumps({"route": "simple", "sub_queries": []})
        )
        result = classify_query("Hallo")
        assert result.sub_queries == []

    @patch("src.query_classifier.get_client")
    def test_fallback_on_malformed_response(self, mock_get_client):
        """If LLM returns garbage, fall back to standard route."""
        mock_get_client.return_value = _mock_client("this is not json")
        result = classify_query("Some question")
        assert result.route == "standard"
        assert result.sub_queries == []
