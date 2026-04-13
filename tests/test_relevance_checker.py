"""Tests for CRAG relevance checker."""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.relevance_checker import check_relevance, RelevanceResult
from src.vector_store import SearchResult


def _make_results(n: int) -> list[SearchResult]:
    """Create test search results."""
    return [
        SearchResult(
            text=f"Chunk {i} about topic X",
            source=f"doc{i}.md",
            chunk_index=i,
            score=0.9 - (i * 0.1),
        )
        for i in range(n)
    ]


def _mock_llm_response(content: str):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


def _mock_client(content: str) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_llm_response(content)
    return client


class TestCheckRelevance:
    @patch("src.relevance_checker.get_client")
    def test_keeps_relevant_chunks(self, mock_get_client):
        """Chunks with confidence >= 0.3 are kept."""
        results = _make_results(3)
        mock_get_client.return_value = _mock_client(json.dumps({
            "scores": [
                {"index": 0, "relevant": True, "confidence": 0.9},
                {"index": 1, "relevant": True, "confidence": 0.7},
                {"index": 2, "relevant": True, "confidence": 0.5},
            ]
        }))
        result = check_relevance("What is X?", results)
        assert len(result.relevant_results) == 3
        assert result.quality.chunks_relevant == 3
        assert result.quality.chunks_filtered == 0
        assert result.quality.fallback_triggered is False

    @patch("src.relevance_checker.get_client")
    def test_filters_low_confidence_chunks(self, mock_get_client):
        """Chunks with confidence < 0.3 are filtered out."""
        results = _make_results(5)
        mock_get_client.return_value = _mock_client(json.dumps({
            "scores": [
                {"index": 0, "relevant": True, "confidence": 0.9},
                {"index": 1, "relevant": True, "confidence": 0.6},
                {"index": 2, "relevant": False, "confidence": 0.2},
                {"index": 3, "relevant": False, "confidence": 0.1},
                {"index": 4, "relevant": False, "confidence": 0.05},
            ]
        }))
        result = check_relevance("What is X?", results)
        assert len(result.relevant_results) == 2
        assert result.quality.chunks_retrieved == 5
        assert result.quality.chunks_relevant == 2
        assert result.quality.chunks_filtered == 3
        assert result.quality.fallback_triggered is False

    @patch("src.relevance_checker.get_client")
    def test_fallback_when_all_filtered(self, mock_get_client):
        """When all chunks are irrelevant, fallback_triggered is True."""
        results = _make_results(3)
        mock_get_client.return_value = _mock_client(json.dumps({
            "scores": [
                {"index": 0, "relevant": False, "confidence": 0.1},
                {"index": 1, "relevant": False, "confidence": 0.2},
                {"index": 2, "relevant": False, "confidence": 0.15},
            ]
        }))
        result = check_relevance("Completely unrelated question?", results)
        assert len(result.relevant_results) == 0
        assert result.quality.fallback_triggered is True
        assert result.quality.chunks_relevant == 0

    def test_empty_results_triggers_fallback(self):
        """Empty search results should trigger fallback without LLM call."""
        result = check_relevance("Any question?", [])
        assert result.quality.fallback_triggered is True
        assert result.quality.chunks_retrieved == 0

    @patch("src.relevance_checker.get_client")
    def test_malformed_response_keeps_all(self, mock_get_client):
        """If LLM returns garbage, keep all chunks (fail open)."""
        results = _make_results(3)
        mock_get_client.return_value = _mock_client("not valid json")
        result = check_relevance("What is X?", results)
        assert len(result.relevant_results) == 3
        assert result.quality.chunks_filtered == 0
