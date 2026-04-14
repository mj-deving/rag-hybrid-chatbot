"""Tests for entity extraction module."""

import json
from unittest.mock import patch, MagicMock

from src.entity_extractor import extract_entities_and_relations


def _mock_client(content: str) -> MagicMock:
    client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    client.chat.completions.create.return_value = mock_response
    return client


class TestExtractEntitiesAndRelations:
    @patch("src.entity_extractor.get_client")
    def test_extracts_entities_and_relations(self, mock_get_client):
        mock_get_client.return_value = _mock_client(json.dumps({
            "entities": [
                {"name": "Alice", "type": "person"},
                {"name": "Acme Corp", "type": "organization"},
            ],
            "relations": [
                {"subject": "Alice", "predicate": "arbeitet_bei", "object": "Acme Corp"},
            ],
        }))
        entities, relations = extract_entities_and_relations("Alice works at Acme Corp.", "doc1")
        assert len(entities) == 2
        assert entities[0].name == "Alice"
        assert entities[0].document_id == "doc1"
        assert len(relations) == 1
        assert relations[0].predicate == "arbeitet_bei"

    @patch("src.entity_extractor.get_client")
    def test_handles_malformed_json(self, mock_get_client):
        mock_get_client.return_value = _mock_client("not valid json")
        entities, relations = extract_entities_and_relations("some text", "doc1")
        assert entities == []
        assert relations == []

    @patch("src.entity_extractor.get_client")
    def test_handles_empty_response(self, mock_get_client):
        mock_get_client.return_value = _mock_client(json.dumps({
            "entities": [],
            "relations": [],
        }))
        entities, relations = extract_entities_and_relations("text", "doc1")
        assert entities == []
        assert relations == []

    @patch("src.entity_extractor.get_client")
    def test_handles_missing_fields(self, mock_get_client):
        """Entities missing 'name' field are skipped."""
        mock_get_client.return_value = _mock_client(json.dumps({
            "entities": [
                {"type": "person"},  # no name
                {"name": "Valid", "type": "concept"},
            ],
            "relations": [
                {"subject": "A"},  # no object
                {"subject": "X", "object": "Y"},  # valid
            ],
        }))
        entities, relations = extract_entities_and_relations("text", "doc1")
        assert len(entities) == 1
        assert entities[0].name == "Valid"
        assert len(relations) == 1

    @patch("src.entity_extractor.get_client")
    def test_handles_api_error(self, mock_get_client):
        mock_get_client.side_effect = RuntimeError("API down")
        entities, relations = extract_entities_and_relations("text", "doc1")
        assert entities == []
        assert relations == []
