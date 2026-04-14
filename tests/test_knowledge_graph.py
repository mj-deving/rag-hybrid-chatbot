"""Tests for knowledge graph module."""

import json
import pytest

from src.knowledge_graph import KnowledgeGraph, Entity, Relation, get_graph


class TestKnowledgeGraph:
    def test_add_entities(self, tmp_path):
        g = KnowledgeGraph(path=str(tmp_path / "g.json"))
        g.add_entities([
            Entity(name="Alice", type="person", document_id="doc1"),
            Entity(name="Acme Corp", type="organization", document_id="doc1"),
        ])
        assert g.node_count == 2

    def test_add_relations_creates_edges(self, tmp_path):
        g = KnowledgeGraph(path=str(tmp_path / "g.json"))
        g.add_entities([
            Entity(name="Alice", type="person", document_id="doc1"),
            Entity(name="Acme Corp", type="organization", document_id="doc1"),
        ])
        g.add_relations([
            Relation(subject="Alice", predicate="arbeitet_bei", object="Acme Corp", document_id="doc1"),
        ])
        assert g.edge_count == 1

    def test_add_relations_creates_missing_nodes(self, tmp_path):
        g = KnowledgeGraph(path=str(tmp_path / "g.json"))
        g.add_relations([
            Relation(subject="Bob", predicate="kennt", object="Carol", document_id="doc1"),
        ])
        assert g.node_count == 2
        assert g.edge_count == 1

    def test_query_entity_returns_neighbors(self, tmp_path):
        g = KnowledgeGraph(path=str(tmp_path / "g.json"))
        g.add_entities([
            Entity(name="Alice", type="person", document_id="doc1"),
            Entity(name="Acme Corp", type="organization", document_id="doc1"),
            Entity(name="Project X", type="product", document_id="doc1"),
        ])
        g.add_relations([
            Relation(subject="Alice", predicate="arbeitet_bei", object="Acme Corp", document_id="doc1"),
            Relation(subject="Alice", predicate="leitet", object="Project X", document_id="doc1"),
        ])
        result = g.query_entity("Alice")
        assert result is not None
        assert result["name"] == "Alice"
        assert len(result["neighbors"]) == 2

    def test_query_entity_not_found(self, tmp_path):
        g = KnowledgeGraph(path=str(tmp_path / "g.json"))
        assert g.query_entity("Nobody") is None

    def test_query_relations_finds_path(self, tmp_path):
        g = KnowledgeGraph(path=str(tmp_path / "g.json"))
        g.add_relations([
            Relation(subject="Alice", predicate="arbeitet_bei", object="Acme Corp", document_id="doc1"),
            Relation(subject="Bob", predicate="arbeitet_bei", object="Acme Corp", document_id="doc1"),
        ])
        paths = g.query_relations("Alice", "Bob")
        assert len(paths) >= 1
        # Path goes Alice -> Acme Corp -> Bob
        assert any(len(p) == 2 for p in paths)

    def test_query_relations_no_connection(self, tmp_path):
        g = KnowledgeGraph(path=str(tmp_path / "g.json"))
        g.add_entities([
            Entity(name="Alice", type="person", document_id="doc1"),
            Entity(name="Bob", type="person", document_id="doc2"),
        ])
        paths = g.query_relations("Alice", "Bob")
        assert paths == []

    def test_get_subgraph(self, tmp_path):
        g = KnowledgeGraph(path=str(tmp_path / "g.json"))
        g.add_entities([
            Entity(name="Alice", type="person", document_id="doc1"),
            Entity(name="Bob", type="person", document_id="doc2"),
        ])
        g.add_relations([
            Relation(subject="Alice", predicate="kennt", object="Bob", document_id="doc1"),
        ])
        sub = g.get_subgraph("doc1")
        assert len(sub["entities"]) == 1
        assert sub["entities"][0]["name"] == "Alice"
        assert len(sub["relations"]) == 1

    def test_find_relevant_entities_exact(self, tmp_path):
        g = KnowledgeGraph(path=str(tmp_path / "g.json"))
        g.add_entities([Entity(name="Berlin", type="location", document_id="doc1")])
        results = g.find_relevant_entities(["Berlin"])
        assert len(results) == 1
        assert results[0]["name"] == "Berlin"

    def test_find_relevant_entities_substring(self, tmp_path):
        g = KnowledgeGraph(path=str(tmp_path / "g.json"))
        g.add_entities([Entity(name="Bundesrepublik Deutschland", type="location", document_id="doc1")])
        results = g.find_relevant_entities(["Deutschland"])
        assert len(results) == 1

    def test_delete_document(self, tmp_path):
        g = KnowledgeGraph(path=str(tmp_path / "g.json"))
        g.add_entities([
            Entity(name="Alice", type="person", document_id="doc1"),
            Entity(name="Bob", type="person", document_id="doc2"),
        ])
        g.add_relations([
            Relation(subject="Alice", predicate="kennt", object="Bob", document_id="doc1"),
        ])
        g.delete_document("doc1")
        assert g.node_count == 1  # Only Bob remains
        assert g.edge_count == 0


class TestGraphPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "g.json")
        g1 = KnowledgeGraph(path=path)
        g1.add_entities([Entity(name="Alice", type="person", document_id="doc1")])
        g1.add_relations([
            Relation(subject="Alice", predicate="kennt", object="Bob", document_id="doc1"),
        ])

        # Load into new instance
        g2 = KnowledgeGraph(path=path)
        assert g2.node_count == 2
        assert g2.edge_count == 1
        result = g2.query_entity("Alice")
        assert result is not None
        assert len(result["neighbors"]) == 1

    def test_load_nonexistent_file(self, tmp_path):
        g = KnowledgeGraph(path=str(tmp_path / "nonexistent.json"))
        assert g.node_count == 0


class TestGetGraphSingleton:
    def test_returns_same_instance(self):
        g1 = get_graph()
        g2 = get_graph()
        assert g1 is g2
