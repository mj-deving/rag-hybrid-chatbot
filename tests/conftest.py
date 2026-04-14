"""Shared test fixtures."""

import shutil
import tempfile

import pytest

import src.vector_store as vs
import src.knowledge_graph as kg


@pytest.fixture(autouse=True)
def isolated_qdrant(tmp_path, monkeypatch):
    """Give each test its own temp Qdrant directory and reset the client."""
    qdrant_dir = tmp_path / "qdrant"
    monkeypatch.setattr(vs, "QDRANT_PATH", str(qdrant_dir))
    vs._client = None
    yield
    if vs._client is not None:
        vs._client.close()
    vs._client = None


@pytest.fixture(autouse=True)
def isolated_graph(tmp_path, monkeypatch):
    """Give each test its own temp graph file and reset the singleton."""
    graph_path = str(tmp_path / "graph.json")
    monkeypatch.setattr(kg, "GRAPH_PATH", graph_path)
    kg._graph = None
    yield
    kg._graph = None
