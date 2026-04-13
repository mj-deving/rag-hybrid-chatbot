"""Shared test fixtures."""

import shutil
import tempfile

import pytest

import src.vector_store as vs


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
