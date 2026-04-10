"""Basic tests."""

from src.main import load_env


def test_load_env_does_not_crash():
    """Smoke test — load_env runs without error."""
    load_env()
