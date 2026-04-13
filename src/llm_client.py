"""Shared OpenRouter/OpenAI client singleton and constants."""

import os

from openai import OpenAI

MODEL = "anthropic/claude-sonnet-4"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

ROUTE_SIMPLE = "simple"
ROUTE_STANDARD = "standard"
ROUTE_COMPLEX = "complex"
VALID_ROUTES = {ROUTE_SIMPLE, ROUTE_STANDARD, ROUTE_COMPLEX}

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Get or initialize the shared OpenRouter client (singleton)."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )
    return _client
