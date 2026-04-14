"""Shared OpenRouter/OpenAI client singleton and constants."""

import os

from openai import OpenAI

MODEL = "anthropic/claude-sonnet-4"
MODEL_FAST = "anthropic/claude-3.5-haiku"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

ROUTE_SIMPLE = "simple"
ROUTE_STANDARD = "standard"
ROUTE_COMPLEX = "complex"
ROUTE_RELATIONAL = "relational"
VALID_ROUTES = {ROUTE_SIMPLE, ROUTE_STANDARD, ROUTE_COMPLEX, ROUTE_RELATIONAL}

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
