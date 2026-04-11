"""RAG query engine: embed question, retrieve context, call LLM via OpenRouter."""

import os
from dataclasses import dataclass

from openai import OpenAI

from src.document_processor import embed_query
from src.vector_store import search, SearchResult

_client: OpenAI | None = None
MODEL = "anthropic/claude-sonnet-4"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = """Du bist ein hilfreicher Assistent, der Fragen basierend auf bereitgestelltem Kontext beantwortet.
Regeln:
- Antworte nur basierend auf dem bereitgestellten Kontext
- Wenn der Kontext die Frage nicht beantwortet, sage das ehrlich
- Nenne immer die Quellen (Dokumentname und Chunk-Nummer) für deine Aussagen
- Antworte auf Deutsch, es sei denn die Frage ist auf Englisch"""


@dataclass
class Source:
    document: str
    chunk: int
    relevance: float


@dataclass
class QueryResult:
    answer: str
    sources: list[Source]
    tokens_used: int


def build_context(results: list[SearchResult]) -> str:
    """Build context string from search results."""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Quelle {i}: {r.source}, Chunk {r.chunk_index}, Relevanz: {r.score:.2f}]\n{r.text}"
        )
    return "\n\n---\n\n".join(parts)


def query(question: str, top_k: int = 5) -> QueryResult:
    """Full RAG pipeline: embed question → retrieve → generate answer."""
    # 1. Embed the question
    question_embedding = embed_query(question)

    # 2. Retrieve relevant chunks
    results = search(question_embedding, top_k=top_k)

    if not results:
        return QueryResult(
            answer="Keine relevanten Dokumente gefunden. Bitte laden Sie zuerst Dokumente hoch.",
            sources=[],
            tokens_used=0,
        )

    # 3. Build prompt with context
    context = build_context(results)
    user_message = f"""Kontext:
{context}

Frage: {question}

Beantworte die Frage basierend auf dem obigen Kontext. Nenne die Quellen."""

    # 4. Call LLM via OpenRouter
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )

    response = _client.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    answer = response.choices[0].message.content or ""
    tokens_used = (response.usage.prompt_tokens + response.usage.completion_tokens) if response.usage else 0

    # 5. Build sources
    sources = [
        Source(document=r.source, chunk=r.chunk_index, relevance=round(r.score, 4))
        for r in results
    ]

    return QueryResult(answer=answer, sources=sources, tokens_used=tokens_used)
