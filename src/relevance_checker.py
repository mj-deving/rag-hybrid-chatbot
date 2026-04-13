"""Corrective RAG (CRAG) — post-retrieval relevance checking and filtering."""

import json
from dataclasses import dataclass

from src.llm_client import get_client, MODEL
from src.vector_store import SearchResult

CONFIDENCE_THRESHOLD = 0.3

RELEVANCE_PROMPT = """Bewerte die Relevanz jedes der folgenden Text-Chunks für die gegebene Frage.

Für jeden Chunk: Ist er relevant für die Beantwortung der Frage? Gib eine Confidence von 0.0 bis 1.0 an.

Antworte NUR mit validem JSON in diesem Format:
{"scores": [{"index": 0, "relevant": true/false, "confidence": 0.0-1.0}, ...]}

Bewerte streng — nur Chunks die tatsächlich zur Beantwortung beitragen sind "relevant"."""


@dataclass
class RetrievalQuality:
    chunks_retrieved: int
    chunks_relevant: int
    chunks_filtered: int
    fallback_triggered: bool


@dataclass
class RelevanceResult:
    relevant_results: list[SearchResult]
    quality: RetrievalQuality


def check_relevance(question: str, results: list[SearchResult]) -> RelevanceResult:
    """Check relevance of retrieved chunks and filter low-confidence ones."""
    if not results:
        return RelevanceResult(
            relevant_results=[],
            quality=RetrievalQuality(
                chunks_retrieved=0,
                chunks_relevant=0,
                chunks_filtered=0,
                fallback_triggered=True,
            ),
        )

    client = get_client()

    chunks_text = "\n\n".join(
        f"[Chunk {i}]: {r.text}" for i, r in enumerate(results)
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": RELEVANCE_PROMPT},
            {"role": "user", "content": f"Frage: {question}\n\nChunks:\n{chunks_text}"},
        ],
    )

    raw = response.choices[0].message.content or ""

    try:
        data = json.loads(raw)
        scores = data.get("scores", [])
        relevant = []
        for score_entry in scores:
            idx = score_entry.get("index", -1)
            confidence = score_entry.get("confidence", 0.0)
            if 0 <= idx < len(results) and confidence >= CONFIDENCE_THRESHOLD:
                relevant.append(results[idx])
    except (json.JSONDecodeError, KeyError, TypeError):
        relevant = list(results)

    filtered_count = len(results) - len(relevant)
    return RelevanceResult(
        relevant_results=relevant,
        quality=RetrievalQuality(
            chunks_retrieved=len(results),
            chunks_relevant=len(relevant),
            chunks_filtered=filtered_count,
            fallback_triggered=len(relevant) == 0,
        ),
    )
