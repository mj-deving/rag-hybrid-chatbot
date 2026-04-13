"""RAG query engine: classify → route → retrieve → check relevance → generate."""

from dataclasses import dataclass

from src.llm_client import get_client, MODEL, ROUTE_SIMPLE, ROUTE_COMPLEX
from src.document_processor import embed_query
from src.vector_store import search, SearchResult
from src.query_classifier import classify_query, ClassificationResult
from src.relevance_checker import check_relevance, RetrievalQuality

SYSTEM_PROMPT = """Du bist ein hilfreicher Assistent, der Fragen basierend auf bereitgestelltem Kontext beantwortet.
Regeln:
- Antworte nur basierend auf dem bereitgestellten Kontext
- Wenn der Kontext die Frage nicht beantwortet, sage das ehrlich
- Nenne immer die Quellen (Dokumentname und Chunk-Nummer) für deine Aussagen
- Antworte auf Deutsch, es sei denn die Frage ist auf Englisch"""

SIMPLE_SYSTEM_PROMPT = """Du bist ein hilfreicher Assistent, der allgemeine Wissensfragen beantwortet.
Regeln:
- Antworte kurz und präzise
- Antworte auf Deutsch, es sei denn die Frage ist auf Englisch"""

FALLBACK_ANSWER = (
    "Zu dieser Frage habe ich keine relevanten Dokumente gefunden. "
    "Möglicherweise müssen weitere Dokumente hochgeladen werden."
)


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
    routing: ClassificationResult
    retrieval_quality: RetrievalQuality | None = None


def build_context(results: list[SearchResult]) -> str:
    """Build context string from search results."""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Quelle {i}: {r.source}, Chunk {r.chunk_index}, Relevanz: {r.score:.2f}]\n{r.text}"
        )
    return "\n\n---\n\n".join(parts)


def _generate_answer(context: str, question: str, system_prompt: str) -> tuple[str, int]:
    """Call LLM and return (answer, tokens_used)."""
    client = get_client()
    user_message = f"""Kontext:
{context}

Frage: {question}

Beantworte die Frage basierend auf dem obigen Kontext. Nenne die Quellen."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    answer = response.choices[0].message.content or ""
    tokens = (response.usage.prompt_tokens + response.usage.completion_tokens) if response.usage else 0
    return answer, tokens


def _generate_simple_answer(question: str) -> tuple[str, int]:
    """Direct LLM answer without retrieval context."""
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": SIMPLE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    answer = response.choices[0].message.content or ""
    tokens = (response.usage.prompt_tokens + response.usage.completion_tokens) if response.usage else 0
    return answer, tokens


def _retrieve_and_filter(question: str, top_k: int) -> tuple[list[SearchResult], RetrievalQuality]:
    """Retrieve chunks and apply CRAG relevance filtering."""
    question_embedding = embed_query(question)
    results = search(question_embedding, top_k=top_k)
    relevance = check_relevance(question, results)
    return relevance.relevant_results, relevance.quality


def _build_result(
    answer: str, tokens: int, routing: ClassificationResult,
    results: list[SearchResult], quality: RetrievalQuality | None,
) -> QueryResult:
    sources = [
        Source(document=r.source, chunk=r.chunk_index, relevance=round(r.score, 4))
        for r in results
    ]
    return QueryResult(
        answer=answer, sources=sources, tokens_used=tokens,
        routing=routing, retrieval_quality=quality,
    )


def query(question: str, top_k: int = 5) -> QueryResult:
    """Full adaptive RAG pipeline: classify → route → retrieve → filter → generate."""
    classification = classify_query(question)

    if classification.route == ROUTE_SIMPLE:
        answer, tokens = _generate_simple_answer(question)
        return _build_result(answer, tokens, classification, [], None)

    if classification.route == ROUTE_COMPLEX and classification.sub_queries:
        all_results: list[SearchResult] = []
        seen: set[tuple[str, int]] = set()
        total_retrieved = 0
        total_filtered = 0

        for sub_q in classification.sub_queries:
            relevant, quality = _retrieve_and_filter(sub_q, top_k=top_k)
            total_retrieved += quality.chunks_retrieved
            total_filtered += quality.chunks_filtered
            for r in relevant:
                key = (r.source, r.chunk_index)
                if key not in seen:
                    seen.add(key)
                    all_results.append(r)

        merged_quality = RetrievalQuality(
            chunks_retrieved=total_retrieved,
            chunks_relevant=len(all_results),
            chunks_filtered=total_filtered,
            fallback_triggered=len(all_results) == 0,
        )

        if not all_results:
            return _build_result(FALLBACK_ANSWER, 0, classification, [], merged_quality)

        context = build_context(all_results)
        answer, tokens = _generate_answer(context, question, SYSTEM_PROMPT)
        return _build_result(answer, tokens, classification, all_results, merged_quality)

    relevant_results, retrieval_quality = _retrieve_and_filter(question, top_k=top_k)

    if not relevant_results:
        return _build_result(FALLBACK_ANSWER, 0, classification, [], retrieval_quality)

    context = build_context(relevant_results)
    answer, tokens = _generate_answer(context, question, SYSTEM_PROMPT)
    return _build_result(answer, tokens, classification, relevant_results, retrieval_quality)
