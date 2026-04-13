"""Adaptive RAG query classifier — routes queries to simple/standard/complex paths."""

import json
from dataclasses import dataclass, field

from src.llm_client import (
    get_client, MODEL_FAST, ROUTE_SIMPLE, ROUTE_STANDARD, ROUTE_COMPLEX, VALID_ROUTES,
)

CLASSIFIER_PROMPT = """Klassifiziere die folgende Frage in eine von drei Kategorien:

- "simple": Allgemeine Wissensfragen, die ohne Dokumenten-Kontext beantwortet werden können (z.B. "Was ist RAG?", "Erkläre Machine Learning")
- "standard": Fragen, die sich auf spezifische Dokumente oder Inhalte beziehen (z.B. "Was sagt Willison über November 2025?", "Welche Workflows werden erwähnt?")
- "complex": Vergleichende oder multi-dimensionale Fragen, die mehrere Aspekte oder Quellen kombinieren (z.B. "Vergleiche die RAG-Architekturen von Karpathy und LightRAG und bewerte ihre Eignung")

Für "complex" Fragen: Zerlege die Frage in 2-3 einfachere Sub-Queries, die jeweils separat beantwortet werden können.

Antworte NUR mit validem JSON in diesem Format:
{"route": "simple|standard|complex", "sub_queries": ["...", "..."]}

Bei "simple" und "standard": sub_queries ist ein leeres Array [].
Bei "complex": sub_queries enthält 2-3 zerlegte Teilfragen."""


@dataclass
class ClassificationResult:
    route: str
    sub_queries: list[str] = field(default_factory=list)


def classify_query(question: str) -> ClassificationResult:
    """Classify a query into simple/standard/complex route."""
    client = get_client()

    response = client.chat.completions.create(
        model=MODEL_FAST,
        max_tokens=256,
        messages=[
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": question},
        ],
    )

    raw = response.choices[0].message.content or ""

    try:
        data = json.loads(raw)
        route = data.get("route", ROUTE_STANDARD)
        if route not in VALID_ROUTES:
            route = ROUTE_STANDARD
        sub_queries = data.get("sub_queries", []) if route == ROUTE_COMPLEX else []
        return ClassificationResult(route=route, sub_queries=sub_queries)
    except (json.JSONDecodeError, KeyError, TypeError):
        return ClassificationResult(route=ROUTE_STANDARD, sub_queries=[])
