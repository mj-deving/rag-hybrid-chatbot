"""LLM-based entity and relation extraction from document text."""

import json
import logging

from src.llm_client import get_client, MODEL_FAST
from src.knowledge_graph import Entity, Relation

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extrahiere Entitäten und Relationen aus dem folgenden Text.

Entitäten sind: Personen, Organisationen, Orte, Technologien, Konzepte, Produkte, Ereignisse.
Relationen beschreiben Beziehungen zwischen Entitäten (z.B. "arbeitet_bei", "verwendet", "befindet_sich_in", "gehoert_zu", "entwickelt", "konkurriert_mit").

Antworte NUR mit validem JSON in diesem Format:
{
  "entities": [
    {"name": "Entitätsname", "type": "person|organization|location|technology|concept|product|event"}
  ],
  "relations": [
    {"subject": "Entität A", "predicate": "relation_type", "object": "Entität B"}
  ]
}

Regeln:
- Maximal 20 Entitäten und 20 Relationen pro Text
- Entitätsnamen normalisieren (z.B. "BRD" -> "Bundesrepublik Deutschland")
- Nur klar im Text beschriebene Relationen extrahieren, nicht spekulieren
- Relationen als snake_case schreiben (z.B. "arbeitet_bei", "teil_von")"""


def extract_entities_and_relations(
    text: str, document_id: str
) -> tuple[list[Entity], list[Relation]]:
    """Extract entities and relations from text via LLM.

    Returns (entities, relations). On failure, returns empty lists.
    """
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=MODEL_FAST,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": text[:4000]},  # Truncate to fit context
            ],
        )

        raw = response.choices[0].message.content or ""
        data = json.loads(raw)

        entities = [
            Entity(
                name=e["name"],
                type=e.get("type", "concept"),
                document_id=document_id,
            )
            for e in data.get("entities", [])
            if "name" in e
        ]

        relations = [
            Relation(
                subject=r["subject"],
                predicate=r.get("predicate", "related_to"),
                object=r["object"],
                document_id=document_id,
            )
            for r in data.get("relations", [])
            if "subject" in r and "object" in r
        ]

        return entities, relations

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Entity extraction parse error: %s", exc)
        return [], []
    except Exception as exc:
        logger.warning("Entity extraction failed: %s", exc)
        return [], []
