# RAG Chatbot

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**Dokumente hochladen, indexieren und per Chat Fragen mit kontextbasierten Antworten stellen.**

RAG (Retrieval-Augmented Generation) Chatbot mit FastAPI, Qdrant Vector DB und Claude via OpenRouter. Lokale Embeddings ohne externe API-Abhaengigkeit.

![RAG Chatbot Screenshot](docs/screenshot.png)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/mj-deving/rag-prototype.git
cd rag-prototype

# 2. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. API Key konfigurieren
echo 'OPENROUTER_API_KEY=sk-or-v1-...' >> ~/.claude/.env
# Oder: export OPENROUTER_API_KEY=sk-or-v1-...

# 4. Server starten
python src/main.py
# -> http://localhost:8000
```

## Features

- **Dokument-Upload** -- PDF, Markdown, TXT per Drag-and-Drop oder API
- **Automatisches Chunking** -- Rekursives Splitting (~500 Tokens, 50 Overlap)
- **Lokale Embeddings** -- fastembed (BAAI/bge-small-en-v1.5, 384-dim, ONNX) -- kein API Key noetig
- **Vector Search** -- Qdrant In-Memory mit Cosine Similarity
- **Adaptive RAG** -- Query-Routing: einfache Fragen direkt beantworten, dokumentspezifische via Vector Search, komplexe via Multi-Query-Decomposition
- **Corrective RAG (CRAG)** -- Post-Retrieval Relevanz-Check filtert irrelevante Chunks, Fallback bei fehlender Relevanz
- **LLM-Antworten** -- Claude via OpenRouter mit Quellenangaben
- **Chat UI** -- Single-Page HTML mit Dark Theme, responsive, zeigt Route und Retrieval-Qualitaet
- **REST API** -- 4 Endpoints mit Swagger UI unter `/docs`

## Architektur

```
Browser (localhost:8000)
    |
    v
FastAPI (src/api.py)
    |
    +-- POST /upload --> document_processor.py
    |                      extract_text (PyMuPDF / Markdown)
    |                      chunk_text (recursive split)
    |                      embed_texts (fastembed ONNX)
    |                          |
    |                          v
    |                    vector_store.py
    |                      Qdrant In-Memory (384-dim, Cosine)
    |
    +-- POST /query  --> rag_engine.py (Orchestrator)
    |                      |
    |                      1. query_classifier.py (Adaptive RAG)
    |                      |    Classify: simple | standard | complex
    |                      |
    |                      2. SIMPLE --> direkte LLM-Antwort (kein Retrieval)
    |                         STANDARD --> embed + search + CRAG + generate
    |                         COMPLEX --> Multi-Query-Decomposition:
    |                      |              Sub-Queries -> je embed + search + CRAG
    |                      |              -> Merge + Deduplicate -> generate
    |                      |
    |                      3. relevance_checker.py (CRAG)
    |                      |    Relevanz-Check pro Chunk (Confidence >= 0.3)
    |                      |    Fallback wenn keine relevanten Chunks
    |                      |
    |                      4. generate (Claude via OpenRouter)
    |
    +-- GET /documents --> vector_store.py (list)
    +-- DELETE /documents/{id} --> vector_store.py (delete)
```

## API Endpoints

| Methode | Pfad | Beschreibung |
|---------|------|--------------|
| `POST` | `/upload` | Datei hochladen und indexieren |
| `POST` | `/query` | Frage stellen, Antwort mit Quellen |
| `GET` | `/documents` | Alle indexierten Dokumente auflisten |
| `DELETE` | `/documents/{id}` | Dokument und Vektoren entfernen |

### Beispiele

```bash
# Dokument hochladen
curl -X POST http://localhost:8000/upload \
  -F "file=@dokument.md"

# Frage stellen (Response enthaelt routing + retrieval_quality)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Was ist der November-2025-Wendepunkt?", "top_k": 5}'

# Dokumente auflisten
curl http://localhost:8000/documents

# Dokument loeschen
curl -X DELETE http://localhost:8000/documents/{document_id}
```

## Projektstruktur

```
src/
  api.py                 # FastAPI Endpoints
  llm_client.py          # Shared OpenRouter Client + Konstanten
  query_classifier.py    # Adaptive RAG: Query-Routing (simple/standard/complex)
  relevance_checker.py   # CRAG: Post-Retrieval Relevanz-Check
  document_processor.py  # Text-Extraktion, Chunking, Embedding
  vector_store.py        # Qdrant Persistent Storage (data/qdrant/)
  rag_engine.py          # RAG Orchestrator (Classify -> Route -> Retrieve -> Filter -> Generate)
  main.py                # Server-Startup
static/
  index.html             # Chat UI (Single-File, kein Build)
scripts/
  upload_test_docs.py    # 5 Testdokumente hochladen + abfragen
tests/
  test_document_processor.py
  test_vector_store.py
  test_query_classifier.py
  test_relevance_checker.py
  test_api.py
```

## Tech Stack

| Komponente | Tool |
|------------|------|
| API Framework | FastAPI + Uvicorn |
| Vector DB | Qdrant (persistent file-based, kein Docker noetig) |
| Embeddings | fastembed / BAAI/bge-small-en-v1.5 (ONNX, lokal, konfigurierbar) |
| LLM | Claude Sonnet via OpenRouter |
| PDF Parsing | PyMuPDF |
| Frontend | Vanilla HTML/CSS/JS |

## Tests

```bash
pytest tests/ -v
# 41 Tests: document_processor (9), vector_store (7), query_classifier (5), relevance_checker (5), api (14), main (1)
```

## Konfiguration

Der Server liest API Keys aus `~/.claude/.env` oder Umgebungsvariablen:

| Variable | Zweck | Default |
|----------|-------|---------|
| `OPENROUTER_API_KEY` | LLM-Zugang (Claude via OpenRouter) | (erforderlich) |
| `RAG_API_KEY` | Bearer-Token fuer API-Authentifizierung | (leer = Auth deaktiviert) |
| `EMBEDDING_MODEL` | fastembed Modellname | `BAAI/bge-small-en-v1.5` |
| `EMBEDDING_DIM` | Vektor-Dimension passend zum Modell | `384` |
| `QDRANT_PATH` | Pfad fuer persistente Qdrant-Daten | `data/qdrant/` |

Embeddings laufen lokal -- kein weiterer Key noetig.
Fuer bessere deutsche Ergebnisse: `EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

## Einschraenkungen

- **Embedding-Modell**: bge-small-en-v1.5 ist gut fuer Englisch, akzeptabel fuer Deutsch (multilingual via Env-Var konfigurierbar)

## License

MIT
