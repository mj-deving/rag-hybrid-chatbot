# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

RAG (Retrieval-Augmented Generation) chatbot prototype. Upload documents (PDF/MD/TXT), index them as vector embeddings, query them via chat with LLM-generated answers citing sources. German-language prompts and UI by default.

## Commands

```bash
source venv/bin/activate
python src/main.py              # Start server at http://localhost:8000

pytest tests/ -v                # Run all tests (41 tests)
pytest tests/test_api.py -v     # Run one test file
pytest tests/test_api.py::TestQueryEndpoint -v  # Run one test class

python scripts/upload_test_docs.py  # Upload 5 test docs and run a query
```

## Architecture

Seven-module pipeline, all in `src/`:

1. **api.py** — FastAPI app with 4 endpoints: `POST /upload`, `POST /query`, `GET /documents`, `DELETE /documents/{id}`. Bearer token auth via `RAG_API_KEY` env var (disabled when unset). Serves `static/index.html` at `/`.
2. **document_processor.py** — Text extraction (PyMuPDF for PDF, raw decode for text), recursive character chunking (~500 tokens with overlap), local embeddings via fastembed. Model configurable via `EMBEDDING_MODEL` env var (default: BAAI/bge-small-en-v1.5, 384-dim).
3. **vector_store.py** — Qdrant persistent file-based storage at `data/qdrant/` (configurable via `QDRANT_PATH`). Collection auto-created on first access. Data survives server restarts.
4. **llm_client.py** — Shared OpenRouter client singleton, MODEL/MODEL_FAST constants, route constants. Classifier and CRAG use Haiku (fast), generation uses Sonnet (quality).
5. **query_classifier.py** — Adaptive RAG: LLM-based classification into simple/standard/complex.
6. **relevance_checker.py** — CRAG: post-retrieval relevance filtering (confidence < 0.3 filtered).
7. **rag_engine.py** — Orchestrator: classify → route → retrieve → filter → generate. Complex route runs sub-queries in parallel via ThreadPoolExecutor.

Query flow: `classify_query()` → route decision → SIMPLE: direct LLM | STANDARD: embed → search → CRAG → generate | COMPLEX: parallel sub-queries → merge → generate.

## Key Details

- **API keys**: `OPENROUTER_API_KEY` loaded from `~/.claude/.env` or project `.env`. `RAG_API_KEY` enables bearer auth when set.
- **Tests**: `conftest.py` provides `isolated_qdrant` fixture — gives each test a temp Qdrant directory via monkeypatch. LLM tests mock `get_client` at each consumption site.
- **Frontend**: Single-file vanilla HTML/CSS/JS at `static/index.html`, no build step.
- **Server runs with reload**: `uvicorn.run` uses `reload=True` and references the app as `"src.api:app"`.

## Stack

Python 3.12+, FastAPI, Qdrant (persistent file-based), fastembed (ONNX), PyMuPDF, OpenAI SDK (targeting OpenRouter)
