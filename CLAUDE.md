# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

RAG (Retrieval-Augmented Generation) chatbot prototype. Upload documents (PDF/MD/TXT), index them as vector embeddings, query them via chat with LLM-generated answers citing sources. German-language prompts and UI by default.

## Commands

```bash
source venv/bin/activate
python src/main.py              # Start server at http://localhost:8000

pytest tests/ -v                # Run all tests (25 tests)
pytest tests/test_api.py -v     # Run one test file
pytest tests/test_api.py::TestQueryEndpoint -v  # Run one test class

python scripts/upload_test_docs.py  # Upload 5 test docs and run a query
```

## Architecture

Six-module pipeline, all in `src/`:

1. **api.py** — FastAPI app with 4 endpoints: `POST /upload`, `POST /query`, `GET /documents`, `DELETE /documents/{id}`. Serves `static/index.html` at `/`.
2. **document_processor.py** — Text extraction (PyMuPDF for PDF, raw decode for text), recursive character chunking (~500 tokens with overlap), local embeddings via fastembed (BAAI/bge-small-en-v1.5, 384-dim ONNX).
3. **vector_store.py** — Qdrant in-memory wrapper. Singleton client initialized on first access. Data does not persist across server restarts.
4. **llm_client.py** — Shared OpenRouter client singleton, MODEL constant, and route constants (ROUTE_SIMPLE/STANDARD/COMPLEX). All LLM-calling modules import from here.
5. **query_classifier.py** — Adaptive RAG: LLM-based classification of queries into simple (direct answer, no retrieval), standard (normal vector search + CRAG), or complex (multi-query decomposition into 2-3 sub-queries).
6. **relevance_checker.py** — Corrective RAG (CRAG): post-retrieval relevance check. Batches all chunks into one LLM call, filters chunks with confidence < 0.3, triggers fallback when nothing is relevant.
7. **rag_engine.py** — Orchestrator: classify → route → (retrieve → CRAG filter →) generate. Handles all three routes and produces `QueryResult` with routing info and retrieval quality metrics.

Query flow: `classify_query()` → route decision → SIMPLE: direct LLM answer | STANDARD: `embed_query()` → `search()` → `check_relevance()` → `build_context()` → LLM | COMPLEX: decompose into sub-queries, retrieve+filter each, merge deduplicated results → LLM.

Upload flow unchanged: `process_document()` → `upsert_chunks()`.

## Key Details

- **API keys**: `OPENROUTER_API_KEY` loaded from `~/.claude/.env` or project `.env` by `main.load_env()`. Embeddings are fully local (no key needed).
- **Tests reset Qdrant**: The `reset_vector_store` fixture sets `vs._client = None` before/after each test. Query tests mock `get_client` at each consumption site (`src.query_classifier.get_client`, `src.relevance_checker.get_client`, `src.rag_engine.get_client`).
- **Frontend**: Single-file vanilla HTML/CSS/JS at `static/index.html`, no build step.
- **Server runs with reload**: `uvicorn.run` uses `reload=True` and references the app as `"src.api:app"`, so `sys.path` must include the project root.

## Stack

Python 3.12+, FastAPI, Qdrant (in-memory), fastembed (ONNX), PyMuPDF, OpenAI SDK (targeting OpenRouter)
