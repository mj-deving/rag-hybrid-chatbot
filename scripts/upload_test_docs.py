"""Upload the 5 test documents from KI-Roadmap to the RAG API."""

import sys
from pathlib import Path
import httpx

BASE_URL = "http://localhost:8000"

TEST_DOCS = [
    Path.home() / "projects/KI-Roadmap/Analyse/extracts/ext_simon-willison-agentic-dev-2026.md",
    Path.home() / "projects/KI-Roadmap/Analyse/extracts/ext_500-workflows-businesses-want-2026.md",
    Path.home() / "projects/KI-Roadmap/Analyse/extracts/ext_claude-managed-agents-vs-n8n-2026.md",
    Path.home() / "projects/KI-Roadmap/Analyse/extracts/ext_obsidian-karpathy-rag-wiki-2026.md",
    Path.home() / "projects/KI-Roadmap/Strategie/Input-Synthese-Strategie-II.md",
]

TEST_QUESTIONS = [
    "Was ist der November-2025-Wendepunkt?",
    "Welche 5 Workflows wollen Unternehmen 2026?",
    "Wie unterscheiden sich Managed Agents von n8n?",
    "Was ist Graph RAG vs Naive RAG?",
    "Was ist das 3-Checkpoint-System für KI-Fluency?",
]


def upload_docs():
    """Upload all test documents."""
    print("=== Uploading Test Documents ===\n")
    for doc_path in TEST_DOCS:
        if not doc_path.exists():
            print(f"SKIP: {doc_path.name} — file not found")
            continue

        with open(doc_path, "rb") as f:
            resp = httpx.post(
                f"{BASE_URL}/upload",
                files={"file": (doc_path.name, f, "text/markdown")},
                timeout=60.0,
            )

        if resp.status_code == 200:
            data = resp.json()
            print(f"OK: {data['filename']} — {data['chunks']} chunks, ID: {data['document_id']}")
        else:
            print(f"ERR: {doc_path.name} — {resp.status_code}: {resp.text}")

    print()


def list_docs():
    """List all indexed documents."""
    print("=== Indexed Documents ===\n")
    resp = httpx.get(f"{BASE_URL}/documents", timeout=10.0)
    data = resp.json()
    for doc in data["documents"]:
        print(f"  {doc['id']}  {doc['filename']}  ({doc['chunks']} chunks)")
    print()


def ask_questions():
    """Run the 5 test questions."""
    print("=== Test Queries ===\n")
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"Q{i}: {question}")
        resp = httpx.post(
            f"{BASE_URL}/query",
            json={"question": question, "top_k": 5},
            timeout=30.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            print(f"A{i}: {data['answer'][:200]}...")
            print(f"    Sources: {', '.join(s['document'] for s in data['sources'])}")
            print(f"    Tokens: {data['tokens_used']}")
        else:
            print(f"    ERR: {resp.status_code}: {resp.text}")
        print()


if __name__ == "__main__":
    if "--upload" in sys.argv or len(sys.argv) == 1:
        upload_docs()
    if "--list" in sys.argv or len(sys.argv) == 1:
        list_docs()
    if "--ask" in sys.argv or len(sys.argv) == 1:
        ask_questions()
