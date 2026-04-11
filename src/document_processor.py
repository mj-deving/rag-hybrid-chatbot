"""Document processing: text extraction, chunking, and embedding."""

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from fastembed import TextEmbedding

# Lazy-loaded singleton
_model: TextEmbedding | None = None
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
CHUNK_SIZE = 2000  # ~500 tokens ≈ 2000 chars
CHUNK_OVERLAP = 200  # ~50 tokens ≈ 200 chars


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    document_id: str
    embedding: list[float] = field(default_factory=list)


def get_model() -> TextEmbedding:
    """Get or initialize the embedding model (singleton)."""
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts."""
    model = get_model()
    return [emb.tolist() for emb in model.embed(texts)]


def embed_query(text: str) -> list[float]:
    """Generate embedding for a single query."""
    model = get_model()
    return next(model.query_embed(text)).tolist()


def extract_text(file_path: Path, content_bytes: bytes | None = None) -> str:
    """Extract text from PDF or Markdown file."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        if content_bytes:
            doc = fitz.open(stream=content_bytes, filetype="pdf")
        else:
            doc = fitz.open(str(file_path))
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        return text

    # Markdown and plain text
    if content_bytes is not None:
        return content_bytes.decode("utf-8", errors="replace")
    return file_path.read_text(encoding="utf-8", errors="replace")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks using recursive character splitting."""
    if not text.strip():
        return []

    separators = ["\n\n", "\n", ". ", " "]
    return _recursive_split(text, separators, chunk_size, overlap)


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Recursively split text by trying separators in order."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    sep = separators[0] if separators else ""
    remaining_seps = separators[1:] if len(separators) > 1 else []

    parts = text.split(sep) if sep else [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part) if current else part

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                if len(current) > chunk_size and remaining_seps:
                    chunks.extend(_recursive_split(current, remaining_seps, chunk_size, overlap))
                else:
                    chunks.append(current.strip())
            current = part

    if current.strip():
        if len(current) > chunk_size and remaining_seps:
            chunks.extend(_recursive_split(current, remaining_seps, chunk_size, overlap))
        else:
            chunks.append(current.strip())

    # Add overlap between consecutive chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + chunks[i])
        chunks = overlapped

    return chunks


def process_document(
    filename: str,
    file_path: Path | None = None,
    content_bytes: bytes | None = None,
) -> list[Chunk]:
    """Full pipeline: extract text, chunk, embed. Returns list of Chunks."""
    path = file_path or Path(filename)
    text = extract_text(path, content_bytes)
    text_chunks = chunk_text(text)

    if not text_chunks:
        return []

    embeddings = embed_texts(text_chunks)
    document_id = uuid.uuid4().hex[:12]

    return [
        Chunk(
            text=chunk_text_item,
            source=filename,
            chunk_index=i,
            document_id=document_id,
            embedding=emb,
        )
        for i, (chunk_text_item, emb) in enumerate(zip(text_chunks, embeddings))
    ]
