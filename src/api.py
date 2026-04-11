"""FastAPI application with RAG endpoints."""

from datetime import date
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load env vars before importing modules that need API keys
from src.main import load_env
load_env()

from src.document_processor import process_document
from src.vector_store import upsert_chunks, list_documents, delete_document
from src.rag_engine import query as rag_query

app = FastAPI(
    title="RAG Chatbot API",
    description="Dokumente hochladen, indexieren und Fragen stellen mit kontextbasierten Antworten.",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).parent.parent / "static"


@app.get("/", include_in_schema=False)
async def root():
    """Serve the chat UI."""
    return FileResponse(STATIC_DIR / "index.html")


# --- Request/Response Models ---


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


class SourceResponse(BaseModel):
    document: str
    chunk: int
    relevance: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    tokens_used: int


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int
    status: str


class DocumentResponse(BaseModel):
    id: str
    filename: str
    chunks: int
    uploaded: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class DeleteResponse(BaseModel):
    document_id: str
    chunks_deleted: int
    status: str


# --- Endpoints ---


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or Markdown file for indexing."""
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()

    if suffix not in (".pdf", ".md", ".txt", ".markdown"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Use .pdf, .md, or .txt",
        )

    content = await file.read()

    # Process: extract text, chunk, embed
    chunks = process_document(
        filename=filename,
        file_path=Path(filename),
        content_bytes=content,
    )

    if not chunks:
        raise HTTPException(status_code=400, detail="No text content found in file")

    # Store in vector DB
    upsert_chunks(chunks)
    document_id = chunks[0].document_id

    return UploadResponse(
        document_id=document_id,
        filename=filename,
        chunks=len(chunks),
        status="indexed",
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    """Ask a question against indexed documents."""
    result = rag_query(req.question, top_k=req.top_k)

    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceResponse(document=s.document, chunk=s.chunk, relevance=s.relevance)
            for s in result.sources
        ],
        tokens_used=result.tokens_used,
    )


@app.get("/documents", response_model=DocumentListResponse)
async def get_documents():
    """List all indexed documents."""
    docs = list_documents()
    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=d.id,
                filename=d.filename,
                chunks=d.chunks,
                uploaded=str(date.today()),
            )
            for d in docs
        ]
    )


@app.delete("/documents/{document_id}", response_model=DeleteResponse)
async def remove_document(document_id: str):
    """Remove a document and all its vectors."""
    deleted = delete_document(document_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    return DeleteResponse(
        document_id=document_id,
        chunks_deleted=deleted,
        status="deleted",
    )
