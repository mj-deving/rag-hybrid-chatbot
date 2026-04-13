"""FastAPI application with RAG endpoints."""

import os
from datetime import date
from pathlib import Path

from fastapi import Depends, FastAPI, Header, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

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

API_KEY = os.environ.get("RAG_API_KEY", "")


async def verify_api_key(authorization: str | None = Header(default=None)):
    """Check bearer token if RAG_API_KEY is configured."""
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.removeprefix("Bearer ")
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

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


class RoutingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    route: str = Field(description="Query route: simple, standard, or complex")
    sub_queries: list[str] = Field(default_factory=list, description="Sub-queries for complex route")


class RetrievalQualityResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    chunks_retrieved: int
    chunks_relevant: int
    chunks_filtered: int
    fallback_triggered: bool


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    tokens_used: int
    routing: RoutingResponse
    retrieval_quality: RetrievalQualityResponse | None = None


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


@app.post("/upload", response_model=UploadResponse, dependencies=[Depends(verify_api_key)])
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


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def query_documents(req: QueryRequest):
    """Ask a question against indexed documents."""
    result = rag_query(req.question, top_k=req.top_k)

    routing = RoutingResponse.model_validate(result.routing)
    retrieval_quality = (
        RetrievalQualityResponse.model_validate(result.retrieval_quality)
        if result.retrieval_quality is not None
        else None
    )

    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceResponse(document=s.document, chunk=s.chunk, relevance=s.relevance)
            for s in result.sources
        ],
        tokens_used=result.tokens_used,
        routing=routing,
        retrieval_quality=retrieval_quality,
    )


@app.get("/documents", response_model=DocumentListResponse, dependencies=[Depends(verify_api_key)])
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


@app.delete("/documents/{document_id}", response_model=DeleteResponse, dependencies=[Depends(verify_api_key)])
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
