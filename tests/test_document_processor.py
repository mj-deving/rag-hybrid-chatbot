"""Tests for document processing module."""

import pytest
from pathlib import Path

from src.document_processor import extract_text, chunk_text, process_document


class TestExtractText:
    def test_markdown_from_bytes(self, tmp_path):
        content = b"# Title\n\nSome content here.\n\n## Section 2\n\nMore text."
        text = extract_text(Path("test.md"), content_bytes=content)
        assert "Title" in text
        assert "Some content here" in text

    def test_markdown_from_file(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello\n\nWorld")
        text = extract_text(md_file)
        assert "Hello" in text
        assert "World" in text

    def test_pdf_from_file(self, tmp_path):
        """Test PDF extraction (creates a minimal PDF)."""
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test PDF content")
        pdf_path = tmp_path / "test.pdf"
        doc.save(str(pdf_path))
        doc.close()

        text = extract_text(pdf_path)
        assert "Test PDF content" in text


class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("Short text", chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_empty_text(self):
        chunks = chunk_text("")
        assert chunks == []

    def test_long_text_multiple_chunks(self):
        text = "Word " * 1000  # ~5000 chars
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) > 1
        for chunk in chunks:
            # Chunks may slightly exceed due to overlap
            assert len(chunk) < 1000

    def test_splits_on_paragraphs(self):
        text = ("Paragraph one. " * 50 + "\n\n" + "Paragraph two. " * 50)
        chunks = chunk_text(text, chunk_size=400, overlap=0)
        assert len(chunks) >= 2


class TestProcessDocument:
    def test_process_markdown(self, tmp_path):
        md_file = tmp_path / "doc.md"
        md_file.write_text("# Test\n\n" + "Content line. " * 100)
        chunks = process_document("doc.md", file_path=md_file)
        assert len(chunks) > 0
        assert chunks[0].source == "doc.md"
        assert chunks[0].document_id  # non-empty
        assert len(chunks[0].embedding) == 384  # bge-small-en-v1.5

    def test_chunk_indices_sequential(self, tmp_path):
        md_file = tmp_path / "doc.md"
        md_file.write_text("Content. " * 500)
        chunks = process_document("doc.md", file_path=md_file)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))
