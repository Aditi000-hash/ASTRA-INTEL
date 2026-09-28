"""
Unit tests for PDF processing and chunking (src/pdf_processor.py).
"""

import os
import pytest
from src.pdf_processor import extract_pages_from_pdf, chunk_pages, PDFProcessingError
from src.utils import clean_text


SAMPLE_PDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "sample_docs", "astra_defence_spec.pdf"
)


def test_extract_valid_pdf():
    """Verify that a valid multi-page PDF is extracted with correct page numbering."""
    assert os.path.exists(SAMPLE_PDF_PATH), "Sample PDF should exist"
    with open(SAMPLE_PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    pages, metadata = extract_pages_from_pdf(pdf_bytes)

    assert len(pages) == 4, "Sample PDF should contain 4 pages"
    assert metadata["total_pages"] == 4
    assert metadata["total_characters"] > 1000
    assert metadata["is_empty"] is False
    assert metadata["has_readable_text"] is True

    # Verify 1-based page numbers are preserved strictly
    page_numbers = [p["page"] for p in pages]
    assert page_numbers == [1, 2, 3, 4], "Pages must be 1-indexed and sequential"

    # Verify specific content on specific pages
    assert "PROJECT ASTRA" in pages[0]["text"]
    assert "PROPULSION & ENDURANCE" in pages[1]["text"]
    assert "MAJOR APPLICATIONS" in pages[2]["text"]
    assert "ENVIRONMENTAL CONSTRAINTS" in pages[3]["text"]


def test_extract_empty_bytes():
    """Verify that empty bytes trigger a PDFProcessingError."""
    with pytest.raises(PDFProcessingError) as exc_info:
        extract_pages_from_pdf(b"")
    assert "empty" in str(exc_info.value).lower()


def test_extract_corrupt_bytes():
    """Verify that random corrupt bytes trigger a PDFProcessingError."""
    corrupt_bytes = b"%PDF-1.4\nCorrupted binary header data that cannot be parsed by mupdf..."
    with pytest.raises(PDFProcessingError) as exc_info:
        extract_pages_from_pdf(corrupt_bytes)
    assert "could not open" in str(exc_info.value).lower()


def test_chunking_preserves_page_numbers():
    """Verify that chunks never cross page boundaries and retain page numbers."""
    pages = [
        {"page": 1, "text": "Page 1 content about tactical aerial reconnaissance systems."},
        {"page": 2, "text": "Page 2 content about hybrid hydrogen-electric powerplant and flight endurance."}
    ]

    chunks = chunk_pages(pages, chunk_size=50, chunk_overlap=10)

    assert len(chunks) >= 2, "Should create multiple chunks"
    for chunk in chunks:
        assert "page" in chunk
        assert chunk["page"] in [1, 2]
        assert "chunk_id" in chunk
        assert "text" in chunk
        # Content on page 1 chunk should never contain page 2 text
        if chunk["page"] == 1:
            assert "powerplant" not in chunk["text"]
        elif chunk["page"] == 2:
            assert "tactical aerial" not in chunk["text"]


def test_clean_text_utility():
    """Verify that text cleaning normalizes linebreaks and collapses whitespace."""
    raw = "Line 1\r\n\r\n\r\nLine 2  \n   Line 3\n\n\n\nLine 4"
    cleaned = clean_text(raw)
    assert "\r" not in cleaned
    assert "Line 1\n\nLine 2\nLine 3\n\nLine 4" == cleaned
