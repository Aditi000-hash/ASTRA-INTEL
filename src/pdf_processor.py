"""
pdf_processor.py
----------------
Page-aware PDF processing and text extraction using PyMuPDF (fitz).

WHAT: Extracts text from uploaded PDF documents page-by-page, retains page
      metadata, detects empty/scanned PDFs, and chunks text for grounded retrieval.
WHY:  Defence and intelligence documents require precise, page-accurate citations.
      We must never merge pages into one blind string, nor fabricate page numbers.
HOW:  PyMuPDF opens the PDF in-memory, iterates sequentially through pages (1-indexed),
      extracts text, computes document diagnostics, and applies page-isolated chunking.
"""

from typing import List, Dict, Tuple, Any
import pymupdf as fitz
from src.utils import clean_text


class PDFProcessingError(Exception):
    """Custom exception raised when PDF processing fails gracefully."""
    pass


def extract_pages_from_pdf(pdf_bytes: bytes) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts text page-by-page from raw PDF bytes using PyMuPDF.

    Args:
        pdf_bytes: The raw byte content of an uploaded PDF file.

    Returns:
        A tuple of:
          - pages: A list of dicts: [{"page": int, "text": str, "char_count": int}]
          - metadata: A dict with diagnostic stats:
                      {"total_pages": int, "total_characters": int,
                       "pages_with_text": int, "is_empty": bool, "has_readable_text": bool}

    Raises:
        PDFProcessingError: If the file is corrupt, not a PDF, or unreadable.
    """
    if not pdf_bytes or len(pdf_bytes) == 0:
        raise PDFProcessingError("Uploaded file is empty (0 bytes).")

    try:
        # Open PDF from memory stream without writing to disk
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise PDFProcessingError(
            f"Could not open file as a valid PDF. It may be corrupt or encrypted. Details: {str(e)}"
        )

    try:
        total_pages = doc.page_count
        if total_pages == 0:
            raise PDFProcessingError("The PDF document contains 0 pages.")

        pages_data: List[Dict[str, Any]] = []
        total_chars = 0
        pages_with_text = 0

        for page_index in range(total_pages):
            page_num = page_index + 1  # 1-indexed for human readability
            page = doc.load_page(page_index)
            raw_text = page.get_text("text")
            normalized_text = clean_text(raw_text)
            char_count = len(normalized_text)

            total_chars += char_count
            if char_count > 0:
                pages_with_text += 1

            pages_data.append({
                "page": page_num,
                "text": normalized_text,
                "char_count": char_count
            })

        metadata = {
            "total_pages": total_pages,
            "total_characters": total_chars,
            "pages_with_text": pages_with_text,
            "is_empty": total_chars == 0,
            "has_readable_text": total_chars > 20  # Threshold to detect non-OCR image scans
        }

        return pages_data, metadata

    except PDFProcessingError:
        raise
    except Exception as e:
        raise PDFProcessingError(f"Unexpected error while extracting text: {str(e)}")
    finally:
        # Always close document handle to avoid resource leaks
        try:
            doc.close()
        except Exception:
            pass


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Splits page-level text into smaller, overlapping chunks while strictly
    preserving the page number for each chunk.

    IMPORTANT ARCHITECTURAL RULE:
    Chunks NEVER cross page boundaries. This guarantees that if a chunk is
    retrieved, its page citation is 100% accurate and verifiable.

    Args:
        pages: List of page dictionaries with "page" and "text" keys.
        chunk_size: Target character length per chunk (default: 800 chars).
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of chunk dictionaries:
        [{"chunk_id": int, "page": int, "text": str, "char_count": int}]
    """
    chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for page_item in pages:
        page_num = page_item["page"]
        text = page_item["text"].strip()

        if not text:
            continue

        # If page text is short enough, keep it as a single chunk
        if len(text) <= chunk_size:
            chunk_id += 1
            chunks.append({
                "chunk_id": chunk_id,
                "page": page_num,
                "text": text,
                "char_count": len(text)
            })
            continue

        # Sliding window chunking within the single page
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)

            # If we're not at the very end, try to snap to the nearest space or newline
            # so we don't slice a word in half
            if end < text_len:
                last_space = text.rfind(" ", start, end)
                last_newline = text.rfind("\n", start, end)
                split_point = max(last_space, last_newline)
                if split_point > start + (chunk_size // 2):
                    end = split_point

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id += 1
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page_num,
                    "text": chunk_text,
                    "char_count": len(chunk_text)
                })

            if end >= text_len:
                break

            # Advance by step size (chunk_size - chunk_overlap)
            step = max(1, (end - start) - chunk_overlap)
            start += step

    return chunks
