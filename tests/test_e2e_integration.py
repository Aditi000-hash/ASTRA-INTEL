"""
Integration tests verifying the end-to-end workflow:
PDF loading -> extraction -> chunking -> retrieval -> citation resolution -> anti-hallucination.
"""

import os
import pytest
from src.pdf_processor import extract_pages_from_pdf, chunk_pages
from src.retrieval import retrieve_relevant_chunks
from src.utils import format_page_citations
from src.llm_service import LLMService, REFUSAL_PHRASE


SAMPLE_PDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "sample_docs", "astra_defence_spec.pdf"
)


def test_end_to_end_retrieval_flow():
    """Simulates a full user interaction workflow using the sample defence document."""
    # 1. Load PDF
    with open(SAMPLE_PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    pages, metadata = extract_pages_from_pdf(pdf_bytes)
    assert len(pages) == 4
    assert metadata["has_readable_text"] is True

    # 2. Chunk pages
    chunks = chunk_pages(pages, chunk_size=800, chunk_overlap=150)
    assert len(chunks) >= 4

    # 3. User asks: "What are the major applications?"
    query = "What are the major operational applications?"
    matched_chunks, has_matches = retrieve_relevant_chunks(chunks, query, top_k=3)
    assert has_matches is True
    assert len(matched_chunks) > 0

    # Ensure Page 3 is in the retrieved results
    retrieved_pages = [c["page"] for c in matched_chunks]
    assert 3 in retrieved_pages, "Page 3 contains the Major Applications section and must be retrieved"

    citation = format_page_citations(retrieved_pages)
    assert "3" in citation
    assert "Pages" in citation or "Page" in citation

    # 4. User asks an unsupported question not in the document:
    # "What is the submarine diving depth?"
    unsupported_query = "What is the submarine diving depth in meters?"
    unsupported_chunks, has_unsupported_matches = retrieve_relevant_chunks(chunks, unsupported_query)

    # Retrieval should either find nothing or LLMService will refuse
    if not has_unsupported_matches:
        llm = LLMService(api_key="AIzaSyDummyKeyForOfflineUnitTesting123456789")
        response = llm.answer_question(unsupported_query, unsupported_chunks)
        assert response["answer"] == REFUSAL_PHRASE
        assert response["is_grounded"] is False
        assert response["sources"] == []
