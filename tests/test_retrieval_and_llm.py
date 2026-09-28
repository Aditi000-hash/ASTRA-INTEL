"""
Unit tests for retrieval logic, anti-hallucination guardrails, and citations.
"""

import pytest
from src.retrieval import (
    tokenize,
    extract_keywords,
    score_chunk,
    retrieve_relevant_chunks
)
from src.utils import format_page_citations, get_api_key
from src.llm_service import LLMService, LLMServiceError, REFUSAL_PHRASE


def test_tokenize_and_keyword_extraction():
    """Verify that tokenization and stopword removal isolate meaningful words."""
    query = "What are the major tactical applications of the system?"
    keywords = extract_keywords(query)
    # Stopwords like 'what', 'are', 'the', 'of' should be removed
    assert "major" in keywords
    assert "tactical" in keywords
    assert "applications" in keywords
    assert "system" in keywords
    assert "the" not in keywords
    assert "what" not in keywords


def test_retrieve_relevant_chunks_found():
    """Verify that chunks matching keywords receive positive scores and correct page tags."""
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "Project Astra is an unmanned aerial reconnaissance platform."},
        {"chunk_id": 2, "page": 2, "text": "Hybrid hydrogen-electric powerplant with 36 hours endurance."},
        {"chunk_id": 3, "page": 3, "text": "Major applications include maritime border surveillance and artillery direction."},
        {"chunk_id": 4, "page": 4, "text": "Weather limitations: operating temperature -40C to +55C."}
    ]

    query = "What are the major applications of the drone?"
    top_chunks, has_matches = retrieve_relevant_chunks(chunks, query, top_k=2)

    assert has_matches is True
    assert len(top_chunks) > 0
    # Chunk 3 has both 'major' and 'applications' so it must be top ranked
    assert top_chunks[0]["chunk_id"] == 3
    assert top_chunks[0]["page"] == 3
    assert top_chunks[0]["score"] > 0


def test_retrieve_relevant_chunks_not_found():
    """Verify that a completely irrelevant query yields no matches."""
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "Project Astra is an unmanned aerial reconnaissance platform."},
        {"chunk_id": 2, "page": 2, "text": "Propulsion and satellite communications details."}
    ]

    irrelevant_query = "quantum microwave baking chocolate cookies recipe"
    top_chunks, has_matches = retrieve_relevant_chunks(chunks, irrelevant_query, top_k=2)

    assert has_matches is False
    assert len(top_chunks) == 0


def test_format_page_citations():
    """Verify citation formatting rules."""
    assert format_page_citations([]) == "Unknown Page"
    assert format_page_citations([4]) == "Page 4"
    assert format_page_citations([2, 5]) == "Pages 2 and 5"
    assert format_page_citations([1, 3, 4]) == "Pages 1, 3, and 4"
    # Duplicates should be deduplicated and sorted
    assert format_page_citations([4, 2, 4, 2]) == "Pages 2 and 4"


def test_get_api_key_validation():
    """Verify API key validation logic."""
    # Empty or whitespace
    assert get_api_key(None) is None or isinstance(get_api_key(None), str)
    assert get_api_key("") is None or isinstance(get_api_key(""), str)
    # Placeholder
    assert get_api_key("your_api_key_here") is None
    # Short bogus string
    assert get_api_key("abc") is None
    # Valid candidate string
    valid_candidate = "AIzaSyD-TEST_KEY_FOR_UNIT_TESTING_12345"
    assert get_api_key(valid_candidate) == valid_candidate


def test_llm_service_initialization_error():
    """Verify that LLMService raises LLMServiceError if initialized without a key."""
    with pytest.raises(LLMServiceError):
        LLMService(api_key="")


def test_llm_service_anti_hallucination_empty_context():
    """
    Verify that when no context chunks are supplied, answer_question returns the
    official refusal phrase immediately without making an external LLM request.
    """
    llm = LLMService(api_key="AIzaSyDummyKeyForOfflineUnitTesting123456789")
    result = llm.answer_question(
        question="What is the nuclear payload capacity?",
        context_chunks=[]
    )
    assert result["answer"] == REFUSAL_PHRASE
    assert result["sources"] == []
    assert result["is_grounded"] is False
