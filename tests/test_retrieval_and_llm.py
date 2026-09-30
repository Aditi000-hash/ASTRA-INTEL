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


def test_llm_service_default_model():
    """Verify that LLMService defaults to gemini-3.8-flash."""
    llm = LLMService(api_key="AIzaSyDummyKeyForModelDefaultCheck12345")
    assert llm.model == "gemini-3.8-flash"


def test_llm_service_503_retry_and_exhaustion(monkeypatch):
    """Verify that a 503 ServerError triggers retries and halts at max_retries with clear message."""
    from google.genai import errors

    llm = LLMService(api_key="AIzaSyDummyKeyFor503Check12345")
    call_count = 0

    def mock_generate_content(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Simulate Google 503 UNAVAILABLE ServerError
        raise errors.ServerError(code=503, response_json={"error": {"message": "503 UNAVAILABLE: This model is currently experiencing high demand."}})

    monkeypatch.setattr(llm.client.models, "generate_content", mock_generate_content)
    # Monkeypatch time.sleep in src.llm_service to avoid waiting during fast unit testing
    monkeypatch.setattr("src.llm_service.time.sleep", lambda s: None)

    with pytest.raises(LLMServiceError) as exc_info:
        llm._call_gemini("test prompt")

    # 1 initial attempt + 3 retries = 4 total calls
    assert call_count == 4
    assert "503" in str(exc_info.value) or "high demand" in str(exc_info.value).lower()
    # Ensure raw API key is never in error message
    assert "AIzaSyDummyKeyFor503Check12345" not in str(exc_info.value)


def test_llm_service_429_quota_immediate_rejection(monkeypatch):
    """Verify that a 429 Quota Exceeded error is raised immediately on attempt 1 without retries."""
    from google.genai import errors

    llm = LLMService(api_key="AIzaSyDummyKeyFor429Check12345")
    call_count = 0

    def mock_generate_content(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Simulate Google 429 RESOURCE_EXHAUSTED ClientError
        raise errors.ClientError(code=429, response_json={"error": {"message": "429 RESOURCE_EXHAUSTED: Quota exceeded for quota metric."}})

    monkeypatch.setattr(llm.client.models, "generate_content", mock_generate_content)

    with pytest.raises(LLMServiceError) as exc_info:
        llm._call_gemini("test prompt")

    # Must fail immediately on attempt 1 without retry
    assert call_count == 1
    assert "quota" in str(exc_info.value).lower() or "429" in str(exc_info.value)
    # Ensure raw API key is never in error message
    assert "AIzaSyDummyKeyFor429Check12345" not in str(exc_info.value)


def test_natural_language_synonym_retrieval():
    """Verify that natural language queries with synonyms (fast -> speed/knots) retrieve the right chunk."""
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "Project Astra carbon-fiber airframe."},
        {"chunk_id": 2, "page": 2, "text": "Cruising speed: 180 knots. Maximum dash speed: 260 knots."},
        {"chunk_id": 3, "page": 3, "text": "EO/IR gyrostabilized turret with 50x optical magnification camera."},
        {"chunk_id": 4, "page": 4, "text": "Operating ambient temperature range -40C to +55C."}
    ]

    # Test 1: speed synonym (fast)
    top_chunks, has_matches = retrieve_relevant_chunks(chunks, "How fast can it fly?", top_k=2)
    assert has_matches is True
    assert top_chunks[0]["page"] == 2
    assert "180 knots" in top_chunks[0]["text"]

    # Test 2: optics synonym (cameras)
    top_chunks, has_matches = retrieve_relevant_chunks(chunks, "What cameras or optics does it carry?", top_k=2)
    assert has_matches is True
    assert top_chunks[0]["page"] == 3


def test_multi_page_retrieval():
    """Verify that multi-section queries retrieve evidence spanning multiple distinct pages."""
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "Project Astra executive overview and carbon composite structure."},
        {"chunk_id": 2, "page": 2, "text": "Hybrid hydrogen-electric fuel cell propulsion powerplant."},
        {"chunk_id": 3, "page": 3, "text": "Maritime border surveillance and tactical reconnaissance payloads."},
        {"chunk_id": 4, "page": 4, "text": "Severe weather limitations: maximum crosswind 35 knots, temperature -40C."}
    ]

    query = "What is the propulsion powerplant and what are the weather limitations?"
    top_chunks, has_matches = retrieve_relevant_chunks(chunks, query, top_k=3)
    assert has_matches is True
    retrieved_pages = set(c["page"] for c in top_chunks)
    assert 2 in retrieved_pages
    assert 4 in retrieved_pages


def test_llm_service_refusal_distinction_empty_context():
    """Verify that empty context returns status_detail 'NO_RELEVANT_PASSAGES'."""
    llm = LLMService(api_key="AIzaSyDummyKeyForRefusalCheck12345")
    result = llm.answer_question(
        question="What is the nuclear warhead capacity?",
        context_chunks=[]
    )
    assert result["answer"] == REFUSAL_PHRASE
    assert result["is_grounded"] is False
    assert result["status_detail"] == "NO_RELEVANT_PASSAGES"


def test_llm_service_refusal_distinction_gemini_refusal(monkeypatch):
    """Verify that Gemini refusing returns status_detail 'INSUFFICIENT_EVIDENCE'."""
    llm = LLMService(api_key="AIzaSyDummyKeyForRefusalCheck12345")
    monkeypatch.setattr(llm, "_call_gemini", lambda prompt: REFUSAL_PHRASE)

    result = llm.answer_question(
        question="What is the nuclear warhead capacity?",
        context_chunks=[{"page": 3, "text": "Tactical payloads include EO/IR turret and radar.", "score": 1.0}]
    )
    assert result["answer"] == REFUSAL_PHRASE
    assert result["is_grounded"] is False
    assert result["status_detail"] == "INSUFFICIENT_EVIDENCE"

