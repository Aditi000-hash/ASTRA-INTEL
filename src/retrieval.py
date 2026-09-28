"""
retrieval.py
------------
Grounded, transparent keyword retrieval for document chunks.

WHAT: Searches and ranks document chunks based on their relevance to a user query.
WHY:  Avoids overloading LLMs with entire 50-page documents, keeps responses
      grounded in actual relevant passages, and provides clear, auditable source pages.
HOW:  Uses tokenized term frequency with stopword filtering and phrase matching.
      This is 100% deterministic, transparent, and easy for a beginner to explain
      without needing complex vector databases or external embeddings.
"""

import re
import math
from typing import List, Dict, Any, Tuple

# Standard English stopwords to prevent common words like 'the' or 'is' from skewing results
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def tokenize(text: str) -> List[str]:
    """
    Converts a string of text into lowercase alphanumeric tokens.
    Filters out punctuation and symbols.

    Args:
        text: Input string.

    Returns:
        List of lowercase tokens.
    """
    if not text:
        return []
    # Find all words (letters and numbers), lowercase them
    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    return tokens


def extract_keywords(query: str) -> List[str]:
    """
    Extracts informative keywords from a question by removing common stopwords.

    Args:
        query: User question string.

    Returns:
        List of meaningful keywords. If all words were stopwords, returns all tokens.
    """
    tokens = tokenize(query)
    keywords = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    # If the user asked something very short like "who is he", fallback to raw tokens
    return keywords if keywords else tokens


def score_chunk(chunk_text: str, query_keywords: List[str], raw_query: str) -> float:
    """
    Calculates a relevance score for a chunk given the query keywords.

    Scoring formula:
    1. Term match: Each occurrence of a keyword adds points.
    2. Exact phrase bonus: If multi-word query appears verbatim in chunk.
    3. Normalization: Slightly penalizes excessively long chunks so short concise
       matches are prioritized.

    Args:
        chunk_text: The text of the document chunk.
        query_keywords: Informative keywords extracted from the user query.
        raw_query: The original question string.

    Returns:
        Numerical relevance score (higher = more relevant).
    """
    chunk_tokens = tokenize(chunk_text)
    if not chunk_tokens or not query_keywords:
        return 0.0

    chunk_token_set = set(chunk_tokens)
    score = 0.0

    # 1. Keyword overlap and frequency
    matched_keywords = 0
    for keyword in query_keywords:
        count = chunk_tokens.count(keyword)
        if count > 0:
            matched_keywords += 1
            # Logarithmic term frequency scaling
            score += 1.0 + math.log(count + 1)

    # If none of the keywords match, score is 0
    if matched_keywords == 0:
        return 0.0

    # Keyword coverage multiplier: rewarding chunks that contain MORE of the distinct keywords
    coverage_ratio = matched_keywords / len(query_keywords)
    score = score * (1.0 + coverage_ratio)

    # 2. Exact phrase match bonus
    cleaned_raw_query = raw_query.lower().strip()
    if len(query_keywords) > 1 and cleaned_raw_query in chunk_text.lower():
        score += 5.0

    # 3. Soft length normalization (prevent extreme bias toward long chunks)
    score = score / math.sqrt(len(chunk_tokens) + 10)

    return score


def retrieve_relevant_chunks(
    chunks: List[Dict[str, Any]],
    query: str,
    top_k: int = 4
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Searches all chunks in the document, ranks them by relevance to the query,
    and returns the top-K chunks along with a flag indicating if matches were found.

    Args:
        chunks: List of document chunk dicts [{"chunk_id", "page", "text", ...}]
        query: The user's question.
        top_k: Number of highest-ranked chunks to return (default: 4).

    Returns:
        A tuple of:
          - top_chunks: List of ranked chunk dicts with an added 'score' field.
          - has_relevant_content: Boolean indicating if at least one chunk had a score > 0.
    """
    if not chunks or not query or not query.strip():
        return [], False

    keywords = extract_keywords(query)
    if not keywords:
        return [], False

    scored_chunks = []
    for chunk in chunks:
        score = score_chunk(chunk["text"], keywords, query)
        if score > 0:
            # Create a copy with the calculated relevance score
            chunk_with_score = dict(chunk)
            chunk_with_score["score"] = round(score, 4)
            scored_chunks.append(chunk_with_score)

    if not scored_chunks:
        # No keyword matches found in any chunk
        return [], False

    # Sort descending by score
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate or limit to top_k
    top_chunks = scored_chunks[:top_k]

    return top_chunks, True
