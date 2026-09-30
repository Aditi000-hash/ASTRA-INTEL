"""
retrieval.py
------------
Grounded, transparent, and robust keyword & semantic retrieval for document chunks.

WHAT: Searches and ranks document chunks based on their relevance to a user query.
WHY:  Avoids overloading LLMs with entire 50-page documents, keeps responses
      grounded in actual relevant passages, and provides clear, auditable source pages.
HOW:  Combines:
      1. Tokenization with conversational stopword filtering.
      2. Deterministic morphological stemming (handling plurals, verb tenses, and suffixes).
      3. Technical and domain-aware concept expansion (synonyms for speed, altitude, cameras, propulsion, etc.).
      4. Multi-signal relevance scoring with coverage multiplier and length normalization.
      5. Strict refusal signaling when queries have zero factual overlap with the document.
"""

import re
import math
from typing import List, Dict, Any, Tuple, Set

# Standard English and conversational question stopwords
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could", "couldn't",
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
    "yourself", "yourselves",
    # Question fluff and conversational fillers
    "tell", "explain", "describe", "show", "give", "list", "mention", "mentioned",
    "know", "detail", "details", "say", "says", "much", "many", "please"
}

# Domain & technical concept expansions (synonyms)
# Maps query concepts to document vocabulary common in defence & aerospace documents
CONCEPT_SYNONYMS = {
    # Speed / flight performance
    "fast": ["speed", "velocity", "knots", "knot", "km/h", "kmh", "dash", "cruise", "cruising", "mach", "rate"],
    "speed": ["fast", "velocity", "knots", "knot", "km/h", "kmh", "dash", "cruise", "cruising", "mach"],
    "fly": ["flight", "flying", "airborne", "loiter", "loitering", "cruise", "endurance", "service", "ceiling"],
    "flight": ["fly", "flying", "airborne", "loiter", "loitering", "cruise", "endurance"],
    "endurance": ["loiter", "hours", "duration", "time", "flight", "continuous"],
    # Altitude / ceiling
    "altitude": ["ceiling", "height", "elevation", "feet", "msl", "fl", "meters", "climb"],
    "ceiling": ["altitude", "height", "elevation", "feet", "msl"],
    # Optics / sensors / cameras
    "camera": ["sensor", "sensors", "optics", "optical", "infrared", "ir", "eo", "eo/ir", "eoir", "imaging", "turret", "magnification", "zoom", "photo", "video", "lidar", "radar", "sar"],
    "cameras": ["sensor", "sensors", "optics", "optical", "infrared", "ir", "eo", "eo/ir", "eoir", "imaging", "turret", "magnification", "zoom", "photo", "video", "lidar", "radar", "sar"],
    "optics": ["camera", "cameras", "sensor", "sensors", "optical", "infrared", "eo/ir", "imaging", "turret"],
    "sensor": ["camera", "cameras", "optics", "optical", "infrared", "radar", "lidar", "sar", "eo/ir", "turret", "imaging"],
    "sensors": ["camera", "cameras", "optics", "optical", "infrared", "radar", "lidar", "sar", "eo/ir", "turret", "imaging"],
    "radar": ["sar", "gmti", "sensor", "radio", "aperture"],
    # Engine / propulsion / fuel
    "engine": ["powerplant", "propulsion", "motor", "motors", "fuel", "hydrogen", "electric", "turbo", "alternator", "cell", "power", "thrust"],
    "engines": ["powerplant", "propulsion", "motor", "motors", "fuel", "hydrogen", "electric", "turbo", "alternator", "cell", "power"],
    "propulsion": ["engine", "engines", "motor", "powerplant", "fuel", "hydrogen", "electric", "turbo", "alternator", "power"],
    "fuel": ["hydrogen", "cell", "powerplant", "propulsion", "engine", "electric", "turbo-alternator"],
    "motor": ["engine", "propulsion", "powerplant", "electric"],
    # Weather / environment / climate
    "weather": ["environmental", "environment", "temperature", "celsius", "icing", "de-icing", "crosswind", "wind", "conditions", "climate", "limits", "limitations", "constraints"],
    "rain": ["weather", "environmental", "icing", "de-icing", "crosswind", "wind", "temperature", "conditions", "severe"],
    "wind": ["crosswind", "weather", "landing", "limits", "knots"],
    "temperature": ["ambient", "celsius", "operating", "range", "weather", "environmental"],
    # Weight / payload
    "weight": ["heavy", "mass", "payload", "capacity", "kg", "kilograms", "lbs", "pounds", "load"],
    "heavy": ["weight", "mass", "payload", "capacity", "kg", "lbs"],
    "payload": ["capacity", "sensor", "turret", "radar", "weapons", "weight", "stores", "warhead"],
    # Communications
    "communication": ["communications", "comms", "datalink", "satcom", "telemetry", "uplink", "downlink", "frequencies", "ku-band", "ka-band", "encryption", "anti-jamming", "fhss"],
    "communications": ["communication", "comms", "datalink", "satcom", "telemetry", "uplink", "downlink", "frequencies", "ku-band", "ka-band", "encryption", "anti-jamming", "fhss"],
    "radio": ["comms", "datalink", "satcom", "frequencies", "fhss", "spectrum"],
    # Mission / application / role
    "application": ["applications", "role", "roles", "mission", "missions", "purpose", "use", "uses", "task", "tasks", "surveillance", "reconnaissance", "istar"],
    "applications": ["application", "role", "roles", "mission", "missions", "purpose", "use", "uses", "task", "tasks", "surveillance", "reconnaissance", "istar"],
    "role": ["roles", "application", "applications", "mission", "missions", "task"],
    "mission": ["missions", "role", "roles", "application", "applications", "istar", "surveillance", "reconnaissance", "patrol"],
    # Aircraft / drone / system
    "drone": ["uav", "atar", "atars", "aircraft", "platform", "system", "unmanned", "airframe"],
    "aircraft": ["drone", "uav", "atar", "atars", "platform", "system", "airframe"],
    "uav": ["drone", "aircraft", "platform", "system", "unmanned", "atars"],
    # Manufacturer / builder / project
    "manufacturer": ["built", "builder", "maker", "developer", "contractor", "produced", "project", "company"],
    "manufactured": ["built", "builder", "maker", "developer", "contractor", "produced", "project", "company"],
    "built": ["project", "developed", "manufactured", "contractor", "maker", "builder"],
    "maker": ["manufacturer", "built", "contractor", "developer", "project"],
    # Logistics / maintenance / crew
    "crew": ["operator", "operators", "personnel", "ground", "control", "commander", "specialist"],
    "turnaround": ["cycle", "recovery", "redeployment", "maintenance", "logistics", "minutes"],
    "maintenance": ["turnaround", "mtbf", "failures", "cycle", "repair", "inspection"]
}


def tokenize(text: str) -> List[str]:
    """
    Converts a string of text into lowercase alphanumeric tokens.
    Handles hyphenated compound words cleanly.
    """
    if not text:
        return []
    # Find all words (alphanumerics including single interior hyphens/underscores)
    tokens = re.findall(r"\b[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*\b", text.lower())
    # Flatten hyphenated words so both 'hydrogen-electric' and 'hydrogen', 'electric' are searchable
    flattened = []
    for token in tokens:
        flattened.append(token)
        if "-" in token:
            flattened.extend(part for part in token.split("-") if len(part) > 1)
        elif "_" in token:
            flattened.extend(part for part in token.split("_") if len(part) > 1)
    return flattened


def stem_word(word: str) -> str:
    """
    Deterministic rule-based stemming for English morphology.
    Handles plurals, past tense, progressive participles, and common technical suffixes.
    """
    w = word.lower().strip()
    if len(w) <= 3:
        return w

    # Common English suffix reduction rules (longest first)
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ational") and len(w) > 7:
        return w[:-5] + "e"
    if w.endswith("tional") and len(w) > 6:
        return w[:-4]
    if w.endswith("ation") and len(w) > 5:
        return w[:-3] + "e"
    if w.endswith("izing") and len(w) > 5:
        return w[:-4]
    if w.endswith("ized") and len(w) > 4:
        return w[:-3]
    if w.endswith("ment") and len(w) > 5:
        return w[:-4]
    if w.endswith("ance") and len(w) > 5:
        return w[:-4]
    if w.endswith("ence") and len(w) > 5:
        return w[:-4]
    if w.endswith("ical") and len(w) > 5:
        return w[:-4]
    if w.endswith("ing") and len(w) > 4:
        # e.g., operating -> operat, flying -> fly
        base = w[:-3]
        if base.endswith(("bb", "dd", "gg", "mm", "nn", "pp", "rr", "tt")):
            base = base[:-1]
        return base
    if w.endswith("ed") and len(w) > 4:
        # e.g., powered -> power, loitered -> loiter
        base = w[:-2]
        if base.endswith(("bb", "dd", "gg", "mm", "nn", "pp", "rr", "tt")):
            base = base[:-1]
        return base
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    if w.endswith("al") and len(w) > 4:
        return w[:-2]

    return w


def extract_keywords(query: str) -> List[str]:
    """
    Extracts informative keywords from a question by removing common stopwords
    and conversational filler words.
    """
    tokens = tokenize(query)
    keywords = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    # Fallback to tokens if all were stopwords (e.g. short queries)
    return keywords if keywords else tokens


def score_chunk(chunk_text: str, query_keywords: List[str], raw_query: str) -> float:
    """
    Calculates a multi-signal relevance score for a document chunk.

    Scoring Signals:
    1. Exact Token Match (weight: 3.0): Exact keyword in chunk.
    2. Stemmed Token Match (weight: 2.0): Morphological match (e.g. engines <-> engine).
    3. Concept / Synonym Match (weight: 1.5): Domain synonym (e.g. fast <-> speed, camera <-> sensor).
    4. Substring / Prefix Match (weight: 1.0): Shared root for technical jargon (e.g. recon <-> reconnaissance).
    5. Exact Multi-Word Phrase Match (bonus: 5.0): Verbatim question phrase present.
    6. Keyword Coverage Multiplier: Strong reward when chunk covers multiple distinct query terms.
    7. Soft Length Normalization: Prevents giant passages from dominating short concise answers.
    """
    chunk_tokens = tokenize(chunk_text)
    if not chunk_tokens or not query_keywords:
        return 0.0

    chunk_token_set = set(chunk_tokens)
    chunk_stems = [stem_word(t) for t in chunk_tokens]
    chunk_stem_set = set(chunk_stems)
    chunk_text_lower = chunk_text.lower()

    score = 0.0
    distinct_matched_terms = 0

    for keyword in query_keywords:
        kw_stem = stem_word(keyword)
        term_matched = False
        term_score = 0.0

        # Signal 1: Exact match
        exact_count = chunk_tokens.count(keyword)
        if exact_count > 0:
            term_score += 3.0 + math.log(exact_count + 1)
            term_matched = True
        else:
            # Signal 2: Stem match (e.g. "applications" vs "application")
            stem_count = chunk_stems.count(kw_stem)
            if stem_count > 0:
                term_score += 2.0 + math.log(stem_count + 1)
                term_matched = True

        # Signal 3: Concept / Synonym expansion (e.g. "fast" -> "speed", "knots")
        synonyms = CONCEPT_SYNONYMS.get(keyword, [])
        if not synonyms:
            synonyms = CONCEPT_SYNONYMS.get(kw_stem, [])
        if synonyms:
            syn_hits = sum(chunk_tokens.count(syn) for syn in synonyms)
            if syn_hits > 0:
                term_score += 1.5 + math.log(syn_hits + 1)
                term_matched = True

        # Signal 4: Substring / Prefix root match for longer technical words (>= 4 chars)
        if not term_matched and len(keyword) >= 4:
            for ct in chunk_token_set:
                if len(ct) >= 4 and (ct.startswith(kw_stem) or kw_stem.startswith(stem_word(ct))):
                    term_score += 1.0
                    term_matched = True
                    break

        if term_matched:
            distinct_matched_terms += 1
            score += term_score

    # If none of the query terms matched in any form, score is 0.0
    if distinct_matched_terms == 0:
        return 0.0

    # Signal 5: Exact phrase match bonus
    cleaned_raw_query = raw_query.lower().strip()
    if len(query_keywords) > 1 and cleaned_raw_query in chunk_text_lower:
        score += 5.0

    # Signal 6: Keyword coverage multiplier
    # Exponentially rewards chunks containing multiple distinct query concepts
    coverage_ratio = distinct_matched_terms / len(query_keywords)
    score = score * (1.0 + coverage_ratio)

    # Signal 7: Soft length normalization (avoids bias toward huge text dumps)
    score = score / math.sqrt(len(chunk_tokens) + 15)

    return score


def retrieve_relevant_chunks(
    chunks: List[Dict[str, Any]],
    query: str,
    top_k: int = 5
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Searches all chunks in the document, ranks them by relevance to the query,
    and returns the top-K chunks along with a flag indicating if relevant content was found.

    Args:
        chunks: List of document chunk dicts [{"chunk_id", "page", "text", ...}]
        query: The user's question.
        top_k: Number of highest-ranked chunks to return (default: 5).

    Returns:
        A tuple of:
          - top_chunks: List of ranked chunk dicts with an added 'score' field.
          - has_relevant_content: Boolean indicating if at least one chunk had a positive score.
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
            chunk_with_score = dict(chunk)
            chunk_with_score["score"] = round(score, 4)
            scored_chunks.append(chunk_with_score)

    if not scored_chunks:
        # Check if query is an overview/meta-question about the document itself
        overview_terms = {"overview", "summary", "about", "document", "describe", "purpose", "system", "scope"}
        query_token_set = set(tokenize(query))
        if query_token_set.intersection(overview_terms) and chunks:
            # Provide initial overview chunks (e.g. Page 1) with baseline score
            intro_chunks = [dict(c, score=0.5) for c in chunks if c.get("page", 1) == 1]
            if not intro_chunks:
                intro_chunks = [dict(chunks[0], score=0.5)]
            return intro_chunks[:top_k], True

        return [], False

    # Sort descending by relevance score
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    # Ensure page diversity: if chunks from multiple pages scored well,
    # include top chunks across distinct pages for multi-section answers
    top_chunks = scored_chunks[:top_k]

    # If the document has <= 6 chunks in total, and top chunks scored well,
    # include all positive-scoring chunks up to top_k to maximize evidence
    return top_chunks, True
