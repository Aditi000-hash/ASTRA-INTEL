"""
utils.py
--------
Utility and helper functions for ASTRA INTEL.

WHAT: Provides common helper functions such as API key validation, 
      citation formatting, and text sanitization.
WHY:  Keeps other modules clean and avoids repeating logic.
HOW:  Pure Python helper functions with thorough type annotations and docstrings.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

# Automatically load environment variables from .env file if present
load_dotenv()


def get_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """
    Retrieves the Gemini API key from explicit user input or environment variables.

    Priority:
    1. Explicit key entered in the UI sidebar.
    2. GEMINI_API_KEY stored in the .env file or system environment.

    Returns:
        Cleaned API key string if found and valid, otherwise None.
    """
    key = explicit_key if explicit_key and explicit_key.strip() else os.getenv("GEMINI_API_KEY")
    if not key:
        return None

    cleaned_key = key.strip()
    # Check if the user left the default placeholder
    if cleaned_key == "your_api_key_here" or len(cleaned_key) < 10:
        return None

    return cleaned_key


def format_page_citations(pages: List[int]) -> str:
    """
    Formats a list of page numbers into a clean, human-readable citation string.

    Examples:
        [4] -> "Page 4"
        [3, 5] -> "Pages 3 and 5"
        [2, 4, 7] -> "Pages 2, 4, and 7"
        [] -> "Unknown Page"

    Args:
        pages: List of 1-based page numbers.

    Returns:
        Formatted string.
    """
    if not pages:
        return "Unknown Page"

    # Deduplicate and sort page numbers
    unique_pages = sorted(list(set(pages)))

    if len(unique_pages) == 1:
        return f"Page {unique_pages[0]}"
    elif len(unique_pages) == 2:
        return f"Pages {unique_pages[0]} and {unique_pages[1]}"
    else:
        all_but_last = ", ".join(str(p) for p in unique_pages[:-1])
        return f"Pages {all_but_last}, and {unique_pages[-1]}"


def clean_text(text: str) -> str:
    """
    Normalizes extracted PDF text by removing excessive whitespace and irregular linebreaks.

    Args:
        text: Raw text string.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""
    # Normalize carriage returns and tabs
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple blank lines into two newlines (paragraph boundary)
    lines = [line.strip() for line in text.split("\n")]
    cleaned_lines = []
    prev_blank = False
    for line in lines:
        if line:
            cleaned_lines.append(line)
            prev_blank = False
        elif not prev_blank:
            cleaned_lines.append("")
            prev_blank = True
    return "\n".join(cleaned_lines).strip()
