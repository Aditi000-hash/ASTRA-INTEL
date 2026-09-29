"""
llm_service.py
--------------
Google Gemini API integration with strict anti-hallucination guardrails.

WHAT: Interfaces with Google Gemini to generate document summaries and answer
      user queries grounded strictly in retrieved context passages.
WHY:  Defence analysts need reliable answers. If information is absent, the AI
      must admit it rather than invent facts or page citations.
HOW:  Uses the official google-genai SDK, structured context injection with page
      demarcations, multi-turn history formatting, and explicit refusal instructions.
"""

import os
import time
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import errors


# Fallback constant required by ASTRA Challenge specification
REFUSAL_PHRASE = "I could not find this information in the uploaded document."

# Officially supported, stable production model for Google AI Studio
DEFAULT_MODEL = "gemini-3.8-flash"


class LLMServiceError(Exception):
    """Custom exception for LLM service failures with user-friendly messages."""
    pass


class LLMService:
    """Wrapper class managing Gemini API interactions."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        """
        Initializes the Gemini client.

        Args:
            api_key: Valid Google Gemini API Key.
            model: Gemini model identifier (default: gemini-3.8-flash, or GEMINI_MODEL env var).
        """
        if not api_key or not api_key.strip():
            raise LLMServiceError("Missing API key. Please configure your Gemini API key.")

        self.api_key = api_key.strip()
        # Support override via GEMINI_MODEL environment variable, defaulting to gemini-3.8-flash
        self.model = os.getenv("GEMINI_MODEL") or model
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise LLMServiceError(f"Failed to initialize Gemini Client: {str(e)}")

    def generate_summary(self, pages: List[Dict[str, Any]]) -> str:
        """
        Generates a concise executive summary based ONLY on the uploaded document.

        Args:
            pages: List of page dictionaries [{"page": int, "text": str}]

        Returns:
            Concise, structured summary string.
        """
        if not pages:
            return "No document text available to summarize."

        # Filter pages with actual text
        valid_pages = [p for p in pages if p.get("text", "").strip()]
        if not valid_pages:
            return "The document does not contain readable text."

        # Check total character length
        total_text = "\n\n".join([f"--- Page {p['page']} ---\n{p['text']}" for p in valid_pages])

        # If document is within moderate size (~25,000 chars), single-pass summary
        if len(total_text) <= 25000:
            prompt = (
                "You are an expert defence intelligence analyst reviewing an uploaded technical/operational document.\n"
                "Generate a concise, professional executive summary based ONLY on the provided text.\n"
                "Structure your summary as follows:\n"
                "- **Overview & Purpose**: Primary objective of the document.\n"
                "- **Key Specifications & Findings**: Important technical, strategic, or operational points.\n"
                "- **Operational Implications / Conclusions**: Critical takeaways.\n\n"
                "RULES:\n"
                "1. Base your summary ONLY on the text below.\n"
                "2. Do NOT speculate or incorporate external knowledge.\n\n"
                f"[DOCUMENT CONTENT]:\n{total_text}"
            )
            return self._call_gemini(prompt)

        # Multi-stage summarization for large documents
        # Stage 1: Summarize chunks/batches of pages
        batch_size = 5
        intermediate_summaries = []
        for i in range(0, len(valid_pages), batch_size):
            batch = valid_pages[i : i + batch_size]
            batch_text = "\n\n".join([f"Page {p['page']}: {p['text'][:1500]}" for p in batch])
            batch_prompt = (
                "Extract the core technical and operational points from these document pages:\n\n"
                f"{batch_text}"
            )
            stage_summary = self._call_gemini(batch_prompt)
            intermediate_summaries.append(stage_summary)

        # Stage 2: Synthesize into final concise summary
        combined_text = "\n\n".join(intermediate_summaries)
        final_prompt = (
            "You are an expert defence intelligence analyst.\n"
            "Synthesize the following section summaries into a single, cohesive, and concise executive summary.\n"
            "Structure:\n"
            "- **Overview & Purpose**\n"
            "- **Key Specifications & Capabilities**\n"
            "- **Critical Takeaways**\n\n"
            "Do NOT invent facts or use outside knowledge.\n\n"
            f"[EXTRACTED SECTION SUMMARIES]:\n{combined_text}"
        )
        return self._call_gemini(final_prompt)

    def answer_question(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Answers a user question based strictly on retrieved document chunks.

        Args:
            question: The user's question.
            context_chunks: Top retrieved chunks [{"page": int, "text": str, "score": float}]
            chat_history: Optional list of previous conversation turns
                          [{"role": "user"|"assistant", "content": str}]

        Returns:
            Dictionary with keys:
              - "answer": str
              - "sources": List[int] (page numbers)
              - "source_excerpts": List[Dict[str, Any]]
              - "is_grounded": bool
        """
        # Edge case: No relevant chunks found by retrieval
        if not context_chunks:
            return {
                "answer": REFUSAL_PHRASE,
                "sources": [],
                "source_excerpts": [],
                "is_grounded": False
            }

        # Build context block with exact page tags
        context_parts = []
        for idx, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"[Source Segment {idx} | Page {chunk['page']}]\n{chunk['text']}"
            )
        formatted_context = "\n\n".join(context_parts)

        # Build multi-turn conversation context if history exists
        history_text = ""
        if chat_history and len(chat_history) > 0:
            history_lines = []
            # Keep up to the last 4 turns to avoid exceeding context
            recent_turns = chat_history[-4:]
            for turn in recent_turns:
                speaker = "Analyst" if turn.get("role") == "user" else "ASTRA Intel"
                history_lines.append(f"{speaker}: {turn.get('content')}")
            history_text = "PRIOR CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n\n"

        prompt = (
            "You are ASTRA INTEL, an AI Defence Document Intelligence Assistant answering questions "
            "about an uploaded defence/technical document.\n\n"
            "STRICT ANTI-HALLUCINATION RULES:\n"
            "1. Use ONLY the supplied document context below to answer the question.\n"
            "2. Do NOT use outside knowledge, prior training data, or speculative inference.\n"
            "3. If the answer cannot be found or directly verified in the supplied context, you MUST reply EXACTLY:\n"
            f"   '{REFUSAL_PHRASE}'\n"
            "4. Do NOT guess, fabricate facts, or cite pages not present in the context.\n"
            "5. Answer in a professional, clear, and technically precise manner.\n\n"
            f"{history_text}"
            f"--- SUPPLIED DOCUMENT CONTEXT ---\n"
            f"{formatted_context}\n"
            f"--- END OF CONTEXT ---\n\n"
            f"QUESTION: {question.strip()}\n\n"
            "ANSWER:"
        )

        raw_answer = self._call_gemini(prompt).strip()

        # Check if model gave the refusal phrase
        is_refusal = (
            REFUSAL_PHRASE.lower() in raw_answer.lower()
            or "not found in the uploaded document" in raw_answer.lower()
            or "information is not mentioned" in raw_answer.lower()
        )

        if is_refusal:
            return {
                "answer": REFUSAL_PHRASE,
                "sources": [],
                "source_excerpts": [],
                "is_grounded": False
            }

        # Answer was found and grounded in context
        pages = sorted(list(set(chunk["page"] for chunk in context_chunks)))
        source_excerpts = [
            {
                "page": chunk["page"],
                "text": chunk["text"][:350] + ("..." if len(chunk["text"]) > 350 else ""),
                "score": chunk.get("score", 0.0)
            }
            for chunk in context_chunks
        ]

        return {
            "answer": raw_answer,
            "sources": pages,
            "source_excerpts": source_excerpts,
            "is_grounded": True
        }

    def _call_gemini(self, prompt: str) -> str:
        """
        Executes a call to Gemini API with robust retry handling for temporary 503 errors.

        Features:
        - Retries up to 3 times on temporary 503 UNAVAILABLE / high demand spikes.
        - Uses exponential backoff (1.5s -> 3.0s -> 6.0s).
        - Immediately raises clear errors for 429 Quota Exceeded without retrying.
        - Sanitizes error messages to ensure API keys are never leaked.

        Args:
            prompt: Text prompt string.

        Returns:
            Model response string.

        Raises:
            LLMServiceError: User-friendly error message on API/network failures.
        """
        max_retries = 3
        base_delay = 1.5
        backoff_factor = 2.0

        for attempt in range(max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
                if not response or not response.text:
                    raise LLMServiceError("Received an empty response from Gemini API.")
                return response.text.strip()

            except errors.ClientError as e:
                # Client errors (400 Bad Request, 403 Invalid Key, 429 Quota Exceeded)
                # These are NOT temporary network spikes and should NOT be retried
                err_msg = str(e).replace(self.api_key, "[REDACTED]")

                if "API_KEY_INVALID" in err_msg or "403" in err_msg:
                    raise LLMServiceError(
                        "Invalid Gemini API key. Please check your API key in the sidebar or .env file."
                    )
                elif "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                    raise LLMServiceError(
                        "Gemini API rate limit or quota exceeded (HTTP 429). "
                        "Please verify your usage quota at Google AI Studio or wait a moment before trying again."
                    )
                elif "404" in err_msg or "NOT_FOUND" in err_msg:
                    raise LLMServiceError(
                        f"The requested Gemini model '{self.model}' was not found (HTTP 404). "
                        "Please select 'gemini-3.8-flash' in the sidebar or check your Google AI Studio project settings."
                    )
                else:
                    raise LLMServiceError(f"Gemini API Client Error: {err_msg}")

            except errors.ServerError as e:
                # Server errors (500 Internal, 503 Unavailable / High Demand)
                err_msg = str(e).replace(self.api_key, "[REDACTED]")
                is_unavailable = (
                    getattr(e, "code", None) == 503
                    or "503" in err_msg
                    or "UNAVAILABLE" in err_msg
                    or "high demand" in err_msg.lower()
                    or "spikes in demand" in err_msg.lower()
                )

                if is_unavailable and attempt < max_retries:
                    delay = base_delay * (backoff_factor ** attempt)
                    time.sleep(delay)
                    continue

                if is_unavailable:
                    raise LLMServiceError(
                        f"The Gemini model '{self.model}' is currently experiencing high demand (HTTP 503) "
                        f"after {max_retries} retry attempts. Please wait a few moments or switch models in the sidebar."
                    )
                else:
                    raise LLMServiceError(
                        f"The AI service is currently unavailable. Please try again. (Details: {err_msg})"
                    )

            except LLMServiceError:
                raise

            except Exception as e:
                clean_msg = str(e).replace(self.api_key, "[REDACTED]")
                raise LLMServiceError(f"Unexpected error communicating with Gemini API: {clean_msg}")

        raise LLMServiceError("Service request failed after retry attempts.")
