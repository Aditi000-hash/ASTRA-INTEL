"""
app.py
------
ASTRA INTEL: AI-Powered Defence Document Intelligence System
Streamlit Frontend Application.

WHAT: The main user interface providing document upload, page-aware diagnostics,
      document summarization, grounded Q&A, and page-level source citations.
WHY:  Gives defence analysts a clean, reliable, and auditable tool to extract
      verifiable insights from technical documents without hallucinated information.
HOW:  Streamlit handles reactive UI state, PyMuPDF extracts page text, the keyword
      retrieval engine selects relevant chunks, and Gemini generates grounded answers.
"""

import os
import streamlit as st

from src.pdf_processor import extract_pages_from_pdf, chunk_pages, PDFProcessingError
from src.retrieval import retrieve_relevant_chunks
from src.llm_service import LLMService, LLMServiceError, REFUSAL_PHRASE
from src.utils import get_api_key, format_page_citations


# Set page configuration
st.set_page_config(
    page_title="ASTRA INTEL | Defence Document Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean technical defence styling
if hasattr(st, "html"):
    st.html("""<style>
/* Technical Defence Theme Styling */
.stApp {
    background-color: #0e1117;
    color: #e0e0e0;
}
</style>""")


# Initialize Session State
if "pages" not in st.session_state:
    st.session_state.pages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "metadata" not in st.session_state:
    st.session_state.metadata = {}
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sample_loaded" not in st.session_state:
    st.session_state.sample_loaded = False


def reset_document_state():
    """Resets all document-specific session state variables."""
    st.session_state.pages = []
    st.session_state.chunks = []
    st.session_state.metadata = {}
    st.session_state.document_name = None
    st.session_state.summary = None
    st.session_state.messages = []
    st.session_state.sample_loaded = False


# ==========================================
# SIDEBAR: Configuration & Document Upload
# ==========================================
with st.sidebar:
    st.markdown("### 🛡️ ASTRA INTEL")
    st.markdown("**Defence Document Intelligence**")
    st.markdown("---")

    # 1. API Key Input
    env_key = get_api_key()
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=env_key if env_key else "",
        type="password",
        help="Reads from .env automatically if available, or enter manually here.",
        placeholder="AIzaSy..."
    )
    active_api_key = get_api_key(api_key_input)

    if active_api_key:
        st.success("🟢 API Key Active")
    else:
        st.warning("🔴 API Key Missing")
        with st.expander("ℹ️ How to configure API key"):
            st.markdown(
                "**Option 1:** Enter your key in the box above.\n\n"
                "**Option 2:** Set `GEMINI_API_KEY=your_key` in the `.env` file.\n\n"
                "[Get a key from Google AI Studio](https://aistudio.google.com/)"
            )

    st.markdown("---")

    # 2. Model Selection
    model_options = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    env_default_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    default_idx = model_options.index(env_default_model) if env_default_model in model_options else 0
    selected_model = st.selectbox(
        "Gemini Model",
        options=model_options,
        index=default_idx,
        help="gemini-3.8-flash is the current flagship model for new projects on Google AI Studio."
    )

    st.markdown("---")

    # 3. Document Upload Section
    st.markdown("#### 📄 Document Ingestion")

    uploaded_file = st.file_uploader(
        "Upload Defence / Technical PDF",
        type=["pdf"],
        help="Upload standard PDF format. Text is extracted page-by-page."
    )

    # Quick demo button: Load synthetic sample PDF
    sample_pdf_path = os.path.join(os.path.dirname(__file__), "sample_docs", "astra_defence_spec.pdf")
    if os.path.exists(sample_pdf_path):
        if st.button("📁 Load Sample Defence Brief", help="Loads pre-built unclassified UAV technical brief"):
            with open(sample_pdf_path, "rb") as f:
                sample_bytes = f.read()
            try:
                pages, meta = extract_pages_from_pdf(sample_bytes)
                chunks = chunk_pages(pages)
                st.session_state.pages = pages
                st.session_state.chunks = chunks
                st.session_state.metadata = meta
                st.session_state.document_name = "astra_defence_spec.pdf (Sample)"
                st.session_state.summary = None
                st.session_state.messages = []
                st.session_state.sample_loaded = True
                st.success("Sample document loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load sample: {str(e)}")

    # Process uploaded file if user uploaded one
    if uploaded_file is not None and (st.session_state.document_name != uploaded_file.name):
        try:
            pdf_bytes = uploaded_file.read()
            with st.spinner("Extracting page text and diagnostics..."):
                pages, meta = extract_pages_from_pdf(pdf_bytes)
                chunks = chunk_pages(pages)

                st.session_state.pages = pages
                st.session_state.chunks = chunks
                st.session_state.metadata = meta
                st.session_state.document_name = uploaded_file.name
                st.session_state.summary = None
                st.session_state.messages = []
                st.session_state.sample_loaded = False

                st.success("✓ Document processed successfully")
        except PDFProcessingError as pe:
            st.error(f"Document Error: {str(pe)}")
            reset_document_state()
        except Exception as e:
            st.error(f"Unexpected processing error: {str(e)}")
            reset_document_state()

    # 3. Document Information Badges
    if st.session_state.document_name and st.session_state.metadata:
        meta = st.session_state.metadata
        st.markdown("---")
        st.markdown("#### 📊 Document Diagnostics")
        st.text(f"File: {st.session_state.document_name}")
        st.text(f"Pages: {meta.get('total_pages', 0)}")
        st.text(f"Total Chars: {meta.get('total_characters', 0):,}")
        st.text(f"Active Chunks: {len(st.session_state.chunks)}")

        if meta.get("is_empty"):
            st.warning("⚠️ Warning: Document contains 0 characters.")
        elif not meta.get("has_readable_text"):
            st.warning("⚠️ Warning: Document appears to be a scanned image with no embedded text.")
        else:
            st.info("✓ Readable text verified across pages")

        if st.button("🗑️ Clear Document"):
            reset_document_state()
            st.rerun()


# ==========================================
# MAIN CONTENT AREA
# ==========================================
st.title("🛡️ ASTRA INTEL")
st.subheader("AI-Powered Defence Document Intelligence System")
st.caption("ASTRA Software Team 3-Day Build Challenge 2026–27 | Challenge 01")

# Guard: Check if document is loaded
if not st.session_state.pages:
    st.info(
        "👋 **Welcome to ASTRA INTEL.**\n\n"
        "To begin analyzing defence and technical intelligence documents:\n"
        "1. Ensure your **Gemini API Key** is configured in the sidebar (or `.env` file).\n"
        "2. **Upload a PDF** using the sidebar uploader, or click **'Load Sample Defence Brief'** to test immediately.\n\n"
        "**Core Capabilities:**\n"
        "- 📑 **Page-Aware Processing**: Preserves exact page numbers for 100% auditable citations.\n"
        "- 📝 **Executive Summary**: Generates concise intelligence summaries from document text.\n"
        "- 🎯 **Grounded Q&A**: Strict anti-hallucination guardrails prevent guessing when facts are absent.\n"
        "- 🔍 **Transparent Citations**: Displays exact page references and source excerpts."
    )
    st.stop()


# ------------------------------------------
# SECTION 1: Document Summary
# ------------------------------------------
with st.expander("📋 DOCUMENT EXECUTIVE SUMMARY", expanded=(st.session_state.summary is not None)):
    col_sum_btn, col_sum_status = st.columns([1, 4])
    with col_sum_btn:
        generate_sum_clicked = st.button("Generate Summary", key="btn_summary")

    if generate_sum_clicked:
        if not active_api_key:
            st.error("Please provide a valid Gemini API key in the sidebar to generate a summary.")
        elif not st.session_state.metadata.get("has_readable_text"):
            st.error("Cannot summarize: This PDF does not contain readable text.")
        else:
            with st.spinner("Analyzing document structure and synthesizing executive summary..."):
                try:
                    llm = LLMService(api_key=active_api_key, model=selected_model)
                    summary_text = llm.generate_summary(st.session_state.pages)
                    st.session_state.summary = summary_text
                except LLMServiceError as le:
                    st.error(str(le))
                except Exception as ex:
                    st.error(f"Error generating summary: {str(ex)}")

    if st.session_state.summary:
        st.markdown(st.session_state.summary)


# ------------------------------------------
# SECTION 2: Grounded Q&A Interface
# ------------------------------------------
st.markdown("### 💬 Ask Your Document")
st.caption(
    "Queries are answered strictly from the document. If information is absent, the system will explicitly state so."
)

# Suggested prompt chips for immediate testing
st.markdown("**Quick Prompts:**")
p_cols = st.columns(4)
chip_query = None
with p_cols[0]:
    if st.button("🎯 Major Applications", key="chip_1"):
        chip_query = "What are the major operational applications of this system?"
with p_cols[1]:
    if st.button("⚡ Flight Endurance", key="chip_2"):
        chip_query = "What is the operational flight endurance and service ceiling?"
with p_cols[2]:
    if st.button("🌡️ Weather Limits", key="chip_3"):
        chip_query = "What are the environmental constraints and weather limitations?"
with p_cols[3]:
    if st.button("🚫 Nuclear Payload (Refusal Test)", key="chip_4"):
        chip_query = "What is the nuclear warhead payload capacity of this aircraft?"


# Question Input Bar
user_query = st.chat_input("Enter your question about the document...")
if chip_query:
    user_query = chip_query


# ------------------------------------------
# SECTION 3: Question Processing & Answering
# ------------------------------------------
if user_query:
    query_text = user_query.strip()
    if not query_text:
        st.warning("Please enter a question.")
    elif not active_api_key:
        st.error("Gemini API key is required to query the document. Please add it in the sidebar.")
    else:
        # Step 1: Grounded Retrieval of relevant chunks
        relevant_chunks, has_matches = retrieve_relevant_chunks(
            chunks=st.session_state.chunks,
            query=query_text,
            top_k=4
        )

        # Step 2: Query Gemini with strict anti-hallucination instructions
        with st.spinner("Retrieving document passages and formulating grounded response..."):
            try:
                llm = LLMService(api_key=active_api_key, model=selected_model)
                response_dict = llm.answer_question(
                    question=query_text,
                    context_chunks=relevant_chunks if has_matches else [],
                    chat_history=st.session_state.messages
                )

                # Step 3: Record exchange in multi-turn session history
                st.session_state.messages.append({
                    "role": "user",
                    "content": query_text
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_dict["answer"],
                    "sources": response_dict["sources"],
                    "source_excerpts": response_dict["source_excerpts"],
                    "is_grounded": response_dict["is_grounded"]
                })

            except LLMServiceError as le:
                st.error(str(le))
            except Exception as e:
                st.error(f"Error answering question: {str(e)}")


# ------------------------------------------
# SECTION 4: Multi-Turn Conversation History
# ------------------------------------------
if st.session_state.messages:
    st.markdown("---")
    st.markdown("#### 📜 Conversation & Intelligence Log")

    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(f"**Analyst Query:** {msg['content']}")
        else:
            with st.chat_message("assistant", avatar="🛡️"):
                st.markdown(msg["content"])

                # Display Citations & Grounding Status
                if msg.get("is_grounded"):
                    sources = msg.get("sources", [])
                    citation_label = format_page_citations(sources)

                    st.success(f"✓ **GROUNDED IN DOCUMENT** — Source: **{citation_label}**")

                    # Expandable source text verification
                    excerpts = msg.get("source_excerpts", [])
                    if excerpts:
                        with st.expander("🔍 View Source Passages & Page References"):
                            for idx, ex in enumerate(excerpts, 1):
                                st.markdown(
                                    f"**Passage {idx} (Page {ex['page']})** — Relevance Score: `{ex.get('score', 'N/A')}`"
                                )
                                st.code(ex["text"], language=None)
                else:
                    st.warning("⚠️ **NOT IN DOCUMENT** — No factual support exists in the uploaded document. Speculation rejected.")
