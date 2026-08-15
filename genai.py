"""
Document Assistant — Streamlit frontend

Reuses your existing helper modules (reader.py, chunk.py) exactly as they
were — only the presentation layer has changed.

Run with:
    streamlit run app.py
"""

import hashlib
import html
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

from reader import read_file
from chunk import chunk_text, create_vector_store, semantic_search

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

MODELS = [
    "google_genai:gemini-3.6-flash",
    "google_genai:gemini-3.5-flash",
    "google_genai:gemini-3.5-flash-lite",
]

SUPPORTED_TYPES = ["pdf", "docx", "txt", "md", "pptx", "csv", "xlsx", "xls", "json"]

st.set_page_config(page_title="Document Assistant", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------
# Design system — "Reading Room": a card-catalog / archival-index aesthetic.
# Paper-sage background, ink-teal "stamp" accent, brass secondary accent,
# a display serif for headers, and a mono utility face for metadata.
# --------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;0,9..144,700;1,9..144,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

        :root {
            --da-ink: #232821;
            --da-ink-soft: #5C6259;
            --da-ink-faint: #74796C;
            --da-paper: #EDEFE7;
            --da-card: #FFFFFF;
            --da-line: #D6D9CB;
            --da-accent: #1F6F5C;
            --da-accent-soft: #E4EEE9;
            --da-brass: #B4802B;
            --da-radius: 10px;
            --da-font-display: 'Fraunces', Georgia, serif;
            --da-font-body: 'IBM Plex Sans', -apple-system, sans-serif;
            --da-font-mono: 'IBM Plex Mono', 'SFMono-Regular', monospace;
        }

        /* ---- base canvas ---- */
        [data-testid="stAppViewContainer"] {
            background: var(--da-paper) !important;
            color: var(--da-ink) !important;
            font-family: var(--da-font-body) !important;
        }
        [data-testid="stHeader"] {
            background: var(--da-paper) !important;
            border-bottom: 1px solid var(--da-line);
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 900px;
            padding-top: 2.25rem;
        }

        /* ---- typography ---- */
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4 {
            font-family: var(--da-font-display) !important;
            font-weight: 600 !important;
            color: var(--da-ink) !important;
        }
        [data-testid="stMarkdownContainer"] p { line-height: 1.65; }
        [data-testid="stMarkdownContainer"] a { color: var(--da-accent); }
        [data-testid="stMarkdownContainer"] code {
            font-family: var(--da-font-mono);
            background: var(--da-paper);
            border: 1px solid var(--da-line);
            border-radius: 4px;
            padding: 0.1rem 0.35rem;
        }
        [data-testid="stCaptionContainer"] {
            font-family: var(--da-font-mono) !important;
            letter-spacing: 0.02em;
        }

        /* ---- sidebar ---- */
        [data-testid="stSidebar"] {
            background: var(--da-card) !important;
            border-right: 1px solid var(--da-line);
        }
        [data-testid="stSidebarContent"] { padding-top: 1.5rem; }
        .da-sidebar-title {
            font-family: var(--da-font-mono);
            font-size: 0.72rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--da-brass);
            border-bottom: 1px solid var(--da-line);
            padding-bottom: 0.6rem;
            margin-bottom: 1.1rem;
        }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            font-family: var(--da-font-mono) !important;
            font-size: 0.7rem !important;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--da-ink-soft) !important;
        }

        /* ---- chips ---- */
        .da-chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
        .da-chip {
            font-family: var(--da-font-mono);
            font-size: 0.66rem;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            color: var(--da-ink-soft);
            background: var(--da-paper);
            border: 1px solid var(--da-line);
            border-radius: 999px;
            padding: 0.15rem 0.55rem;
        }

        /* ---- hero index card ---- */
        .da-hero {
            background: var(--da-card);
            border: 1px solid var(--da-line);
            border-top: 3px solid var(--da-accent);
            border-radius: var(--da-radius);
            padding: 1.75rem 2rem 1.5rem;
            margin-bottom: 1.75rem;
        }
        .da-hero__eyebrow {
            font-family: var(--da-font-mono);
            font-size: 0.7rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--da-brass);
            margin-bottom: 0.5rem;
        }
        .da-hero__title {
            font-family: var(--da-font-display);
            font-weight: 600;
            font-size: 2.1rem;
            color: var(--da-ink);
            line-height: 1.15;
            margin-bottom: 0.4rem;
        }
        .da-hero__subtitle {
            color: var(--da-ink-soft);
            font-size: 0.98rem;
            max-width: 50ch;
            margin-bottom: 1.25rem;
        }
        .da-hero__fields {
            display: flex;
            flex-wrap: wrap;
            gap: 1.75rem;
            border-top: 1px dashed var(--da-line);
            padding-top: 1rem;
        }
        .da-field { display: flex; flex-direction: column; gap: 0.15rem; }
        .da-field__label {
            font-family: var(--da-font-mono);
            font-size: 0.64rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--da-ink-faint);
        }
        .da-field__value {
            font-family: var(--da-font-mono);
            font-size: 0.88rem;
            color: var(--da-ink);
            font-weight: 500;
        }

        /* ---- the signature moment: a rubber-stamp badge on completed results ---- */
        .da-stamp {
            display: inline-block;
            font-family: var(--da-font-mono);
            font-weight: 700;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--da-accent);
            border: 3px double var(--da-accent);
            border-radius: 4px;
            padding: 0.4rem 0.85rem;
            transform: rotate(-6deg);
            margin: 0.25rem 0 1.1rem 0;
            animation: da-stamp-in 0.45s cubic-bezier(.2,1.8,.4,1) both;
        }
        @keyframes da-stamp-in {
            0%   { transform: scale(2.4) rotate(-18deg); opacity: 0; }
            55%  { transform: scale(0.92) rotate(-4deg); opacity: 1; }
            100% { transform: scale(1) rotate(-6deg); opacity: 1; }
        }

        /* ---- empty state ---- */
        .da-empty {
            border: 1px dashed var(--da-line);
            border-radius: var(--da-radius);
            padding: 1.5rem;
            text-align: center;
            font-family: var(--da-font-mono);
            font-style: italic;
            font-size: 0.85rem;
            color: var(--da-ink-faint);
            background: var(--da-card);
            margin: 0.5rem 0 1rem;
        }

        /* ---- buttons ---- */
        [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {
            font-family: var(--da-font-mono) !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            border-radius: 6px !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease, color 0.15s ease, border-color 0.15s ease;
        }
        [data-testid="stBaseButton-primary"] {
            background: var(--da-accent) !important;
            border-color: var(--da-accent) !important;
            color: #fff !important;
        }
        [data-testid="stBaseButton-primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(31, 111, 92, 0.25);
        }
        [data-testid="stBaseButton-secondary"] {
            background: var(--da-card) !important;
            border: 1px solid var(--da-line) !important;
            color: var(--da-ink) !important;
        }
        [data-testid="stBaseButton-secondary"]:hover {
            border-color: var(--da-accent) !important;
            color: var(--da-accent) !important;
        }

        /* ---- inputs ---- */
        [data-testid="stTextInput"] input {
            background: var(--da-card) !important;
            border: 1px solid var(--da-line) !important;
            border-radius: 6px !important;
            color: var(--da-ink) !important;
        }
        [data-testid="stTextInput"] input:focus {
            border-color: var(--da-accent) !important;
            box-shadow: 0 0 0 1px var(--da-accent) !important;
        }
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: var(--da-card) !important;
            border-color: var(--da-line) !important;
            border-radius: 6px !important;
        }

        /* ---- file uploader ---- */
        [data-testid="stFileUploaderDropzone"] {
            background: var(--da-card) !important;
            border: 1.5px dashed var(--da-line) !important;
            border-radius: var(--da-radius) !important;
            transition: border-color 0.15s ease, background 0.15s ease;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--da-accent) !important;
            background: var(--da-accent-soft) !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] {
            font-family: var(--da-font-mono) !important;
        }

        /* ---- alerts / spinner ---- */
        [data-testid="stAlert"] {
            border-radius: 6px !important;
            font-family: var(--da-font-body) !important;
        }
        [data-testid="stSpinner"] {
            font-family: var(--da-font-mono) !important;
            font-style: italic;
            color: var(--da-ink-soft) !important;
        }

        /* ---- tabs styled as catalog-drawer tabs ---- */
        [data-testid="stTabs"] { border-bottom: 1px solid var(--da-line); }
        [data-testid="stTab"] {
            font-family: var(--da-font-mono) !important;
            font-size: 0.82rem !important;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: var(--da-ink-faint) !important;
            padding: 0.65rem 0.3rem !important;
        }
        [data-testid="stTab"][aria-selected="true"] {
            color: var(--da-accent) !important;
            font-weight: 600 !important;
            border-bottom: 2px solid var(--da-accent) !important;
        }
        [data-testid="stTab"]:nth-of-type(1)::before { content: "SUM · "; color: var(--da-brass); }
        [data-testid="stTab"]:nth-of-type(2)::before { content: "CMP · "; color: var(--da-brass); }
        [data-testid="stTab"]:nth-of-type(3)::before { content: "ASK · "; color: var(--da-brass); }
        [data-testid="stTabPanel"] { padding-top: 1.5rem; }

        /* ---- chat, styled as marginalia notes rather than bubbles ---- */
        [data-testid="stChatMessage"] {
            background: var(--da-card) !important;
            border: 1px solid var(--da-line) !important;
            border-radius: var(--da-radius) !important;
            padding: 0.9rem 1.1rem !important;
            margin-bottom: 0.75rem !important;
            box-shadow: none !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            border-left: 3px solid var(--da-accent) !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            border-left: 3px solid var(--da-brass) !important;
            background: var(--da-paper) !important;
        }
        [data-testid="stChatMessageAvatarUser"] { background: var(--da-brass) !important; }
        [data-testid="stChatMessageAvatarAssistant"] { background: var(--da-accent) !important; }
        [data-testid="stChatInput"] {
            border: 1px solid var(--da-line) !important;
            border-radius: var(--da-radius) !important;
            background: var(--da-card) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_stamp(label: str) -> None:
    st.markdown(f'<div class="da-stamp">{html.escape(label)}</div>', unsafe_allow_html=True)


def render_chip_row(items) -> None:
    chips = "".join(f'<span class="da-chip">{html.escape(x)}</span>' for x in items)
    st.markdown(f'<div class="da-chip-row">{chips}</div>', unsafe_allow_html=True)


def render_empty_state(text: str) -> None:
    st.markdown(f'<div class="da-empty">{html.escape(text)}</div>', unsafe_allow_html=True)


def render_hero(model_name: str) -> None:
    model_short = model_name.replace("google_genai:", "")
    docs = st.session_state.get("total_docs_indexed", 0)
    last_action = st.session_state.get("last_action", "—")
    st.markdown(
        f"""
        <div class="da-hero">
          <div class="da-hero__eyebrow">Reading Room · RAG-assisted review</div>
          <div class="da-hero__title">Document Assistant</div>
          <div class="da-hero__subtitle">Summarize, compare, and interrogate your files —
          indexed and searched on the fly.</div>
          <div class="da-hero__fields">
            <div class="da-field">
              <span class="da-field__label">Model</span>
              <span class="da-field__value">{html.escape(model_short)}</span>
            </div>
            <div class="da-field">
              <span class="da-field__label">Docs indexed</span>
              <span class="da-field__value">{docs}</span>
            </div>
            <div class="da-field">
              <span class="da-field__label">Last action</span>
              <span class="da-field__value">{html.escape(last_action)}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def note(n: int) -> str:
    return "" if n == 1 else "s"


# --------------------------------------------------------------------------
# Backend helpers (unchanged logic — only presentation moved elsewhere)
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_model(model_name: str):
    return init_chat_model(model_name)


def extract_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        if parts:
            return "\n".join(parts)
    return str(content)


def save_uploaded_files(uploaded_files, tmpdir: str):
    paths = []
    for uf in uploaded_files:
        path = os.path.join(tmpdir, uf.name)
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
        paths.append(path)
    return paths


def files_signature(uploaded_files) -> str:
    h = hashlib.sha256()
    for uf in uploaded_files:
        h.update(uf.name.encode())
        h.update(str(uf.size).encode())
    return h.hexdigest()


def build_vector_store(uploaded_files):
    unsupported = []
    all_chunks = []
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_uploaded_files(uploaded_files, tmpdir)
        for uf, path in zip(uploaded_files, paths):
            text = read_file(path)
            if isinstance(text, str) and "Unsupported file type" in text:
                unsupported.append(uf.name)
                continue
            all_chunks.extend(chunk_text(text))
    if unsupported:
        return None, unsupported
    vector_store = create_vector_store(all_chunks)
    return vector_store, []


def context_from_query(vector_store, query: str) -> str:
    results = semantic_search(vector_store, query)
    return "\n".join(doc.page_content for doc in results)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

inject_css()

st.session_state.setdefault("total_docs_indexed", 0)
st.session_state.setdefault("last_action", "—")
st.session_state.setdefault("chat_history", [])

# ---- Sidebar — "control plate" ----
st.sidebar.markdown('<div class="da-sidebar-title">Control Plate</div>', unsafe_allow_html=True)

model_name = st.sidebar.selectbox(
    "Model",
    MODELS,
    index=0,
    format_func=lambda m: m.replace("google_genai:", ""),
)
model = None
model_error = None
try:
    model = get_model(model_name)
except Exception as exc:  # missing/invalid API key, provider package, etc.
    model_error = str(exc)

if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    st.sidebar.warning(
        "No Google API key found in your environment (.env). Model calls will fail until one is set."
    )
elif model_error:
    st.sidebar.error(f"Couldn't initialize the model: {model_error[:200]}")

st.sidebar.markdown(
    '<div class="da-field__label" style="margin-top:1.25rem;">Supported file types</div>',
    unsafe_allow_html=True,
)
render_chip_row([f".{t}" for t in SUPPORTED_TYPES])

# ---- Hero ----
render_hero(model_name)

# ---- Tabs ----
tab_summarize, tab_compare, tab_ask = st.tabs(["Summarize", "Compare", "Ask Questions"])

# ---- Tab 1: Summarize ----------------------------------------------------
with tab_summarize:
    st.subheader("Summarize one or more files")

    files = st.file_uploader(
        "Upload files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        key="summarize_files",
    )
    query = st.text_input(
        "What would you like to know?",
        value="Summarize the key points of this document.",
        key="summarize_query",
    )

    if st.button("Run", type="primary", key="summarize_run"):
        if model is None:
            st.error("Model isn't available — check the sidebar for details.")
        elif not files:
            st.error("Please upload at least one file.")
        else:
            with st.spinner("Reading files and building context..."):
                vector_store, unsupported = build_vector_store(files)
            if unsupported:
                st.error(f"Unsupported file type(s): {', '.join(unsupported)}")
            else:
                try:
                    with st.spinner("Thinking..."):
                        ctx = context_from_query(vector_store, query)
                        response = model.invoke(f"Question: {query}\n\nContext:\n{ctx}")
                except Exception as exc:
                    st.error(f"The model call failed: {exc}")
                else:
                    st.session_state["total_docs_indexed"] += len(files)
                    st.session_state["last_action"] = f"Summarized {len(files)} file{note(len(files))}"
                    render_stamp(f"{len(files)} file{note(len(files))} indexed")
                    st.markdown("### Answer")
                    st.write(extract_text(response))

# ---- Tab 2: Compare -------------------------------------------------------
with tab_compare:
    st.subheader("Compare exactly two files")

    files2 = st.file_uploader(
        "Upload two files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        key="compare_files",
    )
    query2 = st.text_input(
        "What do you want to compare?",
        value="Compare these two files and highlight the key differences.",
        key="compare_query",
    )

    if st.button("Compare", type="primary", key="compare_run"):
        if model is None:
            st.error("Model isn't available — check the sidebar for details.")
        elif not files2 or len(files2) != 2:
            st.error("Please upload exactly two files.")
        else:
            with st.spinner("Reading files..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    paths = save_uploaded_files(files2, tmpdir)
                    text1, text2 = read_file(paths[0]), read_file(paths[1])

                unsupported = []
                if isinstance(text1, str) and "Unsupported file type" in text1:
                    unsupported.append(files2[0].name)
                if isinstance(text2, str) and "Unsupported file type" in text2:
                    unsupported.append(files2[1].name)

            if unsupported:
                st.error(f"Unsupported file type(s): {', '.join(unsupported)}")
            else:
                try:
                    with st.spinner("Comparing..."):
                        vs1 = create_vector_store(chunk_text(text1))
                        vs2 = create_vector_store(chunk_text(text2))
                        ctx1 = context_from_query(vs1, query2)
                        ctx2 = context_from_query(vs2, query2)
                        response = model.invoke(
                            f"Question: {query2}\n\n"
                            f"File 1 ({files2[0].name}) Context:\n{ctx1}\n\n"
                            f"File 2 ({files2[1].name}) Context:\n{ctx2}\n\n"
                            f"Compare the two files based on the question above."
                        )
                except Exception as exc:
                    st.error(f"The model call failed: {exc}")
                else:
                    st.session_state["total_docs_indexed"] += 2
                    st.session_state["last_action"] = "Compared 2 files"
                    render_stamp("2 files compared")
                    st.markdown("### Comparison")
                    st.write(extract_text(response))

# ---- Tab 3: Ask Questions (chat) ------------------------------------------
with tab_ask:
    st.subheader("Ask questions — with or without files")

    ask_files = st.file_uploader(
        "Optionally upload files for grounded Q&A (leave empty for general chat)",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        key="ask_files",
    )

    col_a, _ = st.columns([1, 5])
    with col_a:
        if st.button("↺  New chat"):
            st.session_state.pop("chat_history", None)
            st.session_state.pop("ask_vector_store", None)
            st.session_state.pop("ask_files_sig", None)
            st.rerun()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    unsupported_ask = []
    if ask_files:
        sig = files_signature(ask_files)
        if st.session_state.get("ask_files_sig") != sig:
            with st.spinner("Indexing files..."):
                vs, unsupported_ask = build_vector_store(ask_files)
            if not unsupported_ask:
                st.session_state.ask_vector_store = vs
                st.session_state.ask_files_sig = sig
                st.session_state["total_docs_indexed"] += len(ask_files)
                render_stamp(f"{len(ask_files)} file{note(len(ask_files))} indexed")
    else:
        st.session_state.pop("ask_vector_store", None)
        st.session_state.pop("ask_files_sig", None)

    if unsupported_ask:
        st.error(f"Unsupported file type(s): {', '.join(unsupported_ask)}")

    if not st.session_state.chat_history:
        render_empty_state("No conversation yet — ask a question below, with or without files.")

    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(text)

    user_msg = st.chat_input("Type your question...", disabled=model is None)
    if user_msg and model is None:
        st.error("Model isn't available — check the sidebar for details.")
    elif user_msg:
        st.session_state.chat_history.append(("user", user_msg))
        with st.chat_message("user"):
            st.write(user_msg)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    vector_store = st.session_state.get("ask_vector_store")
                    if vector_store is not None:
                        ctx = context_from_query(vector_store, user_msg)
                        prompt = (
                            f"Question: {user_msg}\n\nContext:\n{ctx}\n\n"
                            f"Answer clearly and concisely."
                        )
                    else:
                        prompt = f"Question: {user_msg}\n\nAnswer clearly and concisely."
                    response = model.invoke(prompt)
                    answer = extract_text(response)
            except Exception as exc:
                answer = f"⚠️ The model call failed: {exc}"
            st.write(answer)
        st.session_state.chat_history.append(("assistant", answer))
        st.session_state["last_action"] = "Answered a question"