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
        @import url('[fonts.googleapis.com](https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap)');

        :root {
            --da-ink: #F5F1FF;
            --da-ink-soft: #B8B2D6;
            --da-ink-faint: #827C9F;
            --da-paper: #050612;
            --da-card: rgba(15, 17, 40, 0.88);
            --da-card-solid: #0F1128;
            --da-line: rgba(151, 126, 255, 0.24);
            --da-accent: #8C7BFF;
            --da-accent-bright: #B7A9FF;
            --da-accent-soft: rgba(140, 123, 255, 0.12);
            --da-cyan: #4DE8FF;
            --da-pink: #FF69D4;
            --da-brass: #F5C76B;
            --da-radius: 14px;
            --da-font-display: 'Space Grotesk', sans-serif;
            --da-font-body: 'Space Grotesk', sans-serif;
            --da-font-mono: 'Space Mono', monospace;
            --da-glow: 0 0 24px rgba(140, 123, 255, 0.18);
        }

        /* ---- cosmic canvas ---- */
        [data-testid="stAppViewContainer"] {
            position: relative;
            color: var(--da-ink) !important;
            font-family: var(--da-font-body) !important;
            background:
                radial-gradient(circle at 18% 8%, rgba(79, 91, 255, 0.20), transparent 30rem),
                radial-gradient(circle at 85% 20%, rgba(255, 105, 212, 0.12), transparent 26rem),
                radial-gradient(circle at 65% 90%, rgba(77, 232, 255, 0.10), transparent 30rem),
                linear-gradient(145deg, #050612 0%, #090A1D 48%, #050612 100%) !important;
        }

        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            opacity: 0.55;
            background-image:
                radial-gradient(circle, rgba(255,255,255,0.95) 0 1px, transparent 1.3px),
                radial-gradient(circle, rgba(140,123,255,0.8) 0 1px, transparent 1.4px),
                radial-gradient(circle, rgba(77,232,255,0.65) 0 1px, transparent 1.3px);
            background-position: 0 0, 35px 75px, 115px 30px;
            background-size: 150px 150px, 220px 220px, 285px 285px;
            animation: da-stars-drift 45s linear infinite;
        }

        [data-testid="stAppViewContainer"] > * {
            position: relative;
            z-index: 1;
        }

        @keyframes da-stars-drift {
            from { background-position: 0 0, 35px 75px, 115px 30px; }
            to { background-position: 0 150px, 255px 295px, 400px 315px; }
        }

        [data-testid="stHeader"] {
            background: rgba(5, 6, 18, 0.72) !important;
            border-bottom: 1px solid var(--da-line);
            backdrop-filter: blur(14px);
        }

        [data-testid="stToolbar"] {
            color: var(--da-ink) !important;
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 940px;
            padding-top: 2.25rem;
            padding-bottom: 4rem;
        }

        /* ---- typography ---- */
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
            color: var(--da-ink);
        }

        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4 {
            font-family: var(--da-font-display) !important;
            font-weight: 600 !important;
            color: var(--da-ink) !important;
            letter-spacing: -0.02em;
        }

        [data-testid="stMarkdownContainer"] p {
            line-height: 1.7;
        }

        [data-testid="stMarkdownContainer"] a {
            color: var(--da-cyan);
            text-decoration-color: rgba(77, 232, 255, 0.4);
        }

        [data-testid="stMarkdownContainer"] code {
            font-family: var(--da-font-mono);
            color: var(--da-cyan);
            background: rgba(5, 6, 18, 0.78);
            border: 1px solid var(--da-line);
            border-radius: 5px;
            padding: 0.1rem 0.35rem;
        }

        [data-testid="stCaptionContainer"] {
            color: var(--da-ink-faint) !important;
            font-family: var(--da-font-mono) !important;
            letter-spacing: 0.02em;
        }

        /* ---- sidebar ---- */
        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 25% 5%, rgba(140, 123, 255, 0.15), transparent 18rem),
                rgba(7, 8, 24, 0.94) !important;
            border-right: 1px solid var(--da-line);
            backdrop-filter: blur(18px);
        }

        [data-testid="stSidebarContent"] {
            padding-top: 1.5rem;
        }

        .da-sidebar-title {
            font-family: var(--da-font-mono);
            font-size: 0.7rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--da-cyan);
            border-bottom: 1px solid var(--da-line);
            padding-bottom: 0.65rem;
            margin-bottom: 1.1rem;
            text-shadow: 0 0 14px rgba(77, 232, 255, 0.5);
        }

        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            font-family: var(--da-font-mono) !important;
            font-size: 0.68rem !important;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            color: var(--da-ink-soft) !important;
        }

        /* ---- chips ---- */
        .da-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.55rem;
        }

        .da-chip {
            font-family: var(--da-font-mono);
            font-size: 0.64rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: var(--da-cyan);
            background: rgba(77, 232, 255, 0.06);
            border: 1px solid rgba(77, 232, 255, 0.25);
            border-radius: 999px;
            padding: 0.2rem 0.6rem;
            box-shadow: inset 0 0 12px rgba(77, 232, 255, 0.04);
        }

        /* ---- cosmic hero ---- */
        .da-hero {
            position: relative;
            overflow: hidden;
            isolation: isolate;
            background:
                linear-gradient(125deg, rgba(20, 22, 53, 0.96), rgba(10, 11, 30, 0.92));
            border: 1px solid rgba(151, 126, 255, 0.34);
            border-top: 2px solid var(--da-accent-bright);
            border-radius: var(--da-radius);
            padding: 2rem 2.1rem 1.6rem;
            margin-bottom: 1.75rem;
            box-shadow:
                var(--da-glow),
                inset 0 1px 0 rgba(255,255,255,0.05);
            backdrop-filter: blur(14px);
        }

        .da-hero::before {
            content: "";
            position: absolute;
            z-index: -2;
            width: 260px;
            height: 260px;
            right: -80px;
            top: -110px;
            border-radius: 50%;
            background:
                radial-gradient(circle at 35% 35%,
                    rgba(255,255,255,0.85) 0 1%,
                    var(--da-accent) 3%,
                    rgba(140,123,255,0.3) 24%,
                    rgba(140,123,255,0.05) 54%,
                    transparent 70%);
            filter: blur(1px);
            box-shadow: 0 0 65px rgba(140, 123, 255, 0.2);
        }

        .da-hero::after {
            content: "";
            position: absolute;
            z-index: -1;
            width: 330px;
            height: 110px;
            right: -115px;
            top: 8px;
            border: 1px solid rgba(77, 232, 255, 0.2);
            border-radius: 50%;
            transform: rotate(-18deg);
        }

        .da-hero__eyebrow {
            font-family: var(--da-font-mono);
            font-size: 0.67rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: var(--da-cyan);
            margin-bottom: 0.65rem;
            text-shadow: 0 0 14px rgba(77, 232, 255, 0.45);
        }

        .da-hero__title {
            font-family: var(--da-font-display);
            font-weight: 700;
            font-size: clamp(2.25rem, 7vw, 3.35rem);
            color: var(--da-ink);
            line-height: 1.05;
            letter-spacing: -0.045em;
            margin-bottom: 0.6rem;
            background: linear-gradient(95deg, #FFFFFF, #CFC7FF 52%, #7BEFFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .da-hero__subtitle {
            color: var(--da-ink-soft);
            font-size: 1rem;
            line-height: 1.65;
            max-width: 52ch;
            margin-bottom: 1.35rem;
        }

        .da-hero__fields {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem 2.25rem;
            border-top: 1px solid var(--da-line);
            padding-top: 1rem;
        }

        .da-field {
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
            min-width: 110px;
        }

        .da-field__label {
            font-family: var(--da-font-mono);
            font-size: 0.61rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: var(--da-ink-faint);
        }

        .da-field__value {
            font-family: var(--da-font-mono);
            font-size: 0.84rem;
            color: var(--da-ink);
            font-weight: 700;
        }

        /* ---- completed-result signal ---- */
        .da-stamp {
            display: inline-block;
            font-family: var(--da-font-mono);
            font-weight: 700;
            font-size: 0.69rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: var(--da-cyan);
            background: rgba(77, 232, 255, 0.05);
            border: 1px solid var(--da-cyan);
            border-radius: 999px;
            padding: 0.45rem 0.9rem;
            margin: 0.25rem 0 1.1rem;
            box-shadow:
                0 0 16px rgba(77, 232, 255, 0.25),
                inset 0 0 12px rgba(77, 232, 255, 0.08);
            animation: da-stamp-in 0.5s cubic-bezier(.2,1.5,.4,1) both;
        }

        .da-stamp::before {
            content: "✦";
            margin-right: 0.5rem;
        }

        @keyframes da-stamp-in {
            0% {
                transform: scale(1.5);
                opacity: 0;
                filter: blur(4px);
            }
            60% {
                transform: scale(0.96);
                opacity: 1;
                filter: blur(0);
            }
            100% {
                transform: scale(1);
                opacity: 1;
            }
        }

        /* ---- empty state ---- */
        .da-empty {
            border: 1px dashed rgba(140, 123, 255, 0.38);
            border-radius: var(--da-radius);
            padding: 1.65rem;
            text-align: center;
            font-family: var(--da-font-mono);
            font-size: 0.81rem;
            color: var(--da-ink-faint);
            background:
                radial-gradient(circle at center, rgba(140,123,255,0.08), transparent 70%),
                var(--da-card);
            box-shadow: inset 0 0 30px rgba(140, 123, 255, 0.035);
            margin: 0.5rem 0 1rem;
        }

        .da-empty::before {
            content: "✦";
            display: block;
            margin-bottom: 0.55rem;
            color: var(--da-accent-bright);
            font-style: normal;
            font-size: 1.2rem;
            text-shadow: 0 0 16px var(--da-accent);
        }

        /* ---- buttons ---- */
        [data-testid="stBaseButton-primary"],
        [data-testid="stBaseButton-secondary"] {
            font-family: var(--da-font-mono) !important;
            font-size: 0.73rem !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            border-radius: 8px !important;
            transition:
                transform 0.15s ease,
                box-shadow 0.15s ease,
                color 0.15s ease,
                border-color 0.15s ease;
        }

        [data-testid="stBaseButton-primary"] {
            background: linear-gradient(100deg, #6658E8, var(--da-accent)) !important;
            border-color: var(--da-accent) !important;
            color: #FFFFFF !important;
            box-shadow: 0 0 16px rgba(140, 123, 255, 0.18);
        }

        [data-testid="stBaseButton-primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(140, 123, 255, 0.34);
        }

        [data-testid="stBaseButton-secondary"] {
            background: rgba(15, 17, 40, 0.78) !important;
            border: 1px solid var(--da-line) !important;
            color: var(--da-ink) !important;
        }

        [data-testid="stBaseButton-secondary"]:hover {
            border-color: var(--da-cyan) !important;
            color: var(--da-cyan) !important;
            box-shadow: 0 0 16px rgba(77, 232, 255, 0.12);
        }

        /* ---- inputs ---- */
        [data-testid="stTextInput"] input {
            background: rgba(10, 11, 30, 0.88) !important;
            border: 1px solid var(--da-line) !important;
            border-radius: 8px !important;
            color: var(--da-ink) !important;
        }

        [data-testid="stTextInput"] input::placeholder {
            color: var(--da-ink-faint) !important;
        }

        [data-testid="stTextInput"] input:focus {
            border-color: var(--da-cyan) !important;
            box-shadow: 0 0 0 1px var(--da-cyan), 0 0 16px rgba(77,232,255,0.12) !important;
        }

        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: rgba(10, 11, 30, 0.88) !important;
            border-color: var(--da-line) !important;
            border-radius: 8px !important;
            color: var(--da-ink) !important;
        }

        div[data-baseweb="popover"],
        div[data-baseweb="menu"] {
            background: var(--da-card-solid) !important;
            color: var(--da-ink) !important;
        }

        /* ---- file uploader ---- */
        [data-testid="stFileUploaderDropzone"] {
            background:
                linear-gradient(135deg, rgba(140,123,255,0.06), rgba(77,232,255,0.03)),
                rgba(10, 11, 30, 0.82) !important;
            border: 1px dashed rgba(140, 123, 255, 0.42) !important;
            border-radius: var(--da-radius) !important;
            transition:
                border-color 0.15s ease,
                background 0.15s ease,
                box-shadow 0.15s ease;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--da-cyan) !important;
            background: rgba(77, 232, 255, 0.06) !important;
            box-shadow: inset 0 0 28px rgba(77, 232, 255, 0.05);
        }

        [data-testid="stFileUploaderDropzoneInstructions"] {
            font-family: var(--da-font-mono) !important;
            color: var(--da-ink-soft) !important;
        }

        /* ---- alerts and spinner ---- */
        [data-testid="stAlert"] {
            border: 1px solid var(--da-line) !important;
            border-radius: 9px !important;
            color: var(--da-ink) !important;
            font-family: var(--da-font-body) !important;
            background: rgba(15, 17, 40, 0.9) !important;
        }

        [data-testid="stSpinner"] {
            font-family: var(--da-font-mono) !important;
            color: var(--da-cyan) !important;
        }

        /* ---- tabs ---- */
        [data-testid="stTabs"] {
            border-bottom: 1px solid var(--da-line);
        }

        [data-testid="stTab"] {
            font-family: var(--da-font-mono) !important;
            font-size: 0.77rem !important;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: var(--da-ink-faint) !important;
            padding: 0.7rem 0.35rem !important;
        }

        [data-testid="stTab"][aria-selected="true"] {
            color: var(--da-cyan) !important;
            font-weight: 700 !important;
            border-bottom: 2px solid var(--da-cyan) !important;
            text-shadow: 0 0 14px rgba(77, 232, 255, 0.4);
        }

        [data-testid="stTab"]:nth-of-type(1)::before {
            content: "SUM · ";
            color: var(--da-accent-bright);
        }

        [data-testid="stTab"]:nth-of-type(2)::before {
            content: "CMP · ";
            color: var(--da-accent-bright);
        }

        [data-testid="stTab"]:nth-of-type(3)::before {
            content: "ASK · ";
            color: var(--da-accent-bright);
        }

        [data-testid="stTabPanel"] {
            padding-top: 1.5rem;
        }

        /* ---- chat as research transmissions ---- */
        [data-testid="stChatMessage"] {
            background: rgba(15, 17, 40, 0.86) !important;
            border: 1px solid var(--da-line) !important;
            border-radius: var(--da-radius) !important;
            padding: 0.95rem 1.15rem !important;
            margin-bottom: 0.8rem !important;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12) !important;
            backdrop-filter: blur(10px);
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            border-left: 3px solid var(--da-cyan) !important;
            box-shadow:
                -5px 0 18px rgba(77, 232, 255, 0.06),
                0 8px 30px rgba(0, 0, 0, 0.12) !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            border-left: 3px solid var(--da-pink) !important;
            background: rgba(20, 18, 43, 0.88) !important;
        }

        [data-testid="stChatMessageAvatarUser"] {
            background: linear-gradient(135deg, #A93F92, var(--da-pink)) !important;
        }

        [data-testid="stChatMessageAvatarAssistant"] {
            background: linear-gradient(135deg, #347E99, var(--da-cyan)) !important;
        }

        [data-testid="stChatInput"] {
            border: 1px solid var(--da-line) !important;
            border-radius: var(--da-radius) !important;
            background: rgba(10, 11, 30, 0.92) !important;
            box-shadow: 0 0 24px rgba(140, 123, 255, 0.08);
        }

        /* ---- reduced motion and responsive layout ---- */
        @media (prefers-reduced-motion: reduce) {
            [data-testid="stAppViewContainer"]::before,
            .da-stamp {
                animation: none !important;
            }

            [data-testid="stBaseButton-primary"],
            [data-testid="stBaseButton-secondary"] {
                transition: none !important;
            }
        }

        @media (max-width: 640px) {
            [data-testid="stMainBlockContainer"] {
                padding-top: 1.25rem;
            }

            .da-hero {
                padding: 1.5rem 1.25rem 1.3rem;
            }

            .da-hero__title {
                font-size: 2.35rem;
            }

            .da-hero__fields {
                gap: 1rem 1.5rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_stamp(label: str) -> None:
    st.markdown(
        f'<div class="da-stamp">{html.escape(label)}</div>',
        unsafe_allow_html=True,
    )


def render_chip_row(items) -> None:
    chips = "".join(
        f'<span class="da-chip">{html.escape(str(item))}</span>'
        for item in items
    )
    st.markdown(
        f'<div class="da-chip-row">{chips}</div>',
        unsafe_allow_html=True,
    )


def render_empty_state(text: str) -> None:
    st.markdown(
        f'<div class="da-empty">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def render_hero(model_name: str) -> None:
    model_short = model_name.replace("google_genai:", "")
    docs = st.session_state.get("total_docs_indexed", 0)
    last_action = st.session_state.get("last_action", "-")

    st.markdown(
        f"""
        <div class="da-hero">
          <div class="da-hero__eyebrow">
            Research Orbit · RAG-assisted discovery
          </div>
          <div class="da-hero__title">Researcher</div>
          <div class="da-hero__subtitle">
            Summarize, compare, and interrogate your files -
            indexed and searched across your knowledge universe.
          </div>
          <div class="da-hero__fields">
            <div class="da-field">
              <span class="da-field__label">Model</span>
              <span class="da-field__value">
                {html.escape(model_short)}
              </span>
            </div>
            <div class="da-field">
              <span class="da-field__label">Docs indexed</span>
              <span class="da-field__value">{docs}</span>
            </div>
            <div class="da-field">
              <span class="da-field__label">Last action</span>
              <span class="da-field__value">
                {html.escape(str(last_action))}
              </span>
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
def get_google_api_key() -> str:
    """Return the Gemini key, preferring the provider-specific variable."""
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


@st.cache_resource(show_spinner=False)
def get_model(model_name: str):
    api_key = get_google_api_key()
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY in .env before using the model.")
    os.environ["GOOGLE_API_KEY"] = api_key
    return init_chat_model(model_name, api_key=api_key)


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

if not get_google_api_key():
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
                answer = f"The model call failed: {exc}"
            st.write(answer)
        st.session_state.chat_history.append(("assistant", answer))
        st.session_state["last_action"] = "Answered a question"