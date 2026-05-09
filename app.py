"""
LexQuery — Indian Legal Document AI Assistant
Streamlit chat interface comparing two retrieval strategies.

Run locally:
    streamlit run app.py

Deployed on Streamlit Community Cloud:
    Set OPENAI_API_KEY in App Settings → Secrets (TOML format).
"""

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Legal Document AI Assistant",
    page_icon="⚖️",
    layout="wide",
)


def _resolve_api_key() -> str:
    """API key from Streamlit secrets first, then env var. No user input."""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")


api_key = _resolve_api_key()


# ── Load chains (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building vector stores… (first run only — takes ~1 min)")
def load_chains(openai_api_key: str):
    from ingestion.loader import load_all_pdfs
    from chains.cosine_similarity import build_cosine_chain
    from chains.hybrid_bm25_dense import build_hybrid_chain

    pdf_dir = str(Path(__file__).parent)
    documents = load_all_pdfs(pdf_dir)

    c1, r1 = build_cosine_chain(documents, openai_api_key)
    c2, r2 = build_hybrid_chain(documents, openai_api_key)

    chains_dict = {
        "Cosine Similarity (Dense)": (c1, r1),
        "Hybrid BM25 + Dense ⭐": (c2, r2),
    }
    # Distinct acts in the loaded corpus, for the sidebar listing
    loaded_acts = sorted({
        d.metadata.get("act_name", "Unknown Act") for d in documents
    })
    return chains_dict, loaded_acts


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ Indian Legal Document AI")
    st.markdown("**RAG over Indian parliamentary acts**")
    st.divider()

    chain_label = st.radio(
        "Retrieval Strategy",
        options=[
            "Cosine Similarity (Dense)",
            "Hybrid BM25 + Dense ⭐",
        ],
        index=1,
        help=(
            "Cosine Similarity: character chunking + single-modality dense vector search.\n"
            "Hybrid BM25 + Dense: legal-aware chunking + per-act ensemble of "
            "BM25 and dense MMR retrieval. Best context recall."
        ),
    )

    st.divider()

    # Loaded acts — populated dynamically from the document metadata so new
    # PDFs added to the project root show up here automatically.
    st.markdown("**Loaded Acts**")
    if api_key:
        try:
            _, loaded_acts = load_chains(api_key)
            for act in loaded_acts:
                st.markdown(f"- 📄 {act}")
        except Exception:
            st.caption("_Could not load act list._")
    else:
        st.caption("_Configure OPENAI_API_KEY to load acts._")

    st.divider()
    st.caption("Built with LangChain · ChromaDB · OpenAI · RAGAS")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("⚖️ Indian Legal Document AI Assistant")
st.markdown(
    "Ask natural-language questions about Indian parliamentary acts. "
    "Answers are grounded in the source text with citations to the Act and section."
)

if not api_key:
    st.error(
        "OpenAI API key not configured. Set `OPENAI_API_KEY` in Streamlit secrets "
        "(App Settings → Secrets) or in a local `.env` file."
    )
    st.stop()

try:
    chains, _ = load_chains(api_key)
except Exception as e:
    st.error(f"Failed to build chains: {e}")
    st.stop()

chain, retriever = chains[chain_label]

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render existing messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['act']}** · Section {src['section']} "
                        f"*(Chapter: {src['chapter']})*"
                    )
                    st.caption(src["snippet"])
        if msg.get("latency"):
            st.caption(f"⏱ {msg['latency']:.2f}s · {chain_label}")


def _process_question(question: str) -> None:
    """Append user message, run the chain, render assistant reply with sources."""
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching legal documents…"):
            t0 = time.time()
            try:
                answer = chain.invoke(question)
                latency = time.time() - t0

                try:
                    source_docs = retriever.invoke(question)
                except Exception:
                    source_docs = []

                sources = []
                seen = set()
                for doc in source_docs:
                    m = doc.metadata
                    key = (m.get("act_name", ""), m.get("section_id", ""))
                    if key not in seen:
                        seen.add(key)
                        sources.append(
                            {
                                "act": m.get("act_name", "Unknown"),
                                "section": m.get("section_id", "—"),
                                "chapter": m.get("chapter_title", "—"),
                                "snippet": doc.page_content[:200] + "…",
                            }
                        )

                st.markdown(answer)

                with st.expander("📚 Sources", expanded=False):
                    for src in sources:
                        st.markdown(
                            f"**{src['act']}** · Section {src['section']} "
                            f"*(Chapter: {src['chapter']})*"
                        )
                        st.caption(src["snippet"])

                st.caption(f"⏱ {latency:.2f}s · {chain_label}")

                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "latency": latency,
                    }
                )

            except Exception as e:
                st.error(f"Error: {e}")


# ── Question intake: chat_input OR pending sample-question from a button click ──
typed_question = st.chat_input("Ask a question about Indian law…")
pending_question = st.session_state.pop("pending_question", None)
question = typed_question or pending_question

if question:
    _process_question(question)

# ── Sample questions (only when conversation is empty) ────────────────────────
if not st.session_state["messages"]:
    st.divider()
    st.markdown("**Try these questions:**")
    cols = st.columns(3)
    sample_qs = [
        "What is the punishment for triple talaq under the Muslim Women Act?",
        "What rights does a copyright holder have?",
        "What is the term of office for tribunal members?",
        "What does the Farm Laws Repeal Act repeal?",
        "Which countries are mentioned in the Citizenship (Amendment) Act?",
        "What qualifications are required to be a tribunal chairperson?",
    ]
    for i, q in enumerate(sample_qs):
        with cols[i % 3]:
            if st.button(q, use_container_width=True, key=f"sample_{i}"):
                # Store the question and rerun so it gets picked up at the
                # top of the script and processed via _process_question().
                st.session_state["pending_question"] = q
                st.rerun()
