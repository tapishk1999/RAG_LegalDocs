"""
LexQuery — Indian Legal Document RAG
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
    page_title="LexQuery — Indian Legal RAG",
    page_icon="⚖️",
    layout="wide",
)


def _resolve_api_key(user_input: str) -> str:
    """Prefer (in order): user-typed key → Streamlit secrets → env var."""
    if user_input:
        return user_input
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ LexQuery")
    st.markdown("**Indian Legal Document RAG**")
    st.divider()

    # API Key input — falls back to st.secrets / env if blank
    typed_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value="",
        help="Your OpenAI API key (sk-…). Leave blank to use the deployment's secret.",
    )
    api_key = _resolve_api_key(typed_key)

    st.divider()

    # Retrieval strategy selector
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
    st.markdown("**Loaded Acts**")
    st.markdown("- 📄 The Copyright Act, 1957")
    st.markdown("- 📄 Muslim Women Act, 2019")
    st.markdown("- 📄 Tribunals Reforms Act, 2021")

    st.divider()
    if st.button("🔄 Re-ingest Documents", use_container_width=True):
        st.cache_resource.clear()
        st.session_state["messages"] = []
        st.rerun()

    st.divider()
    st.caption("Built with LangChain · ChromaDB · OpenAI · RAGAS")


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

    return {
        "Cosine Similarity (Dense)": (c1, r1),
        "Hybrid BM25 + Dense ⭐": (c2, r2),
    }


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("⚖️ LexQuery — Indian Legal Document Assistant")
st.markdown(
    "Ask questions about the **Copyright Act 1957**, "
    "**Muslim Women (Protection of Rights on Marriage) Act 2019**, "
    "or the **Tribunals Reforms Act 2021**."
)

if not api_key:
    st.warning(
        "👈 Enter your OpenAI API key in the sidebar (or configure `OPENAI_API_KEY` "
        "in Streamlit secrets) to get started."
    )
    st.stop()

# Load chains
try:
    chains = load_chains(api_key)
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

# ── Chat input ────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about Indian law…"):
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

# ── Sample questions ──────────────────────────────────────────────────────────
if not st.session_state["messages"]:
    st.divider()
    st.markdown("**Try these questions:**")
    cols = st.columns(3)
    sample_qs = [
        "What is the punishment for triple talaq under the Muslim Women Act?",
        "What rights does a copyright holder have under the Copyright Act?",
        "What is the term of office for members of a tribunal under the Reforms Act?",
        "Can a husband be imprisoned for pronouncing instant triple talaq?",
        "How is copyright infringement defined and what are its penalties?",
        "What qualifications are required to be appointed as a tribunal chairperson?",
    ]
    for i, q in enumerate(sample_qs):
        with cols[i % 3]:
            if st.button(q, use_container_width=True, key=f"sample_{i}"):
                st.session_state["messages"].append({"role": "user", "content": q})
                st.rerun()
