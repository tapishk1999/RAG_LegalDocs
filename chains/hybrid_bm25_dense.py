"""
Hybrid BM25 + Dense RAG
───────────────────────
Strategy : Legal-aware recursive chunking + BM25 (sparse) + Dense vector
           (Chroma w/ MMR), combined via EnsembleRetriever.

           Three design choices, tuned against the RAGAS baseline:
           1. Micro-chunks (< MIN_BM25_CHARS) are dropped from the BM25 corpus.
              Section titles and chapter headings (~30–60 chars) inflate BM25
              scores and shove real provision bodies out of the top-k.
           2. Ensemble weights 0.75 dense / 0.25 BM25, dense k = 8.
           3. Per-act routing: when a question names an act, retrieval is
              dispatched to a per-act ensemble (filtered Chroma + BM25 over
              that act's chunks only). Prevents cross-act contamination.

Retrieval : Per-act ensemble of BM25 + MMR vector search
Model     : GPT-4o-mini
"""

import os
from typing import List, Optional, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


CHAIN_NAME = "Hybrid BM25 + Dense"
COLLECTION_NAME = "lexquery_hybrid"
PERSIST_DIR = "./vectorstore/hybrid_bm25_dense"

# Drop chunks shorter than this from BM25 (titles/headings inflate BM25 scores)
MIN_BM25_CHARS = 150

# Ensemble weights and k values (tuned 2026-05-09 against RAGAS baseline)
DENSE_WEIGHT = 0.75
BM25_WEIGHT = 0.25
DENSE_K = 8
DENSE_FETCH_K = 24
BM25_K = 5

LEGAL_SEPARATORS = [
    "\nCHAPTER ",
    "\nSECTION ",
    "\nSection ",
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]

PROMPT_TEMPLATE = """You are a legal assistant specializing in Indian law.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I could not find relevant information in the provided Acts."
Always cite the Act name and section number when referencing a specific provision.

Context:
{context}

Question: {question}

Answer:"""


def _format_docs(docs: List[Document]) -> str:
    parts = []
    seen = set()
    for doc in docs:
        # Deduplicate on (act_name, section_id) — tighter than content hash,
        # avoids whitespace/boundary mismatches between BM25 and dense results.
        meta = doc.metadata
        key = (meta.get("act_name", ""), meta.get("section_id", ""))
        if key in seen:
            continue
        seen.add(key)

        act = meta.get("act_name", "Unknown Act")
        sec = meta.get("section_id", "")
        chapter = meta.get("chapter_title", "")
        header = f"[{act} | Chapter: {chapter} | Section {sec}]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _detect_act_filter(query: str) -> Optional[str]:
    """
    Heuristic: if query mentions a specific act, return its act_name string
    so retrieval can be filtered to that act's documents.
    """
    query_lower = query.lower()
    if "copyright" in query_lower:
        return "The Copyright Act, 1957"
    if "muslim" in query_lower or "marriage" in query_lower or "talaq" in query_lower:
        return "The Muslim Women (Protection of Rights on Marriage) Act, 2019"
    if "tribunal" in query_lower:
        return "The Tribunals Reforms Act, 2021"
    return None


def _build_bm25(chunks: List[Document], k: int) -> BM25Retriever:
    r = BM25Retriever.from_documents(chunks)
    r.k = k
    return r


def build_hybrid_chain(documents: List[Document], openai_api_key: str, force_rebuild: bool = False):
    """
    Build the Hybrid BM25 + Dense chain.
    Returns (chain, retriever) where retriever is a Runnable that accepts a
    query string and returns a list of Documents (compatible with eval_suite).
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key,
    )

    # ── Chunking ──
    splitter = RecursiveCharacterTextSplitter(
        separators=LEGAL_SEPARATORS,
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # ── BM25 corpus: drop micro-chunks (titles/headings) ──
    long_chunks = [c for c in chunks if len(c.page_content) >= MIN_BM25_CHARS]
    print(
        f"[Hybrid] BM25 corpus: {len(long_chunks)} chunks "
        f"({len(chunks) - len(long_chunks)} micro-chunks dropped)"
    )

    # ── Dense vector store (indexes ALL chunks) ──
    if os.path.exists(PERSIST_DIR) and not force_rebuild:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
        )

    # ── Default (no-act-filter) ensemble ──
    default_dense = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": DENSE_K, "fetch_k": DENSE_FETCH_K},
    )
    default_bm25 = _build_bm25(long_chunks, k=BM25_K)
    default_ensemble = EnsembleRetriever(
        retrievers=[default_dense, default_bm25],
        weights=[DENSE_WEIGHT, BM25_WEIGHT],
    )

    # ── Per-act ensembles (pre-built once, routed by _detect_act_filter) ──
    per_act_ensembles: Dict[str, EnsembleRetriever] = {}
    acts = sorted({c.metadata.get("act_name", "") for c in long_chunks if c.metadata.get("act_name")})
    for act in acts:
        dense_filtered = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": DENSE_K,
                "fetch_k": DENSE_FETCH_K,
                "filter": {"act_name": act},
            },
        )
        act_chunks = [c for c in long_chunks if c.metadata.get("act_name") == act]
        if not act_chunks:
            continue
        bm25_filtered = _build_bm25(act_chunks, k=BM25_K)
        per_act_ensembles[act] = EnsembleRetriever(
            retrievers=[dense_filtered, bm25_filtered],
            weights=[DENSE_WEIGHT, BM25_WEIGHT],
        )
        print(f"[Hybrid] Built per-act retriever for: {act} ({len(act_chunks)} chunks)")

    # ── Routing retriever: if query names an act, use the per-act ensemble ──
    def smart_retrieve(query: str) -> List[Document]:
        target_act = _detect_act_filter(query)
        if target_act and target_act in per_act_ensembles:
            return per_act_ensembles[target_act].invoke(query)
        return default_ensemble.invoke(query)

    routing_retriever = RunnableLambda(smart_retrieve)

    # ── LLM ──
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=openai_api_key,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": routing_retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, routing_retriever
