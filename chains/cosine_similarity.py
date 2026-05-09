"""
Cosine Similarity (Dense) RAG
─────────────────────────────
Strategy : CharacterTextSplitter (1000 chars / 200 overlap)
Retrieval : Cosine similarity over OpenAI text-embedding-3-small (top-k = 4)
Model     : GPT-4o-mini
Purpose   : Baseline naive RAG. Single-modality dense vector retrieval.
"""

import os
from typing import List

from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


CHAIN_NAME = "Cosine Similarity (Dense)"
COLLECTION_NAME = "lexquery_cosine"
PERSIST_DIR = "./vectorstore/cosine_similarity"

PROMPT_TEMPLATE = """You are a legal assistant specializing in Indian law.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I could not find relevant information in the provided Acts."
Cite the Act name and section number when possible.

Context:
{context}

Question: {question}

Answer:"""


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        meta = doc.metadata
        header = f"[{meta.get('act_name','Unknown Act')} — Section {meta.get('section_id','')}]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_cosine_chain(documents: List[Document], openai_api_key: str, force_rebuild: bool = False):
    """
    Build the Cosine Similarity (Dense) chain.
    Persists the Chroma vector store to disk so it can be reloaded.
    Returns (chain, retriever).
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key,
    )

    # ── Chunking ──
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n",
    )
    chunks = splitter.split_documents(documents)

    # ── Vector Store ──
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

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    # ── LLM ──
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=openai_api_key,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever
