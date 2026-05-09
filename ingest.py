"""
ingest.py — Run once to build both vector stores from the PDF documents.

Usage:
    python ingest.py

Requires OPENAI_API_KEY in .env or environment.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from ingestion.loader import load_all_pdfs
from chains.cosine_similarity import build_cosine_chain, CHAIN_NAME as COSINE_NAME
from chains.hybrid_bm25_dense import build_hybrid_chain, CHAIN_NAME as HYBRID_NAME


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Add it to .env or set as environment variable.")

    pdf_dir = Path(__file__).parent
    print(f"Loading PDFs from: {pdf_dir}\n")

    documents = load_all_pdfs(str(pdf_dir))

    print(f"\n── Building {COSINE_NAME} vector store ──")
    build_cosine_chain(documents, api_key, force_rebuild=True)
    print(f"✓ {COSINE_NAME} ready")

    print(f"\n── Building {HYBRID_NAME} vector store ──")
    build_hybrid_chain(documents, api_key, force_rebuild=True)
    print(f"✓ {HYBRID_NAME} ready")

    print("\n✅ Ingestion complete. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
