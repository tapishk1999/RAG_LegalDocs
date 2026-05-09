# ⚖️ LexQuery — Indian Legal Document RAG

A production-style Retrieval-Augmented Generation (RAG) system for querying Indian parliamentary acts. Compares two retrieval techniques side-by-side under a RAGAS evaluation harness.

## What It Does

LexQuery answers natural-language questions over three Indian acts:

- **The Copyright Act, 1957**
- **The Muslim Women (Protection of Rights on Marriage) Act, 2019**
- **The Tribunals Reforms Act, 2021**
- **Farm laws Repeal Act, 2021**
- **Citizenship (Amendment) Bill, 2019**

Answers are grounded in the source text, with citations to the specific Act and section number, and surfaced through a Streamlit chat UI that lets you switch between retrieval strategies on the fly.

---

## Architecture

```
PDF Documents
     │
     ▼
┌──────────────────────────────────────────────┐
│  Ingestion Pipeline (ingestion/loader.py)    │
│  • pdfplumber text extraction                │
│  • Section / Chapter / Schedule extraction   │
│  • Metadata: act, year, section, chapter     │
└──────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│                Two Retrieval Strategies                    │
│                                                            │
│  Cosine Similarity (Dense) — baseline                      │
│  • CharacterTextSplitter (1000 chars / 200 overlap)        │
│  • ChromaDB cosine similarity, top-k = 4                   │
│                                                            │
│  Hybrid BM25 + Dense ⭐                                    │
│  • RecursiveCharacterTextSplitter w/ legal separators      │
│  • EnsembleRetriever: 75 % dense MMR + 25 % BM25           │
│  • Per-act routing (act_filter → filtered ensemble)        │
│  • Micro-chunks dropped from BM25 corpus                   │
└────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────┐
│  GPT-4o-mini Generation    │
│  • Zero-temperature        │
│  • Citation-aware prompt   │
│  • Source panel in UI      │
└────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────┐
│  RAGAS Evaluation Suite                      │
│  • 20 ground-truth Q&A pairs (3 acts)        │
│  • Metrics: Faithfulness, Answer Relevancy,  │
│    Context Precision, Context Recall         │
│  • Latency comparison across chains          │
└──────────────────────────────────────────────┘
```

---

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your OpenAI API key
cp .env.example .env
# Edit .env and paste your key

# 3. Build the vector stores from the PDFs (one-time, ~1 min)
python ingest.py

# 4. Launch the Streamlit chat UI
streamlit run app.py
```

---

## Evaluation

Run the full RAGAS evaluation across both chains:

```bash
python -m evaluation.eval_suite
```

Results are saved to `evaluation/results.csv` (summary) and `evaluation/all_results.csv` (per-question detail).

### Latest results (20 questions, GPT-4o-mini judge)

| Chain | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Avg Latency |
|---|---|---|---|---|---|
| Cosine Similarity (Dense) | 0.703 | **0.868** | **0.661** | 0.750 | 2.40 s |
| Hybrid BM25 + Dense | **0.705** | 0.818 | 0.494 | **0.800** | 2.45 s |

**How to read this:**

The hybrid strategy wins on **context recall** — it finds the right chunks more often (0.80 vs 0.75), thanks to the per-act routing and BM25's strength on exact legal terminology like *talaq-e-biddat* and *Search-cum-Selection Committee*. It also matches the dense baseline on faithfulness.

The dense baseline wins on **context precision** and **answer relevancy** because it returns fewer, larger, less noisy chunks. The hybrid retriever's BM25 component pulls in extra paragraphs that match query keywords without actually answering the question, dragging precision down.

The honest takeaway: pick **Hybrid** when "did the system surface the right legal provision?" is what you care about; pick **Cosine** when you need the LLM to give the cleanest, most grounded one-shot answer. A reranker between retrieval and generation would close the gap and is the obvious next iteration.

---

## Project Structure

```
RAG Chatbot Project/
├── app.py                        # Streamlit chat UI
├── ingest.py                     # One-time vector store builder
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── secrets.toml.example      # template for Streamlit Cloud
│
├── ingestion/
│   └── loader.py                 # PDF → Documents (sections + schedules)
│
├── chains/
│   ├── cosine_similarity.py      # Dense vector baseline
│   └── hybrid_bm25_dense.py      # Hybrid ensemble (best recall)
│
├── evaluation/
│   ├── test_questions.py         # 20 ground-truth Q&A pairs
│   └── eval_suite.py             # RAGAS evaluation runner
│
└── vectorstore/                  # Persisted Chroma stores (gitignored)
    ├── cosine_similarity/
    └── hybrid_bm25_dense/
```

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (the included `.gitignore` keeps `.env`, `vectorstore/`, and `evaluation/*.csv` out of source control).
2. On [share.streamlit.io](https://share.streamlit.io), click **New app**, point it at your repo, and set the entry point to `app.py`.
3. In the app's **Settings → Secrets**, paste:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
4. First load takes ~1 minute while the vector stores build inside the container; subsequent loads are cached by `@st.cache_resource`.

---

## Deploying to GitHub

```bash
cd "RAG Chatbot Project"
git init
git add .
git commit -m "Initial commit — LexQuery RAG"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

The included `.gitignore` already excludes secrets, vector stores, evaluation outputs, and Python cache directories.

---

## Key Design Decisions

**Why BM25 + Dense Hybrid?** Legal text has precise terminology — exact phrases like *talaq-e-biddat*, *Search-cum-Selection Committee*, or *Section 14(1)(a)* are semantically opaque to embeddings but trivially matched by BM25. The ensemble captures both dimensions.

**Why drop micro-chunks from BM25?** The legal-aware recursive splitter produces some very short fragments (section titles, chapter headings ~30–60 chars). BM25 over-rewards them because of huge keyword density per character — they'd shove real provision bodies out of the top-k. Dropping anything under 150 chars from the BM25 corpus only fixes that.

**Why per-act routing?** When a question mentions an act by name, we filter both the dense store (via Chroma metadata filter) and the BM25 corpus to that act's chunks before fusion. Stops cross-act contamination on the Copyright Act in particular, which is the largest and most-referenced of the three.

**Why RAGAS?** LLM-as-a-Judge captures quality dimensions string-matching cannot — paraphrases score correctly, hallucinations are penalised even when confident-sounding. We use it to make the "which chain is better" debate empirical instead of vibes-based.

---

## Tech Stack

| Component | Library |
|---|---|
| Orchestration | LangChain LCEL |
| Vector Store | ChromaDB |
| Sparse Retrieval | BM25 (rank_bm25) |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | GPT-4o-mini |
| PDF Parsing | pdfplumber |
| Evaluation | RAGAS |
| Frontend | Streamlit |
