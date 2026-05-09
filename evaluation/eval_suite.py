"""
RAGAS Evaluation Suite — LexQuery
─────────────────────────────────
Runs both retrieval chains against the 20-question test set and produces
a comparison table of RAGAS metrics + latency.

Usage:
    python -m evaluation.eval_suite

Requires OPENAI_API_KEY in .env.
Output: evaluation/results.csv  +  evaluation/all_results.csv  +  printed summary.
"""

import os
import time
import traceback
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

load_dotenv()

from ingestion.loader import load_all_pdfs
from chains.cosine_similarity import build_cosine_chain, CHAIN_NAME as COSINE_NAME
from chains.hybrid_bm25_dense import build_hybrid_chain, CHAIN_NAME as HYBRID_NAME
from evaluation.test_questions import TEST_DATASET

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def run_chain_on_dataset(chain, retriever, test_data, chain_name):
    """
    Run a chain on all test questions and collect answers + contexts.
    Returns a list of dicts ready for RAGAS Dataset.
    """
    results = []
    print(f"\n── Evaluating {chain_name} ({len(test_data)} questions) ──")

    for i, item in enumerate(test_data):
        question = item["question"]
        ground_truth = item["ground_truth"]

        t0 = time.time()
        try:
            answer = chain.invoke(question)
            latency = time.time() - t0
            retrieved_docs = retriever.invoke(question)
            contexts = [doc.page_content for doc in retrieved_docs]
        except Exception as e:
            print(f"  ✗ Q{i+1} error: {e}")
            answer = ""
            contexts = []
            latency = 0.0

        results.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "latency_s": round(latency, 3),
                "chain": chain_name,
                "act": item["act"],
            }
        )
        print(f"  ✓ Q{i+1}/{len(test_data)} — {latency:.2f}s")

    return results


def compute_ragas_metrics(results_list):
    """
    Run RAGAS evaluation on a list of result dicts.
    Returns {metric_name: mean_float}. Robust across RAGAS 0.1.x / 0.2.x.
    """
    clean = [
        r for r in results_list
        if r.get("answer") and r.get("contexts") and r.get("ground_truth")
    ]
    dropped = len(results_list) - len(clean)
    if dropped:
        print(f"    (dropped {dropped} rows with empty answer/context/gt before RAGAS)")

    if not clean:
        return {m: float("nan") for m in METRIC_NAMES}

    dataset = Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r["ground_truth"],
            }
            for r in clean
        ]
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    try:
        df_scores = result.to_pandas()
        means = {}
        for m in METRIC_NAMES:
            if m in df_scores.columns:
                means[m] = float(df_scores[m].dropna().mean())
            else:
                means[m] = float("nan")
        return means
    except Exception as e:
        print(f"    [warn] .to_pandas() failed ({e}); falling back to dict-style access")
        means = {}
        for m in METRIC_NAMES:
            try:
                v = result[m]
                if hasattr(v, "mean"):
                    v = float(v.mean())
                else:
                    v = float(v)
                means[m] = v
            except Exception:
                means[m] = float("nan")
        return means


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Add it to .env")

    pdf_dir = str(Path(__file__).parent.parent)
    print(f"Loading documents from: {pdf_dir}")
    documents = load_all_pdfs(pdf_dir)

    print("\nBuilding chains…")
    c1, r1 = build_cosine_chain(documents, api_key)
    c2, r2 = build_hybrid_chain(documents, api_key)

    chains = [
        (COSINE_NAME, c1, r1),
        (HYBRID_NAME, c2, r2),
    ]

    all_results = []
    ragas_summary = []
    output_dir = Path(__file__).parent

    def _safe_round(v):
        try:
            return round(float(v), 3)
        except Exception:
            return None

    for chain_name, chain, retriever in chains:
        results = run_chain_on_dataset(chain, retriever, TEST_DATASET, chain_name)
        all_results.extend(results)
        avg_latency = sum(r["latency_s"] for r in results) / len(results)

        print(f"\n  Computing RAGAS metrics for {chain_name}…")
        try:
            scores = compute_ragas_metrics(results)
            ragas_summary.append(
                {
                    "Chain": chain_name,
                    "Faithfulness": _safe_round(scores.get("faithfulness")),
                    "Answer Relevancy": _safe_round(scores.get("answer_relevancy")),
                    "Context Precision": _safe_round(scores.get("context_precision")),
                    "Context Recall": _safe_round(scores.get("context_recall")),
                    "Avg Latency (s)": round(avg_latency, 3),
                }
            )
        except Exception as e:
            print(f"  RAGAS error for {chain_name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            ragas_summary.append(
                {
                    "Chain": chain_name,
                    "Faithfulness": None,
                    "Answer Relevancy": None,
                    "Context Precision": None,
                    "Context Recall": None,
                    "Avg Latency (s)": round(avg_latency, 3),
                }
            )

        # Incremental write so a later crash doesn't destroy partial results
        try:
            pd.DataFrame(ragas_summary).to_csv(output_dir / "results.csv", index=False)
            pd.DataFrame(all_results).to_csv(output_dir / "all_results.csv", index=False)
        except Exception as e:
            print(f"  [warn] incremental CSV write failed: {e}")

    df_summary = pd.DataFrame(ragas_summary)
    print("\n\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print("=" * 70)
    print(f"\nDetailed results saved to: {output_dir}/all_results.csv")
    print(f"Summary saved to: {output_dir}/results.csv")


if __name__ == "__main__":
    main()
