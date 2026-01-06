"""
Task 3: Semantic search over FAISS index with metadata.

Usage:
  python src/task3_search.py --index vector_store/faiss.index \
                             --meta vector_store/metadata.parquet \
                             --model sentence-transformers/all-MiniLM-L6-v2 \
                             --query "late fee dispute on credit card" \
                             --k 5
"""

import argparse
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

try:
    from rich import print
    from rich.table import Table
except Exception:
    pass

def load_index(index_path: str, meta_path: str):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing index: {index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    index = faiss.read_index(index_path)
    meta = pd.read_parquet(meta_path)
    return index, meta

def normalize_embeddings(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return X / norms

def search(index, meta: pd.DataFrame, model_name: str, query: str, k: int = 5):
    model = SentenceTransformer(model_name)
    q_emb = model.encode([query])
    q_emb = np.asarray(q_emb, dtype=np.float32)
    q_emb = normalize_embeddings(q_emb)

    # FAISS inner product → cosine similarity since vectors are normalized
    scores, ids = index.search(q_emb, k)
    ids = ids[0]
    scores = scores[0]

    results = meta.iloc[ids].copy()
    results["score"] = scores

    return results

def print_results(results: pd.DataFrame):
    try:
        table = Table(title="Top results")
        table.add_column("score", justify="right")
        table.add_column("product")
        table.add_column("company")
        table.add_column("complaint_id", justify="right")
        table.add_column("chunk_index", justify="right")
        table.add_column("text", overflow="fold")

        for _, r in results.iterrows():
            table.add_row(
                f"{r['score']:.4f}",
                str(r.get("product", "")),
                str(r.get("company", "")),
                str(r.get("complaint_id", "")),
                str(r.get("chunk_index", "")),
                str(r.get("text", ""))[:500]
            )
        print(table)
    except Exception:
        # Fallback plain print
        for i, (_, r) in enumerate(results.iterrows(), 1):
            print(f"\n[{i}] score={r['score']:.4f} | product={r.get('product','')} | company={r.get('company','')} | complaint_id={r.get('complaint_id','')} | chunk_index={r.get('chunk_index','')}")
            print(f"Text: {str(r.get('text',''))[:500]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default="vector_store/faiss.index")
    parser.add_argument("--meta", type=str, default="vector_store/metadata.parquet")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    index, meta = load_index(args.index, args.meta)
    results = search(index, meta, args.model, args.query, args.k)
    print_results(results)

if __name__ == "__main__":
    main()
