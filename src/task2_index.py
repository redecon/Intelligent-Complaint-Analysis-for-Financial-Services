"""
Task 2: Stratified sampling, text chunking, embedding, and FAISS indexing.

Usage:
  python src/task2_index.py --input data/processed/filtered_complaints.csv \
                            --out_dir vector_store \
                            --sample_size 12000 \
                            --chunk_size 500 \
                            --chunk_overlap 50
"""

import argparse
import os
import math
import re
import uuid
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

# ----------------------------
# Configurable defaults
# ----------------------------
TARGET_PRODUCTS = {"Credit Card", "Personal Loan", "Savings Account", "Money Transfers"}

# ----------------------------
# Helpers
# ----------------------------
def stratified_sample(df: pd.DataFrame, label_col: str, n: int) -> pd.DataFrame:
    """Stratified sample of size n with proportional representation by label_col."""
    # Compute per-class fractions and target counts
    counts = df[label_col].value_counts()
    frac = counts / counts.sum()
    target_counts = (frac * n).round().astype(int)

    # Ensure at least 1 per class if n >= number of classes
    for cls in target_counts.index:
        if target_counts[cls] == 0:
            target_counts[cls] = 1

    # Sample per class
    parts = []
    for cls, k in target_counts.items():
        part = df[df[label_col] == cls].sample(n=min(k, len(df[df[label_col] == cls])), random_state=42)
        parts.append(part)
    sample_df = pd.concat(parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return sample_df

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Simple character-based chunker with overlap."""
    if not isinstance(text, str):
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks

def normalize_embeddings(X: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings for cosine similarity with inner product index."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return X / norms

# ----------------------------
# Main pipeline
# ----------------------------
def build_index(input_csv: str, out_dir: str, sample_size: int, chunk_size: int, chunk_overlap: int):
    os.makedirs(out_dir, exist_ok=True)

    # Load cleaned dataset (from Task 1)
    df = pd.read_csv(input_csv, low_memory=False)
    # Expected columns: complaint_id, product, issue, company, state, date_received, narrative_clean
    # Standardize product names to Title case to match TARGET_PRODUCTS
    df["product"] = df["product"].astype(str).str.strip().str.title()

    # Filter to target products only (safety check)
    df = df[df["product"].isin(TARGET_PRODUCTS)].copy()

    # Stratified sample across products
    n = min(sample_size, len(df))
    sampled = stratified_sample(df, label_col="product", n=n)
    print(f"Sampled {len(sampled)} records across: {sorted(sampled['product'].unique())}")

    # Chunking
    rows = []
    for _, r in tqdm(sampled.iterrows(), total=len(sampled), desc="Chunking"):
        chunks = chunk_text(r["narrative_clean"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, ch in enumerate(chunks):
            rows.append({
                "chunk_id": str(uuid.uuid4()),
                "complaint_id": r["complaint_id"],
                "product": r["product"],
                "issue": r.get("issue", ""),
                "company": r.get("company", ""),
                "state": r.get("state", ""),
                "date_received": r.get("date_received", ""),
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "text": ch
            })
    chunks_df = pd.DataFrame(rows)
    print(f"Total chunks: {len(chunks_df)}")

    # Embeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks_df["text"].tolist(), batch_size=256, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings = normalize_embeddings(embeddings)  # for cosine similarity

    # FAISS index (inner product)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors (dim={dim})")

    # Persist index
    index_path = os.path.join(out_dir, "faiss.index")
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index: {index_path}")

    # Persist metadata aligned to row order
    meta_cols = [
        "chunk_id", "complaint_id", "product", "issue", "company", "state",
        "date_received", "chunk_index", "total_chunks", "text"
    ]
    meta_path = os.path.join(out_dir, "metadata.parquet")
    chunks_df[meta_cols].to_parquet(meta_path, index=False)
    print(f"Saved metadata: {meta_path}")

    # Persist model name and chunking params for reproducibility
    info = {
        "model_name": model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "sample_size": len(sampled),
        "products": sorted(sampled["product"].unique())
    }
    pd.Series(info).to_json(os.path.join(out_dir, "index_info.json"))
    print("Saved index_info.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/filtered_complaints.csv")
    parser.add_argument("--out_dir", type=str, default="vector_store")
    parser.add_argument("--sample_size", type=int, default=12000)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    args = parser.parse_args()

    build_index(
        input_csv=args.input,
        out_dir=args.out_dir,
        sample_size=args.sample_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
