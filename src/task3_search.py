import faiss
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ----------------------------
# Load FAISS index and metadata
# ----------------------------
INDEX_PATH = "vector_store/faiss.index"
META_PATH = "vector_store/metadata.parquet"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

index = faiss.read_index(INDEX_PATH)
meta = pd.read_parquet(META_PATH)
embedder = SentenceTransformer(MODEL_NAME)

# Load a lightweight LLM (can swap for larger models if resources allow)
generator = pipeline("text2text-generation", model="google/flan-t5-small")

# ----------------------------
# Prompt template
# ----------------------------
PROMPT_TEMPLATE = """
You are an assistant for CrediTrust Financial. Use the following retrieved complaint excerpts to answer the question.
If the answer is not contained in the excerpts, say you don't know.

Question: {question}

Retrieved Context:
{context}

Answer:
"""

# ----------------------------
# Helper functions
# ----------------------------
def normalize_embeddings(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return X / norms

def retrieve_context(query: str, k: int = 5):
    q_emb = embedder.encode([query])
    q_emb = normalize_embeddings(np.asarray(q_emb, dtype=np.float32))
    scores, ids = index.search(q_emb, k)
    results = meta.iloc[ids[0]].copy()
    results["score"] = scores[0]
    return results

# ----------------------------
# RAG pipeline
# ----------------------------
def rag_pipeline(query: str, k: int = 5):
    # Step 1: Retrieve top-k chunks
    results = retrieve_context(query, k)
    context = "\n".join(results["text"].tolist())

    # Step 2: Build prompt
    prompt = PROMPT_TEMPLATE.format(question=query, context=context)

    # Step 3: Generate answer
    llm_output = generator(prompt, max_length=300, do_sample=False)[0]["generated_text"]

    # Step 4: Return answer + sources
    sources = results[["complaint_id", "product", "text"]].head(2).to_dict(orient="records")
    return llm_output.strip(), sources

# ----------------------------
# CLI entry point
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RAG search pipeline")
    parser.add_argument("--query", type=str, required=True, help="User query")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args()

    answer, sources = rag_pipeline(args.query, k=args.k)
    print("\nQuestion:", args.query)
    print("\nAnswer:\n", answer)
    print("\nSources:")
    for s in sources:
        print(f"- Complaint {s['complaint_id']} [{s['product']}]: {s['text'][:200]}...")
