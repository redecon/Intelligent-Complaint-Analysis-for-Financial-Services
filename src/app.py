import gradio as gr
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ----------------------------
# Load FAISS index and metadata
# ----------------------------
INDEX_PATH = "vector_store/faiss.index"
META_PATH = "vector_store/metadata.parquet"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

index = faiss.read_index(INDEX_PATH)
meta = pd.read_parquet(META_PATH)
model = SentenceTransformer(MODEL_NAME)

def normalize_embeddings(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return X / norms

# ----------------------------
# Core search + response function
# ----------------------------
def rag_search(query, k=5):
    # Encode query
    q_emb = model.encode([query])
    q_emb = np.asarray(q_emb, dtype=np.float32)
    q_emb = normalize_embeddings(q_emb)

    # Search FAISS
    scores, ids = index.search(q_emb, k)
    ids = ids[0]
    scores = scores[0]

    # Retrieve metadata
    results = meta.iloc[ids].copy()
    results["score"] = scores

    # Build answer (placeholder: concatenate top chunks)
    answer = "Based on complaint data, here are relevant insights:\n\n"
    for _, r in results.iterrows():
        answer += f"- {r['text'][:300]}...\n"

    # Sources
    sources = []
    for _, r in results.iterrows():
        sources.append(
            f"[{r['product']}] Complaint {r['complaint_id']} (score={r['score']:.4f}): {r['text'][:200]}..."
        )

    return answer.strip(), "\n".join(sources)

# ----------------------------
# Gradio UI
# ----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 💬 CrediTrust RAG Chatbot\nAsk questions about customer complaints.")

    with gr.Row():
        query = gr.Textbox(label="Your Question", placeholder="Type your question here...")
    with gr.Row():
        submit_btn = gr.Button("Ask")
        clear_btn = gr.Button("Clear")

    answer = gr.Textbox(label="AI Answer", interactive=False, lines=6)
    sources = gr.Textbox(label="Sources", interactive=False, lines=10)

    submit_btn.click(fn=rag_search, inputs=[query], outputs=[answer, sources])
    clear_btn.click(fn=lambda: ("", ""), inputs=[], outputs=[answer, sources])

demo.launch()
