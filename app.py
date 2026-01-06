import gradio as gr
from task3_search import rag_pipeline

# ----------------------------
# Chat function
# ----------------------------
def chat(query: str):
    if not query.strip():
        return "Please enter a question.", ""
    answer, sources = rag_pipeline(query, k=5)
    sources_text = "\n".join(
        [f"- Complaint {s['complaint_id']} [{s['product']}]: {s['text'][:200]}..." for s in sources]
    )
    return answer, sources_text

# ----------------------------
# Gradio Interface
# ----------------------------
with gr.Blocks(theme="gradio/soft") as demo:
    gr.Markdown("""
    # 🧾 CrediTrust RAG Chatbot
    Welcome! This tool helps **Product Managers, Support, and Compliance teams** quickly explore customer complaints.

    ### How to use:
    - Type your question in plain English (e.g., *"What are common credit card complaints?"*).
    - Click **Ask** to see the AI’s answer.
    - Review the **Sources** shown below to verify the response.
    - Use **Clear** to reset and ask another question.

    ⚠️ Note: Answers are based on retrieved complaint data. If the context doesn’t contain the answer, the chatbot will say it doesn’t know.
    """)

    with gr.Row():
        query = gr.Textbox(label="Your Question", placeholder="Type your question here...")
    with gr.Row():
        ask_btn = gr.Button("Ask")
        clear_btn = gr.Button("Clear")

    answer = gr.Textbox(label="AI Answer")
    sources = gr.Textbox(label="Sources")

    ask_btn.click(chat, inputs=query, outputs=[answer, sources])
    clear_btn.click(lambda: ("", ""), None, [answer, sources])

# ----------------------------
# Launch
# ----------------------------
if __name__ == "__main__":
    demo.launch()
