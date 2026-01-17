from task3_search import rag_pipeline
import pandas as pd

test_questions = [
    "What are common credit card complaints?",
    "Why do customers complain about money transfers?",
    "Are savings account complaints frequent?",
    "Explain issues with personal loans.",
    "Tell me about hidden fees."
]

results = []
for q in test_questions:
    answer, sources = rag_pipeline(q, k=3)
    results.append({
        "Question": q,
        "Answer": answer,
        "Sources": sources["text"].head(2).tolist()  # show top 2 sources
    })

df_eval = pd.DataFrame(results)
print(df_eval.to_markdown())
