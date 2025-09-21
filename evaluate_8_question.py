import json
from Code_files.searching import search, rerank, generate_answer
from Code_files.insert import build_embeddings, pdf
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


print("Started Embedding...........")

# if os.path.exists("E:\GenAI\QnA_pdf\db\faiss_index.bin") and os.path.exists("E:\GenAI\QnA_pdf\db\ids.npy"):
index = faiss.read_index(r"E:\GenAI\QnA_pdf\db\faiss_index.bin")
with open(r"E:\GenAI\QnA_pdf\db\ids.npy", "rb") as f:
    ids = np.load(f, allow_pickle=True)
    ids = ids.tolist()
# else:
#     print("Data not exist")
#     ids, index, embedding, model = build_embeddings()
print("Embedding Ready for Use..........")


with open("E:\GenAI\QnA_pdf\question.json") as f:
    questions = json.load(f)


# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# load_dotenv()
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# def generate_with_gemini(results, question):
#     if not results:
#         return "No relevant context found."

#     # If user accidentally passed a single string (e.g. baseline_answer), handle it:
#     if isinstance(results, str):
#         context = results
#     else:
#         # Expecting an iterable of dicts with 'text'
#         try:
#             context = "\n\n".join(r.get("text", "") if isinstance(r, dict) else str(r) for r in results)
#         except TypeError:
#             raise TypeError("generate_with_gemini expects a list of dicts or a string. Got: " + repr(type(results)))
        

#     prompt = f"""You are a helpful assistant. 
#         Answer the question **only using the context below**. 
#         If you are not sure, say 'I don't know'.

#         Context:
#         {context}

#         Question: {question}
#         Answer:"""

#     response = llm.invoke(prompt)
#     return response.content if hasattr(response, "content") else str(response)



results_table = []

for q in questions:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    baseline_results = search(q['question'], model, index, ids, k=5)
    baseline_answer = generate_answer(baseline_results)

    reranked_results = rerank(baseline_results, q['question'])
    rerank_answer = generate_answer(reranked_results)
    # gemini_answer_baseline = generate_with_gemini(baseline_answer, q)
    # gemini_answer_rerank = generate_with_gemini(rerank_answer, q)

    results_table.append({
        "question": q['question'],
        "baseline_answer": baseline_answer,
        "rerank_answer": rerank_answer,
        # "gemini_baseline_answer" : gemini_answer_baseline,
        # "gemini_rerank_answer" : gemini_answer_rerank,
        "baseline_top_score": baseline_results[0].get("vector_score", 0) if baseline_results else 0,
        "rerank_top_score": reranked_results[0].get("total_score", 0) if reranked_results else 0
    })


import pandas as pd
df = pd.DataFrame(results_table)
df.to_csv("eval_final.csv", index=False)
print(df)
print("Evaluation complete! Table saved to eval_final.csv")
