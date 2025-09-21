from fastapi import FastAPI
from pydantic import BaseModel
from searching import search, rerank, generate_answer
from insert import build_embeddings, pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import HTTPException
import threading
import random
import torch
import threading
import sqlite3


app = FastAPI()
import numpy as np
np.random.seed(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


ids, index, embedding, model = None, None, None, None
embeddings_ready = False

def prepare_data():
    global ids, index, embedding, model, embeddings_ready
    print("Populating database with PDF chunks...")
    pdf(r"E:\GenAI\QnA_pdf\pdf_copy")
    print("Database population complete.")

    print("Generating embeddings and FAISS index...")
    ids, index, embedding, model = build_embeddings()
    embeddings_ready = True
    print("Embeddings ready for use.")

@app.on_event("startup")
def startup_event():
    # Run in background to avoid blocking
    threading.Thread(target=prepare_data, daemon=True).start()


class AskRequest(BaseModel):
    q: str
    k: int = 5
    mode: str = "baseline"

conn = sqlite3.connect("db/database.db")
curr = conn.cursor()

try:
    curr.execute("ALTER TABLE chunks ADD COLUMN source_title TEXT")
except sqlite3.OperationalError:
    pass  # column already exists

try:
    curr.execute("ALTER TABLE chunks ADD COLUMN source_url TEXT")
except sqlite3.OperationalError:
    pass

conn.commit()
conn.close()

@app.post("/ask")
def ask(req: AskRequest):
    global embeddings_ready, ids, index, embedding, model

    if not embeddings_ready:
        raise HTTPException(status_code=503, detail="Embeddings are still being generated. Please try again shortly.")

    results = search(req.q, model, index, ids, req.k)

    reranker_used = False
    if req.mode == "rerank":
        results = rerank(results, req.q)
        reranker_used = True

    answer = generate_answer(results)

    contexts = [
        {
            "db_id": r["db_id"],
            "doc_id": r["doc_id"],
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "vector_score": r.get("vector_score"),
            "bm25_score": r.get("bm25_score"),
            "total_score": r.get("total_score"),
            "source_title": r["source_title"],
            "source_url": r["source_url"]
        } for r in results[:req.k]
    ]

    return {
        "answer": answer,
        "contexts": contexts,
        "reranker_used": reranker_used
    }