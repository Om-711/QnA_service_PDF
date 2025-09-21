import sqlite3
import faiss
from rank_bm25 import BM25Okapi
import numpy as np



def search(query, model, index, ids, k=10):

    query_s = model.encode([query]).astype('float32')

    faiss.normalize_L2(query_s)
    scores, indices = index.search(query_s, k)

    top_idxs = indices[0]
    top_scores = scores[0]

    conn = sqlite3.connect("db/database.db")
    curr = conn.cursor()
    results = []

    for pos, score in zip(top_idxs, top_scores):
        if pos < 0 or pos >= len(ids):
            continue

        chunk_db_id = ids[pos]
        
        curr.execute("SELECT doc_id, chunk_id, chunk_text, source_title, source_url FROM chunks WHERE id=?", (chunk_db_id,))
        
        row = curr.fetchone()
        
        if not row: 
            continue
        
        doc_id, chunk_id, text, source_title, source_url = row

        results.append({
            "db_id": chunk_db_id, # id in db
            "doc_id": doc_id,#pdf name
            "chunk_id": chunk_id, # pdf chunk id
            "text": text,
            "vector_score": float(score),
            "source_title": source_title,
            "source_url": source_url
        })

    conn.close()
    return results



def rerank(results, query, alpha=0.6):
    if not results:
        return results

    texts = [r['text'] for r in results]
    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.split())

    vec_scores = np.array([r['vector_score'] for r in results], dtype=float)

    vec_norm = vec_scores
    bm_norm = bm25_scores

    for i, r in enumerate(results):
        r['bm25_score'] = float(bm25_scores[i])
        r['vector_score_norm'] = float(vec_norm[i])
        r['bm25_score_norm'] = float(bm_norm[i])
        r['total_score'] = float(alpha * vec_norm[i] + (1 - alpha) * bm_norm[i])

    results = sorted(results, key=lambda x: x['total_score'], reverse=True)
    return results




def generate_answer(results, threshold=0.5):
    # results must be sorted by final score if reranked; otherwise by vector_score
    if not results:
        return None
    
    top = results[0]
    
    # score_for_abstain = top["total_score"]
    score_for_abstain = top.get("total_score", top.get("vector_score", 0.0))
    
    if score_for_abstain < threshold:
        return None
    
    snippet = top['text'][:330].strip()
    
    citation = f"{top['source_title'] or top['doc_id']} (chunk: {top['chunk_id']})"
    
    return f"{snippet}\n\nSource: {citation}\nURL: {top['source_url']}"


