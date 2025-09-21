import sqlite3
import os
import pdfplumber
from pathlib import Path
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer

DATA_DIR = "E:\GenAI\QnA_pdf\pdf"
INDEX_PATH = "db/faiss_index.bin"
IDS_PATH = "db/ids.npy"



def create_db(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT,
        chunk_id TEXT,
        chunk_text TEXT,
        source_title TEXT,
        source_url TEXT
    )
    """)

    conn.commit()
    conn.close()

def chunk_text(text, max_len=500):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    buffer = ""

    for p in paragraphs:
        if len(buffer) + len(p) + 1 <= max_len:  
            buffer = (buffer + " " + p).strip()
        else:
            chunks.append(buffer)
            buffer = p

    if buffer:
        chunks.append(buffer)

    return chunks

def pdf(DATA_DIR):
    create_db(r"E:\GenAI\QnA_pdf\Code_files\db\database.db")
    count = 1
    with sqlite3.connect(r"E:\GenAI\QnA_pdf\Code_files\db\database.db", timeout=30) as conn:
        curr = conn.cursor()
        for pdf_file in Path(DATA_DIR).glob("*.pdf"):

            print(f"Processing: {pdf_file.name}")
            with pdfplumber.open(pdf_file) as pdf:
                full_text = "\n".join([page.extract_text() or "" for page in pdf.pages])

            chunks = chunk_text(full_text)
            # chunks = text_splitter.split_text(full_text)

            # load sources.json metadata once per pdf
            sources_map = {}
            sources_path = Path(DATA_DIR) / "sources.json"
            if sources_path.exists():
                sources_map = json.loads(sources_path.read_text())

            meta = sources_map.get(pdf_file.name, {})
            title = meta.get("title") or pdf_file.name
            url = meta.get("url") or ""

            for idx, chunk in enumerate(chunks):
                curr.execute(
                    "INSERT INTO chunks (doc_id, chunk_id, chunk_text, source_title, source_url) VALUES (?, ?, ?, ?, ?)",
                    (pdf_file.name, f"{pdf_file.stem}-{idx}", chunk, title, url)
                )

            conn.commit()
            

def build_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')

    conn = sqlite3.connect("E:\GenAI\QnA_pdf\Code_files\db\database.db")
    curr = conn.cursor()

    curr.execute("SELECT id, chunk_text FROM chunks")

    data = curr.fetchall()
    ids = [d[0] for d in data]
    texts = [d[1] for d in data]

    # try load existing index & ids
    if os.path.exists(INDEX_PATH) and os.path.exists(IDS_PATH):
        index = faiss.read_index(INDEX_PATH)
        saved_ids = np.load(IDS_PATH, allow_pickle=True).tolist()
        return saved_ids, index, None, model

    #  embeddings
    batch_size = 32
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch)
        embeddings.append(emb)

    embed = np.vstack(embeddings).astype('float32')

    faiss.normalize_L2(embed)

    dim = embed.shape[1]
    index = faiss.IndexFlatIP(dim)#IP = Innner product

    index.add(embed)

    faiss.write_index(index, INDEX_PATH)
    np.save(IDS_PATH, np.array(ids, dtype=np.int64))

    conn.close()
    return ids, index, embed, model

