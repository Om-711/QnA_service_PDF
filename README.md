# Industrial Safety QnA Service

A question-answering system for industrial safety PDFs using semantic search and hybrid ranking.

## Features

- **PDF Processing**: Extract and chunk safety documents
- **Hybrid Search**: Combines vector similarity (FAISS) + keyword matching (BM25)
- **Source Citations**: Answers include document references
- **Confidence Scoring**: Filters out uncertain responses

## Project Structure
```
QnA_service_PDF/
├── Code_files/
│   ├── insert.py          # PDF processing and embedding creation
│   └── searching.py       # Search and answer generation
├── db/
│   ├── database.db        # SQLite database with processed chunks
│   ├── faiss_index.bin    # FAISS vector index
│   └── ids.npy           # Index mapping for chunk identification
├── pdf/
│   ├── [20 safety PDFs]   # Source documents
│   └── sources.json      # PDF metadata (titles + URLs)
├── question.json          # Test questions
├── evaluate_8_question.py # Evaluation script
└── requirements.txt       # Dependencies
```

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Build knowledge base** (first time):
```python
from Code_files.insert import build_embeddings, pdf

pdf("pdf/")  # Process PDFs
ids, index, embedding, model = build_embeddings()  # Create embeddings
```

3. **Ask questions**:
```python
from Code_files.searching import search

answer = search("What are machine guard safety requirements?", model, index, ids)
print(answer)

```
where model is Sentence transformer and index and ids are of database

4. **Run evaluation**:
```bash
python evaluate_8_question.py
```

## Dependencies

```txt
faiss-cpu
sentence-transformers
numpy
pandas
pdfplumber
rank_bm25
```

Run the API:
```bash
uvicorn app:app --reload
```

Then ask questions at `http://localhost:8000/docs` or POST to `/ask` with:
```json
{
  "q": "Standardized safety functions",
  "k": 5,
  "mode": "rerank"
}
```
## Configuration

- **Confidence threshold**: 0.4 (adjustable in `searching.py`)
- **Search mode**: `"baseline"` or `"rerank"`
- **Top-K results**: 5 (default)

## Evaluation

The system compares baseline vs hybrid search results on test questions and outputs performance metrics to `evaluation_table.csv`.

---

### cURL Examples

**Simple question**:
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
        "q": "What is Industrial Robot?",
         "k": 8,
       "mode": "baseline"}'
```

**Complex safety query**:
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "q": "Do Industrial Robots Improve Workplace Safety",
       "k": 8,
       "mode": "rerank"
     }'
```

**Ready to search your safety documents intelligently!**
