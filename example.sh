#!/bin/bash

# Example 1: Easy query (should return a short, cited answer)
curl -s -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "What is Industrail Robot?",
    "k": 10,
    "mode": "baseline"
  }' | jq

echo ""
echo "---------------------------------------"
echo ""

# Example 2: Tricky query (may abstain if score < threshold, or show reranker helps)
curl -s -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "Explain ISO 13849-1 PLd requirements",
    "k": 10,
    "mode": "rerank"
  }' | jq
