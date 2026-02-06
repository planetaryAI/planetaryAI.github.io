# Planetary AI – Lunar Mineralogy RAG Demo

Lightweight public demo using FAISS + Llama 3.1 3B (Q4_K_M).

## Purpose
Demonstrates interaction design and retrieval grounding.
The full museum kiosk runs locally with a larger model.

## Run locally
pip install -r requirements.txt
python app.py

## Deploy on Render
Render installs deps, downloads the quantized model,
and runs the API on port 7860.

## Endpoints
GET /
POST /ask { "query": "your question" }