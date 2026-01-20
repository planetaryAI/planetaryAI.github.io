# Local RAG

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Add documents
Put `.txt`, `.md`, or `.pdf` into `docs/`.

## Ingest
```bash
python app/ingest.py
```

## Chat
```bash
python app/chat.py
```

## Files persisted on disk
- `data/index.bin` : HNSW index
- `data/meta.sqlite` : chunk text + metadata
- `data/index_config.json` : index settings / counts
