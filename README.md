# 🧠 NeuroFusion-RAG

> A local agentic RAG system for EEG / Neuroscience literature using:
>
> * FastAPI
> * LangGraph
> * ChromaDB
> * BM25 Hybrid Retrieval
> * Ollama + Llama3
> * Streamlit UI
>
> Designed for EEG/BCI literature exploration, grounded citation generation, and neuroscience-focused retrieval.

---

# Features

* Hybrid RAG retrieval

  * ChromaDB dense retrieval
  * BM25 sparse retrieval
  * Reciprocal Rank Fusion (RRF)

* Local LLM inference

  * Ollama
  * Llama3
  * Fully offline after model download

* EEG / BCI focused retrieval

* Citation-grounded answers

* FastAPI backend

  * OpenAPI / Swagger docs
  * Health endpoints
  * Structured JSON responses

* Streamlit conversational UI

* Optional terminal chat interface

* PDF ingestion pipeline

* LangGraph multi-node orchestration

---

# Architecture Overview

```text
User Question
      │
      ▼
┌──────────────────────────────────────┐
│          Streamlit / CLI UI          │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│              FastAPI API             │
│          /api/v1/query               │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│          LangGraph Pipeline          │
│                                      │
│  query_analysis                      │
│          ↓                           │
│  hybrid_retrieval                    │
│     ├── ChromaDB                     │
│     └── BM25                         │
│          ↓                           │
│  reranking                           │
│          ↓                           │
│  generation (Ollama Llama3)          │
│          ↓                           │
│  citation_validation                 │
│          ↓                           │
│  response_formatting                 │
└────────────────┬─────────────────────┘
                 │
                 ▼
         Citation-grounded Answer
```

---

# Repository Structure

```text
NeuroFusion-RAG/
├── app/
│   ├── api/
│   ├── agent/
│   ├── ingestion/
│   ├── retrieval/
│   ├── utils/
│   ├── config.py
│   ├── main.py
│   └── schema.py
│
├── scripts/
│   ├── ingest_corpus.py
│   ├── evaluate_retrieval.py
│   └── chat.py
│
├── data/
│   ├── pdfs/
│   └── chroma_db/
│
├── app_ui.py
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

# Tech Stack

| Layer            | Technology            | Purpose                  |
| ---------------- | --------------------- | ------------------------ |
| Backend API      | FastAPI               | REST API                 |
| Agent Framework  | LangGraph             | Multi-node orchestration |
| Dense Retrieval  | ChromaDB              | Vector database          |
| Sparse Retrieval | BM25                  | Keyword retrieval        |
| Embeddings       | sentence-transformers | Semantic embeddings      |
| Local LLM        | Ollama + Llama3       | Offline generation       |
| Frontend         | Streamlit             | Chat-style UI            |
| Parsing          | pypdf                 | PDF ingestion            |
| Validation       | Pydantic v2           | Schemas/config           |
| Logging          | structlog             | Structured logs          |

---

# Setup & Installation

## Prerequisites

* Python 3.11+
* Conda recommended
* Ollama installed
* Llama3 model downloaded
* Windows/Linux/macOS

Recommended RAM:

* 8 GB minimum
* 16 GB recommended for larger models

---

# 1. Install Ollama

Download:

[https://ollama.com/download](https://ollama.com/download)

After installation:

```bash
ollama pull llama3
```

Test Ollama:

```bash
ollama run llama3
```

---

# 2. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/NeuroFusion-RAG.git
cd NeuroFusion-RAG
```

---

# 3. Create Backend Environment

```bash
conda create -n neurofusion python=3.11
conda activate neurofusion
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install ollama
```

---

# 4. Create Streamlit UI Environment

Using a separate UI environment avoids dependency conflicts.

```bash
conda create -n neurofusion-ui python=3.11
conda activate neurofusion-ui
pip install streamlit requests
```

---

# 5. Add PDFs

Place EEG / neuroscience PDFs into:

```text
data/pdfs/
```

---

# 6. Ingest Corpus

Run:

```bash
python -m scripts.ingest_corpus --pdf-dir data/pdfs --collection eeg_papers
```

This will:

* Parse PDFs
* Chunk documents
* Generate embeddings
* Build ChromaDB index
* Build BM25 index

---

# Running the Application

The system uses 3 processes:

1. Ollama
2. FastAPI backend
3. Streamlit frontend

---

# Terminal 1 — Start Ollama

```bash
ollama run llama3
```

Keep this terminal running.

---

# Terminal 2 — Start FastAPI Backend

```bash
conda activate neurofusion

uvicorn app.main:app \
  --reload \
  --host 0.0.0.0 \
  --port 8000
```

Swagger docs:

```text
http://localhost:8000/docs
```

Health endpoints:

```text
http://localhost:8000/health/live
http://localhost:8000/health/ready
```

---

# Terminal 3 — Start Streamlit UI

```bash
conda activate neurofusion-ui
streamlit run app_ui.py
```

Open:

```text
http://localhost:8501
```

---

# Optional Terminal Chat Interface

Create:

```text
scripts/chat.py
```

Example:

```python
import requests
import uuid

URL = "http://localhost:8000/api/v1/query"

while True:
    q = input("\nAsk EEG Copilot: ")

    if q.lower() in {"exit", "quit", "q"}:
        break

    payload = {
        "request_id": str(uuid.uuid4()),
        "query": q,
        "retrieval_config": {
            "strategy": "rrf",
            "top_k_dense": 5,
            "top_k_bm25": 5,
            "top_k_final": 3,
            "rrf_k": 60,
            "dense_weight": 0.6,
            "domain_filter": "bci",
            "year_min": 1900,
            "year_max": 2100,
        },
        "conversation_history": [],
        "stream": False,
    }

    response = requests.post(URL, json=payload, timeout=180)
    response.raise_for_status()

    data = response.json()

    print("\nAnswer:\n")
    print(data["answer"])
```

Install requests:

```bash
pip install requests
```

Run:

```bash
python scripts/chat.py
```

---

# API Reference

## POST `/api/v1/query`

Submit a neuroscience / EEG research question.

Example request:

```json
{
  "request_id": "test-001",
  "query": "What is motor imagery in EEG BCI?",
  "retrieval_config": {
    "strategy": "rrf",
    "top_k_dense": 5,
    "top_k_bm25": 5,
    "top_k_final": 3,
    "rrf_k": 60,
    "dense_weight": 0.6,
    "domain_filter": "bci",
    "year_min": 1900,
    "year_max": 2100
  },
  "conversation_history": [],
  "stream": false
}
```

Returns:

* Generated answer
* Retrieved citations
* Pipeline steps
* Latency metrics

---

## POST `/api/v1/ingest`

Upload and ingest a PDF into the vector database.

---

## GET `/health/live`

Liveness probe.

---

## GET `/health/ready`

Readiness probe.

---

# Example Pipeline Output

```json
{
  "steps_executed": [
    "query_analysis",
    "hybrid_retrieval",
    "reranking",
    "generation",
    "citation_validation",
    "response_formatting"
  ]
}
```

---

# Current Limitations

* CPU inference can be slow (~30–60s per generation)
* Retrieval quality depends heavily on PDF chunking
* Metadata extraction is basic
* Reference sections may dominate retrieval without filtering

---

# Future Improvements

* GPU Ollama inference
* Better semantic chunking
* Metadata extraction
* Conversational memory
* Streaming responses
* Multi-user support
* Citation ranking improvements
* Frontend improvements
* Docker deployment
* Authentication

---

# Notes

This project currently uses:

* Local embeddings
* Local vector database
* Local sparse retrieval
* Local Llama3 inference

No paid API is required.

---

# License

MIT License
