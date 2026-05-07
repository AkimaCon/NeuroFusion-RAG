# 🧠 Agentic EEG Research Copilot

> A production-style RAG agent for EEG / Neuroscience literature —  
> built on **FastAPI**, **LangGraph**, **ChromaDB**, and **Hybrid Search**.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Repository Structure](#repository-structure)
3. [Tech Stack](#tech-stack)
4. [Data Flow](#data-flow)
5. [LangGraph Node Reference](#langgraph-node-reference)
6. [Hybrid Search Design](#hybrid-search-design)
7. [Setup & Installation](#setup--installation)
8. [Environment Variables](#environment-variables)
9. [Running the API](#running-the-api)
10. [API Reference](#api-reference)
11. [Testing](#testing)
12. [Roadmap](#roadmap)

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  FastAPI  /query                    │
│           (QueryRequest → QueryResponse)            │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│               LangGraph Agent Graph                 │
│                                                     │
│  ┌──────────────┐    ┌──────────────────────────┐   │
│  │query_analysis│───▶│   hybrid_retrieval       │   │
│  │  (expand +   │    │  ┌────────┐ ┌──────────┐ │   │
│  │  classify)   │    │  │ChromaDB│ │  BM25    │ │   │
│  └──────────────┘    │  │(dense) │ │(sparse)  │ │   │
│                      │  └────┬───┘ └────┬─────┘ │   │
│                      │       └────┬─────┘        │   │
│                      │      RRF Fusion            │   │
│                      └──────────────────────────┘   │
│                              │                       │
│                              ▼                       │
│                    ┌─────────────────┐              │
│                    │   reranking     │              │
│                    │  (cross-encode) │              │
│                    └────────┬────────┘              │
│                             │                       │
│                             ▼                       │
│                    ┌─────────────────┐              │
│                    │   generation    │              │
│                    │ (GPT-4o + ctx)  │              │
│                    └────────┬────────┘              │
│                             │                       │
│                             ▼                       │
│                  ┌────────────────────┐             │
│                  │ citation_validation│             │
│                  └────────┬───────────┘             │
│                           │                         │
│                           ▼                         │
│                ┌─────────────────────┐              │
│                │ response_formatting │              │
│                └─────────────────────┘              │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
               QueryResponse (JSON)
```

---

## Repository Structure

```
eeg-copilot/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app factory & lifespan
│   ├── schema.py                # Pydantic v2 models (AgentState, Citation, etc.)
│   ├── config.py                # pydantic-settings env config
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py            # /query, /ingest, /health endpoints
│   │   └── dependencies.py     # FastAPI Depends() providers
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py             # LangGraph StateGraph definition
│   │   └── nodes/
│   │       ├── __init__.py
│   │       ├── query_analysis.py
│   │       ├── hybrid_retrieval.py
│   │       ├── reranking.py
│   │       ├── generation.py
│   │       ├── citation_validation.py
│   │       └── response_formatting.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── dense.py             # ChromaDB dense retriever
│   │   ├── sparse.py            # BM25 sparse retriever (rank-bm25)
│   │   └── fusion.py            # RRF & linear fusion logic
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py        # pypdf-based chunking pipeline
│   │   ├── embedder.py          # sentence-transformers embedding
│   │   └── indexer.py           # ChromaDB + BM25 index management
│   └── utils/
│       ├── __init__.py
│       ├── logging.py           # structlog configuration
│       └── timing.py            # latency tracking decorator
├── tests/
│   ├── conftest.py
│   ├── test_schema.py
│   ├── test_retrieval.py
│   ├── test_agent.py
│   └── test_api.py
├── data/
│   ├── pdfs/                    # Raw EEG paper PDFs (git-ignored)
│   └── chroma_db/               # Persistent ChromaDB volume (git-ignored)
├── scripts/
│   ├── ingest_corpus.py         # CLI: bulk-ingest a folder of PDFs
│   └── evaluate_retrieval.py    # Retrieval quality metrics (MRR, nDCG)
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml               # ruff + mypy config
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Library | Purpose |
|---|---|---|
| API | FastAPI 0.115 | Async REST + SSE streaming |
| Agent | LangGraph 0.2 | Stateful multi-node graph execution |
| LLM | langchain-openai | GPT-4o generation & query analysis |
| Dense Retrieval | ChromaDB 0.5 | HNSW vector store |
| Sparse Retrieval | rank-bm25 | BM25Okapi over chunk corpus |
| Embeddings | sentence-transformers | `BAAI/bge-large-en-v1.5` |
| PDF Parsing | pypdf | Text + metadata extraction |
| Validation | Pydantic v2 | Schema, env config, serialisation |
| Observability | structlog | Structured JSON logging |

---

## Data Flow

### Ingestion Pipeline

```
PDF File
  │
  ▼
pdf_parser.py  ──▶  DocumentChunk list (text + metadata)
  │
  ├──▶  embedder.py  ──▶  float[] embeddings
  │                           │
  │                           ▼
  │                     ChromaDB collection
  │                      (dense index)
  │
  └──▶  indexer.py  ──▶  BM25 in-memory corpus
                           (serialised to disk)
```

### Query Pipeline (LangGraph)

```
QueryRequest
  │
  ▼
[query_analysis]      Expand acronyms (e.g. ERP→Event-Related Potential),
                      classify EEG domain, rewrite for retrieval.
  │
  ▼
[hybrid_retrieval]    Run dense (ChromaDB top-K) + sparse (BM25 top-K)
                      in parallel; fuse via RRF or linear weighting.
  │
  ▼
[reranking]           Cross-encoder re-scores fused candidates;
                      keeps top_k_final chunks.
  │
  ▼
[generation]          Constructs grounded prompt with ranked context;
                      calls GPT-4o with citation instruction template.
  │
  ▼
[citation_validation] Verifies every [N] marker maps to a real chunk;
                      strips hallucinated citations.
  │
  ▼
[response_formatting] Assembles QueryResponse with latency + token stats.
```

---

## LangGraph Node Reference

| Node | Input keys (AgentState) | Output keys |
|---|---|---|
| `query_analysis` | `query`, `conversation_history` | `expanded_query`, `retrieval_config` |
| `hybrid_retrieval` | `expanded_query`, `retrieval_config` | `raw_chunks` |
| `reranking` | `raw_chunks`, `expanded_query` | `reranked_chunks` |
| `generation` | `reranked_chunks`, `expanded_query`, `conversation_history` | `draft_answer`, `citations` |
| `citation_validation` | `draft_answer`, `citations`, `reranked_chunks` | `final_answer`, `citations` |
| `response_formatting` | all keys | *(returns QueryResponse)* |

---

## Hybrid Search Design

### Reciprocal Rank Fusion (RRF)

For each retrieved chunk *d*, given rank lists from dense retriever *R_dense*
and BM25 *R_bm25*:

```
RRF(d) = 1 / (k + rank_dense(d))  +  1 / (k + rank_bm25(d))
```

where `k = 60` by default (configurable via `RetrievalConfig.rrf_k`).

### Linear Fusion

```
score(d) = w * cosine_sim(d)  +  (1 - w) * norm_bm25(d)
```

where `w = RetrievalConfig.dense_weight` (default 0.6).

BM25 scores are min-max normalised to [0, 1] before fusion.

### Why Hybrid?

Dense retrieval excels at semantic paraphrasing but struggles with rare EEG
acronyms (ERN, SSVEP, P300). BM25 captures exact keyword overlap for technical
terms. Fusion consistently outperforms either alone on domain-specific corpora.

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- An OpenAI API key (GPT-4o access)
- 8 GB RAM recommended (sentence-transformers model)

### 1. Clone & create virtual environment

```bash
git clone https://github.com/your-org/eeg-copilot.git
cd eeg-copilot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY at minimum
```

### 4. Ingest your EEG corpus

```bash
mkdir -p data/pdfs
# Copy your EEG PDFs into data/pdfs/
python scripts/ingest_corpus.py --pdf-dir data/pdfs --collection eeg_papers
```

This will:
- Parse and chunk all PDFs
- Compute `BAAI/bge-large-en-v1.5` embeddings
- Persist vectors to `data/chroma_db/`
- Serialise the BM25 corpus index

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | Generation model name |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | sentence-transformers model |
| `CHROMA_PERSIST_DIR` | `data/chroma_db` | ChromaDB persistence path |
| `CHROMA_COLLECTION` | `eeg_papers` | Collection name |
| `BM25_INDEX_PATH` | `data/bm25_index.pkl` | Serialised BM25 corpus |
| `LOG_LEVEL` | `INFO` | structlog level |
| `API_HOST` | `0.0.0.0` | Uvicorn host |
| `API_PORT` | `8000` | Uvicorn port |
| `MAX_CHUNK_TOKENS` | `512` | PDF splitting chunk size |
| `CHUNK_OVERLAP_TOKENS` | `64` | Sliding-window overlap |

---

## Running the API

### Development

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production (Gunicorn + Uvicorn workers)

```bash
gunicorn app.main:app \
  -k uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:8000
```

### Docker

```bash
docker compose up --build
```

---

## API Reference

### `POST /query`

Submit a research question and receive a grounded answer with citations.

**Request body:** `QueryRequest`

```json
{
  "query": "What signal features best discriminate motor imagery classes in EEG-BCI?",
  "retrieval_config": {
    "strategy": "rrf",
    "top_k_dense": 20,
    "top_k_bm25": 20,
    "top_k_final": 5,
    "domain_filter": "bci"
  },
  "stream": false
}
```

**Response:** `QueryResponse`

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "answer": "Common Spatial Patterns (CSP) combined with log-variance features remain the dominant approach [1]. Recent work has shown that Riemannian geometry-based methods outperform CSP on cross-session tasks [2]...",
  "citations": [
    {
      "citation_id": "...",
      "chunk_id": "...",
      "source_file": "blankertz_2008_optimizing.pdf",
      "page_number": 3,
      "text": "...",
      "fusion_score": 0.91,
      "authors": ["Blankertz, B.", "Tomioka, R."],
      "year": 2008,
      "doi": "10.1109/TNSRE.2007.100899"
    }
  ],
  "steps_executed": ["query_analysis", "hybrid_retrieval", "reranking", "generation", "citation_validation", "response_formatting"],
  "latency_ms": { "query_analysis": 312.4, "hybrid_retrieval": 89.1, "generation": 1803.2 },
  "token_usage": { "prompt": 2841, "completion": 487, "total": 3328 }
}
```

### `POST /ingest`

Ingest a single PDF file into the hybrid index.

**Request:** `multipart/form-data` with `file` field (PDF)

### `GET /health`

Returns `{"status": "ok", "chroma_docs": N, "bm25_corpus_size": M}`.

---

## Testing

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=term-missing

# Specific node tests
pytest tests/test_agent.py -v -k "retrieval"
```

---

## Roadmap

- **Step 2** — `app/config.py` (pydantic-settings) + `app/main.py` (FastAPI lifespan)
- **Step 3** — `app/ingestion/` pipeline (PDF parser, embedder, indexer)
- **Step 4** — `app/retrieval/` (ChromaDB dense, BM25 sparse, RRF fusion)
- **Step 5** — `app/agent/nodes/` (all six LangGraph nodes)
- **Step 6** — `app/agent/graph.py` (StateGraph wiring + conditional edges)
- **Step 7** — `app/api/` (routes, dependencies, streaming SSE)
- **Step 8** — `scripts/` (bulk ingest CLI, retrieval evaluation)
- **Step 9** — `tests/` (full pytest suite with fixtures)
- **Step 10** — Dockerfile + docker-compose + CI workflow
