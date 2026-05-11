"""
app/main.py
===========
FastAPI application factory for the Agentic EEG Research Copilot.

Responsibilities
----------------
1. **Lifespan** (startup / shutdown):
     - Validate environment and settings on boot.
     - Load the SentenceTransformer embedding model into process memory.
     - Initialise (or re-open) the ChromaDB persistent client + collection.
     - Load the serialised BM25 corpus index from disk (if it exists).
     - Store all long-lived resources on `app.state` for dependency injection.
     - Gracefully tear down resources on shutdown.

2. **Middleware stack** (outermost → innermost):
     - RequestIDMiddleware  – stamps every request with a UUID trace ID.
     - StructlogMiddleware  – binds the trace ID into structlog context-vars
                              and logs request start / finish with latency.
     - CORSMiddleware       – configurable allowed origins.
     - GZipMiddleware       – compresses responses ≥ 1 kB.

3. **Exception handlers**:
     - RequestValidationError → 422 with structured field-level error list.
     - HTTPException          → passthrough with JSON body.
     - Generic Exception      → 500 with opaque error ID (no stack leak).

4. **Routers**:
     - /health   – liveness + readiness probe (inline, no dependency)
     - /api/v1/* – all domain routes (wired in Step 7)

Resource access pattern
-----------------------
    from fastapi import Request

    def my_dependency(request: Request):
        chroma_collection = request.app.state.chroma_collection
        bm25_retriever    = request.app.state.bm25_retriever
        embedder          = request.app.state.embedder
"""

from __future__ import annotations

import pickle
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import chromadb
import structlog
from chromadb.config import Settings as ChromaSettings
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.config import Settings, get_settings
from app.utils.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
)

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Resource initialisation helpers
# ──────────────────────────────────────────────────────────────────────────────


def _init_embedding_model(settings: Settings) -> SentenceTransformer:
    """
    Load the SentenceTransformer model into memory.

    The model is downloaded from HuggingFace Hub on first run and cached
    locally (~1.3 GB for bge-large-en-v1.5). Subsequent startups load from
    the local cache (typically < 10 s).
    """
    log.info(
        "loading_embedding_model",
        model=settings.embedding.model_name,
        device=settings.embedding.device,
    )
    t0 = time.perf_counter()
    model = SentenceTransformer(
        settings.embedding.model_name,
        device=settings.embedding.device,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    log.info(
        "embedding_model_loaded",
        model=settings.embedding.model_name,
        elapsed_ms=round(elapsed, 1),
    )
    return model


def _init_chroma(settings: Settings) -> chromadb.Collection:
    """
    Open (or create) the persistent ChromaDB client and EEG collection.

    Uses the cosine-distance HNSW index with L2-normalised embeddings.
    The `get_or_create_collection` call is idempotent: a fresh install creates
    the collection, subsequent restarts re-open the existing one.
    """
    log.info(
        "initialising_chromadb",
        persist_dir=str(settings.chroma.persist_dir),
        collection=settings.chroma.collection_name,
    )
    client = chromadb.PersistentClient(
        path=str(settings.chroma.persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=settings.chroma.collection_name,
        metadata={"hnsw:space": settings.chroma.distance_function},
    )
    doc_count = collection.count()
    log.info(
        "chromadb_ready",
        collection=settings.chroma.collection_name,
        document_chunks=doc_count,
    )
    return collection


def _load_bm25_index(index_path: Path) -> object | None:
    """
    Deserialise the BM25 index from disk, if it exists.

    Returns None on first run (before ingestion). The retrieval layer
    checks for None and falls back to dense-only search with a warning.
    """
    if not index_path.exists():
        log.warning(
            "bm25_index_not_found",
            path=str(index_path),
            advice=(
                "Run `python scripts/ingest_corpus.py` to build the index. "
                "Falling back to dense-only retrieval until then."
            ),
        )
        return None

    log.info("loading_bm25_index", path=str(index_path))
    t0 = time.perf_counter()
    with open(index_path, "rb") as fh:
        bm25 = pickle.load(fh)  # noqa: S301 — trusted internal artifact
    elapsed = (time.perf_counter() - t0) * 1000
    log.info("bm25_index_loaded", elapsed_ms=round(elapsed, 1))
    return bm25


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan context manager — runs startup then shutdown logic.

    Resources are stored on `app.state` and retrieved via FastAPI's
    dependency injection system (see `app/api/dependencies.py`).

    Startup order
    -------------
    1. Parse & validate Settings (fails fast on bad env vars).
    2. Configure structlog (before any other logging calls).
    3. Load SentenceTransformer embedding model.
    4. Initialise ChromaDB persistent client + collection.
    5. Load BM25 corpus index from disk.

    Shutdown order
    --------------
    1. Log graceful shutdown intent.
    2. (ChromaDB PersistentClient has no explicit close; data is flushed
       automatically. Extend here if using a network ChromaDB server.)
    """
    # ── STARTUP ───────────────────────────────────────────────────────────────
    settings: Settings = get_settings()

    configure_logging(
        log_level=settings.logging.level,
        json_logs=settings.logging.json_logs,
    )

    log.info(
        "startup_begin",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )

    # 1. Embedding model
    embedder: SentenceTransformer = _init_embedding_model(settings)
    app.state.embedder = embedder

    # 2. ChromaDB
    chroma_collection: chromadb.Collection = _init_chroma(settings)
    app.state.chroma_collection = chroma_collection

    # 3. BM25 index (may be None on first run)
    bm25_retriever: object | None = _load_bm25_index(settings.bm25.index_path)
    app.state.bm25_retriever = bm25_retriever

    # 4. Expose settings on app.state for convenience
    app.state.settings = settings

    log.info(
        "startup_complete",
        app=settings.app_name,
        chroma_chunks=chroma_collection.count(),
        bm25_ready=bm25_retriever is not None,
    )

    yield  # ← application runs here

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    log.info("shutdown_begin", app=settings.app_name)
    # ChromaDB PersistentClient persists automatically; no explicit close needed.
    # Add explicit teardown here if resources require it (e.g. GPU model unload).
    log.info("shutdown_complete", app=settings.app_name)


# ──────────────────────────────────────────────────────────────────────────────
# Middleware
# ──────────────────────────────────────────────────────────────────────────────


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Stamp every incoming request with a UUID trace ID.

    Resolution order (first found wins):
      1. Client-provided X-Request-ID header (allows end-to-end tracing).
      2. Auto-generated UUID4.

    The resolved ID is:
      - Stored in `request.state.request_id` for handler access.
      - Echoed back in the X-Request-ID response header.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        settings: Settings = request.app.state.settings
        header: str = settings.api.request_id_header
        request_id: str = request.headers.get(header) or str(uuid.uuid4())
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers[header] = request_id
        return response


class StructlogMiddleware(BaseHTTPMiddleware):
    """
    Bind the request_id into structlog context-vars and log request lifecycle.

    Emits two log events per request:
      - `http_request_received`  at the start (method, path, client IP).
      - `http_request_complete`  at the end (status_code, latency_ms).

    Context-vars are cleared in a `finally` block to prevent cross-request
    contamination in async worker reuse scenarios.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id: str = getattr(request.state, "request_id", str(uuid.uuid4()))
        bind_request_context(request_id)

        t_start = time.perf_counter()
        log.info(
            "http_request_received",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

        try:
            response: Response = await call_next(request)
            elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
            log.info(
                "http_request_complete",
                status_code=response.status_code,
                elapsed_ms=elapsed_ms,
            )
            return response
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
            log.error(
                "http_request_error",
                exc_type=type(exc).__name__,
                elapsed_ms=elapsed_ms,
                exc_info=True,
            )
            raise
        finally:
            clear_request_context()


# ──────────────────────────────────────────────────────────────────────────────
# Exception handlers
# ──────────────────────────────────────────────────────────────────────────────


async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    422 Unprocessable Entity – Pydantic validation failures.

    Returns a structured list of field-level errors so API clients can surface
    precise validation messages without parsing raw exception strings.
    """
    errors = [
        {
            "field": " → ".join(str(loc) for loc in err["loc"]),
            "message": err["msg"],
            "type": err["type"],
        }
        for err in exc.errors()
    ]
    log.warning(
        "request_validation_error",
        path=request.url.path,
        error_count=len(errors),
    )
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed.",
            "errors": errors,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def _http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Pass-through handler that adds request_id to every HTTP error body."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def _unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    500 Internal Server Error – catch-all for unexpected exceptions.

    Logs the full traceback internally but returns only an opaque error_id
    to the client to avoid leaking implementation details.
    """
    error_id = str(uuid.uuid4())
    log.error(
        "unhandled_exception",
        error_id=error_id,
        exc_type=type(exc).__name__,
        traceback=traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred. Please retry or contact support.",
            "error_id": error_id,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """
    Construct and configure the FastAPI application.

    Separated from module-level instantiation so that test fixtures can call
    `create_app()` with a patched environment / settings without side effects.

    Returns
    -------
    FastAPI
        A fully configured application instance ready for Uvicorn.
    """
    settings: Settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Agentic RAG system for EEG / Neuroscience literature. "
            "Combines ChromaDB dense retrieval, BM25 sparse search, "
            "Reciprocal Rank Fusion, and GPT-4o generation with citation validation."
        ),
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        openapi_url="/openapi.json" if settings.environment != "production" else None,
        lifespan=lifespan,
        debug=settings.debug,
    )

    # ── Middleware (registration order = outermost first) ─────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    # Starlette BaseHTTPMiddleware wrappers are added last (run first):
    app.add_middleware(StructlogMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # ── Exception handlers ────────────────────────────────────────────────────
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, _http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unhandled_exception_handler)

    # ── Inline health routes (no router dependency) ───────────────────────────
    @app.get(
        "/health/live",
        tags=["health"],
        summary="Liveness probe",
        description="Returns 200 if the process is alive. Used by Kubernetes liveness checks.",
    )
    async def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/health/ready",
        tags=["health"],
        summary="Readiness probe",
        description=(
            "Returns 200 when all startup resources are loaded. "
            "Returns 503 if ChromaDB or the embedding model are not ready."
        ),
    )
    async def readiness(request: Request) -> JSONResponse:
        chroma_ok: bool = False
        bm25_ok: bool = False
        embedder_ok: bool = False
        chroma_count: int = 0

        try:
            collection: chromadb.Collection = request.app.state.chroma_collection
            chroma_count = collection.count()
            chroma_ok = True
        except Exception:
            log.warning("readiness_chroma_unavailable", exc_info=True)

        try:
            bm25_ok = request.app.state.bm25_retriever is not None
        except Exception:
            log.warning("readiness_bm25_unavailable", exc_info=True)

        try:
            embedder_ok = request.app.state.embedder is not None
        except Exception:
            log.warning("readiness_embedder_unavailable", exc_info=True)

        payload = {
            "status": "ready" if (chroma_ok and embedder_ok) else "not_ready",
            "chroma": {"ok": chroma_ok, "document_chunks": chroma_count},
            "bm25": {"ok": bm25_ok},
            "embedder": {"ok": embedder_ok},
        }
        status_code = 200 if (chroma_ok and embedder_ok) else 503
        return JSONResponse(content=payload, status_code=status_code)

    # ── Domain routers ────────────────────────────────────────────────────────
    # Wired in Step 7 when app/api/routes.py is implemented:
    #
    from app.api.routes import router as api_router
    app.include_router(api_router, prefix="/api/v1")


    return app


# ──────────────────────────────────────────────────────────────────────────────
# ASGI entry-point
# ──────────────────────────────────────────────────────────────────────────────

# Module-level `app` instance consumed by Uvicorn:
#   uvicorn app.main:app --reload
app: FastAPI = create_app()
