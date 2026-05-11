"""
app/schema.py
=============
Pydantic v2 schemas for the Agentic EEG Research Copilot.

Covers:
  - Citation          – a single retrieved document chunk with provenance
  - QueryRequest      – incoming user query with retrieval hyper-parameters
  - QueryResponse     – final API response envelope
  - AgentState        – LangGraph node-to-node state (TypedDict + Pydantic)
  - RetrievalConfig   – runtime knobs for hybrid search
  - DocumentChunk     – normalised chunk returned from any retriever

All fields are fully type-hinted; no Optional[X] without a default.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import TypedDict


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────


class RetrievalStrategy(str, Enum):
    """Supported hybrid-search fusion strategies."""

    RRF = "rrf"              # Reciprocal Rank Fusion (default)
    LINEAR = "linear"        # Weighted linear combination
    BM25_ONLY = "bm25_only"  # Sparse retrieval only (ablation)
    DENSE_ONLY = "dense_only"  # Dense retrieval only (ablation)


class EEGDomain(str, Enum):
    """High-level EEG research sub-domains for metadata filtering."""

    BCI = "bci"                        # Brain-Computer Interface
    EPILEPSY = "epilepsy"
    SLEEP = "sleep"
    COGNITIVE = "cognitive"
    SIGNAL_PROCESSING = "signal_processing"
    CLINICAL = "clinical"
    GENERAL = "general"


class AgentStep(str, Enum):
    """Named LangGraph node identifiers (matches graph node keys)."""

    QUERY_ANALYSIS = "query_analysis"
    HYBRID_RETRIEVAL = "hybrid_retrieval"
    RERANKING = "reranking"
    GENERATION = "generation"
    CITATION_VALIDATION = "citation_validation"
    RESPONSE_FORMATTING = "response_formatting"


# ──────────────────────────────────────────────────────────────────────────────
# Sub-models
# ──────────────────────────────────────────────────────────────────────────────


class Citation(BaseModel):
    """
    A single retrieved document chunk with full provenance metadata.
    Used both as a retrieval unit and as an inline citation in responses.
    """

    citation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique citation identifier (UUID4).",
    )
    chunk_id: str = Field(
        ...,
        description="Stable chunk identifier within the vector store.",
    )
    document_id: str = Field(
        ...,
        description="Parent document identifier (e.g. DOI or internal ID).",
    )
    source_file: str = Field(
        ...,
        description="Original filename or S3 URI of the source PDF.",
    )
    page_number: int | None = Field(
        default=None,
        ge=1,
        description="1-indexed page number within the source PDF.",
    )
    section_title: str | None = Field(
        default=None,
        description="Nearest section heading above the chunk (if parseable).",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Raw text content of the retrieved chunk.",
    )
    dense_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine-similarity score from the dense retriever.",
    )
    bm25_score: float = Field(
        ...,
        ge=0.0,
        description="BM25 relevance score from the sparse retriever.",
    )
    fusion_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final fused score used for ranking.",
    )
    domain: EEGDomain = Field(
        default=EEGDomain.GENERAL,
        description="EEG sub-domain tag for this chunk.",
    )
    authors: list[str] = Field(
        default_factory=list,
        description="Author list extracted from document metadata.",
    )
    year: int | None = Field(
        default=None,
        ge=1900,
        le=2100,
        description="Publication year.",
    )
    doi: str | None = Field(
        default=None,
        description="Digital Object Identifier.",
    )

    model_config = {"frozen": True}

    @field_validator("text")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class DocumentChunk(BaseModel):
    """
    Normalised representation of a PDF chunk after parsing and splitting.
    Stored in ChromaDB and consumed by both BM25 and dense retrievers.
    """

    chunk_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable unique identifier for this chunk.",
    )
    document_id: str = Field(..., description="Parent document identifier.")
    source_file: str = Field(..., description="Original source filename or URI.")
    page_number: int | None = Field(default=None, ge=1)
    section_title: str | None = Field(default=None)
    text: str = Field(..., min_length=1)
    token_count: int = Field(..., ge=1, description="Approximate token count.")
    embedding: list[float] | None = Field(
        default=None,
        description="Pre-computed dense embedding vector.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extra metadata (authors, year, DOI, etc.).",
    )

    @field_validator("text")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class RetrievalConfig(BaseModel):
    """
    Runtime hyper-parameters controlling hybrid search behaviour.
    Can be overridden per-request via QueryRequest.
    """

    strategy: RetrievalStrategy = Field(
        default=RetrievalStrategy.RRF,
        description="Fusion strategy for combining dense and sparse scores.",
    )
    top_k_dense: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Number of candidates from the dense (ChromaDB) retriever.",
    )
    top_k_bm25: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Number of candidates from the BM25 sparse retriever.",
    )
    top_k_final: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of chunks passed to the generation node after fusion.",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="Reciprocal Rank Fusion constant k (larger → less aggressive).",
    )
    dense_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for dense score in LINEAR fusion (1-w for BM25).",
    )
    domain_filter: EEGDomain | None = Field(
        default=None,
        description="Optional ChromaDB metadata filter on EEG sub-domain.",
    )
    year_min: int | None = Field(
        default=None,
        ge=1900,
        description="Filter to papers published at or after this year.",
    )
    year_max: int | None = Field(
        default=None,
        le=2100,
        description="Filter to papers published at or before this year.",
    )

    @model_validator(mode="after")
    def validate_year_range(self) -> "RetrievalConfig":
        if self.year_min and self.year_max and self.year_min > self.year_max:
            raise ValueError(
                f"year_min ({self.year_min}) must be ≤ year_max ({self.year_max})."
            )
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response
# ──────────────────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """
    Incoming API request payload for the /query endpoint.
    """

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Client-facing request trace ID (auto-generated if omitted).",
    )
    query: str = Field(
        ...,
        min_length=3,
        max_length=2048,
        description="Natural-language research question about EEG.",
        examples=["What are the best feature extraction methods for motor imagery BCI?"],
    )
    retrieval_config: RetrievalConfig = Field(
        default_factory=RetrievalConfig,
        description="Hybrid search hyper-parameters; defaults are research-tuned.",
    )
    conversation_history: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Prior turns in OpenAI message format: "
            "[{'role': 'user'|'assistant', 'content': '...'}]"
        ),
    )
    stream: bool = Field(
        default=False,
        description="If True, response will be streamed via Server-Sent Events.",
    )

    @field_validator("conversation_history")
    @classmethod
    def validate_message_format(
        cls, v: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        for msg in v:
            if "role" not in msg or "content" not in msg:
                raise ValueError(
                    "Each conversation_history item must have 'role' and 'content' keys."
                )
            if msg["role"] not in {"user", "assistant", "system"}:
                raise ValueError(
                    f"Invalid role '{msg['role']}'. Must be 'user', 'assistant', or 'system'."
                )
        return v


class QueryResponse(BaseModel):
    """
    Final API response envelope returned from the /query endpoint.
    """

    request_id: str = Field(..., description="Mirrors the request trace ID.")
    answer: str = Field(
        ...,
        description="Generated answer with inline citation markers (e.g. [1], [2]).",
    )
    citations: list[Citation] = Field(
        ...,
        description="Ordered list of source chunks referenced in the answer.",
    )
    steps_executed: list[AgentStep] = Field(
        ...,
        description="Ordered list of LangGraph nodes executed for this request.",
    )
    retrieval_config_used: RetrievalConfig = Field(
        ...,
        description="Actual retrieval config applied (post-defaults resolution).",
    )
    latency_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Per-step wall-clock latency in milliseconds.",
    )
    token_usage: dict[str, int] = Field(
        default_factory=dict,
        description="Token counts: {'prompt': N, 'completion': N, 'total': N}.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# LangGraph Agent State (TypedDict for graph compatibility)
# ──────────────────────────────────────────────────────────────────────────────


class AgentState(TypedDict, total=False):
    """
    Shared mutable state flowing through every LangGraph node.

    All keys are optional (total=False) so nodes only update what they touch.
    Downstream nodes must treat absent keys as their zero-value equivalents.

    Key design choices:
      - `raw_chunks` holds the unfiltered union of dense + BM25 results.
      - `reranked_chunks` is populated by the reranking node and consumed by generation.
      - `citations` is the validated subset of reranked_chunks actually cited.
      - `error` carries any non-fatal diagnostic message for observability.
    """

    # ── Input ─────────────────────────────────────────────────────────────────
    request_id: Annotated[str, "Trace ID propagated from QueryRequest"]
    query: Annotated[str, "Original user query string"]
    expanded_query: Annotated[str, "Query after synonym/acronym expansion"]
    conversation_history: Annotated[
        list[dict[str, str]], "Prior turns in OpenAI message format"
    ]
    retrieval_config: Annotated[RetrievalConfig, "Resolved retrieval hyper-parameters"]

    # ── Retrieval ─────────────────────────────────────────────────────────────
    raw_chunks: Annotated[
        list[Citation], "Union of dense + BM25 candidates before reranking"
    ]
    reranked_chunks: Annotated[
        list[Citation], "Top-K chunks after RRF/linear fusion"
    ]

    # ── Generation ────────────────────────────────────────────────────────────
    draft_answer: Annotated[str, "Raw LLM output before citation validation"]
    citations: Annotated[
        list[Citation], "Validated citations actually referenced in the answer"
    ]
    final_answer: Annotated[str, "Post-validation answer with corrected citation markers"]

    # ── Observability ─────────────────────────────────────────────────────────
    steps_executed: Annotated[list[AgentStep], "Ordered log of completed nodes"]
    latency_ms: Annotated[dict[str, float], "Per-node wall-clock milliseconds"]
    token_usage: Annotated[dict[str, int], "Cumulative token accounting"]
    error: Annotated[str | None, "Non-fatal error message for diagnostics"]
    response: Annotated[QueryResponse, "Final formatted API response"]