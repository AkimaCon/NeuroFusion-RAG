"""
app/config.py
=============
Centralised, environment-driven configuration for the EEG Research Copilot.

Design principles:
  - Every tunable is an env-var with a sensible research-tuned default.
  - Nested settings models group related knobs (OpenAI, ChromaDB, BM25, …).
  - A module-level `get_settings()` function (lru_cache) provides a
    process-wide singleton – safe to call from any module.
  - Field-level validators catch misconfiguration at startup, not at runtime.
  - No secret values are logged; SecretStr is used for API keys.

Usage
-----
    from app.config import get_settings
    settings = get_settings()
    print(settings.openai.model)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ──────────────────────────────────────────────────────────────────────────────
# Nested settings groups
# ──────────────────────────────────────────────────────────────────────────────


class OpenAISettings(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model: str = Field(
        default="llama3",
        description="LLM model name.",
    )

    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )

    max_tokens: int = Field(
        default=2048,
        ge=64,
        le=16384,
        description="Maximum completion tokens.",
    )

    request_timeout: float = Field(
        default=60.0,
        gt=0.0,
        description="HTTP timeout.",
    )

    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama local server host.",
    )

    # @field_validator("model")
    # @classmethod
    # def model_must_be_gpt4_class(cls, v: str) -> str:
    #     allowed_prefixes = ("gpt-4", "gpt-3.5", "o1", "o3")
    #     if not any(v.startswith(p) for p in allowed_prefixes):
    #         raise ValueError(
    #             f"Model '{v}' is not a recognised OpenAI chat model. "
    #             f"Expected prefix one of: {allowed_prefixes}"
    #         )
    #     return v


class EmbeddingSettings(BaseSettings):
    """Sentence-transformers embedding model configuration."""

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_", 
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",)

    model_name: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="HuggingFace model ID for sentence-transformers.",
    )
    device: str = Field(
        default="cpu",
        description="Torch device: 'cpu', 'cuda', or 'mps'.",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Batch size for embedding inference.",
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="L2-normalise embeddings before storing (required for cosine sim).",
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        valid = {"cpu", "cuda", "mps"}
        # Also allow cuda:N indexing (e.g. "cuda:0")
        base = v.split(":")[0]
        if base not in valid:
            raise ValueError(f"device must be one of {valid} (got '{v}').")
        return v


class ChromaDBSettings(BaseSettings):
    """ChromaDB vector store configuration."""

    model_config = SettingsConfigDict(env_prefix="CHROMA_", 
                                      env_file=".env",
                                      env_file_encoding="utf-8",
                                      extra="ignore",)

    persist_dir: Path = Field(
        default=Path("data/chroma_db"),
        description="Directory where ChromaDB persists its DuckDB + Parquet files.",
    )
    collection_name: str = Field(
        default="eeg_papers",
        description="Name of the ChromaDB collection for EEG paper chunks.",
    )
    distance_function: Literal["cosine", "l2", "ip"] = Field(
        default="cosine",
        description="Distance metric for HNSW index (cosine recommended for normalised embeddings).",
    )
    n_results_buffer: int = Field(
        default=5,
        ge=0,
        description=(
            "Extra results fetched beyond top_k_dense to absorb metadata-filtered exclusions. "
            "Actual fetch = top_k_dense + n_results_buffer."
        ),
    )

    @model_validator(mode="after")
    def ensure_persist_dir_exists(self) -> "ChromaDBSettings":
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        return self


class BM25Settings(BaseSettings):
    """BM25 sparse retriever configuration."""

    model_config = SettingsConfigDict(env_prefix="BM25_", extra="ignore")

    index_path: Path = Field(
        default=Path("data/bm25_index.pkl"),
        description="Path to the serialised BM25Okapi corpus (pickle).",
    )
    k1: float = Field(
        default=1.5,
        ge=0.0,
        description="BM25 term-frequency saturation parameter.",
    )
    b: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="BM25 document-length normalisation parameter.",
    )
    epsilon: float = Field(
        default=0.25,
        ge=0.0,
        description="BM25Plus epsilon for IDF lower-bound.",
    )

    @model_validator(mode="after")
    def ensure_index_parent_exists(self) -> "BM25Settings":
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        return self


class IngestionSettings(BaseSettings):
    """PDF ingestion / chunking pipeline configuration."""

    model_config = SettingsConfigDict(env_prefix="INGESTION_", extra="ignore")

    pdf_dir: Path = Field(
        default=Path("data/pdfs"),
        description="Directory scanned for raw EEG PDF papers.",
    )
    max_chunk_tokens: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Maximum token budget per DocumentChunk.",
    )
    chunk_overlap_tokens: int = Field(
        default=64,
        ge=0,
        description="Token overlap between consecutive chunks (sliding window).",
    )
    min_chunk_chars: int = Field(
        default=80,
        ge=1,
        description="Discard chunks shorter than this character count (likely noise).",
    )

    @model_validator(mode="after")
    def overlap_less_than_chunk(self) -> "IngestionSettings":
        if self.chunk_overlap_tokens >= self.max_chunk_tokens:
            raise ValueError(
                f"chunk_overlap_tokens ({self.chunk_overlap_tokens}) must be "
                f"< max_chunk_tokens ({self.max_chunk_tokens})."
            )
        return self


class APISettings(BaseSettings):
    """FastAPI server configuration."""

    model_config = SettingsConfigDict(env_prefix="API_", extra="ignore")

    host: str = Field(default="0.0.0.0", description="Uvicorn bind host.")
    port: int = Field(default=8000, ge=1, le=65535, description="Uvicorn bind port.")
    workers: int = Field(
        default=1,
        ge=1,
        description="Number of Uvicorn/Gunicorn worker processes.",
    )
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins. Use specific origins in production.",
    )
    request_id_header: str = Field(
        default="X-Request-ID",
        description="HTTP header name used to propagate trace IDs.",
    )
    max_request_body_mb: float = Field(
        default=50.0,
        gt=0.0,
        description="Maximum ingest request body size in megabytes.",
    )


class LoggingSettings(BaseSettings):
    """structlog / JSON logging configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG_", extra="ignore")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Root log level.",
    )
    json_logs: bool = Field(
        default=True,
        description="Emit structured JSON logs (True) or human-readable dev logs (False).",
    )
    log_llm_prompts: bool = Field(
        default=False,
        description=(
            "Include full LLM prompt text in logs. "
            "Disable in production to avoid leaking PII in log sinks."
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Root settings
# ──────────────────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """
    Root application settings.

    Composes all nested settings groups. Each group reads its own
    env-prefix independently, so OPENAI_API_KEY, CHROMA_PERSIST_DIR,
    BM25_INDEX_PATH etc. are all sourced from the environment / .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Nested models read their own prefixed env vars
    )

    # ── Metadata ──────────────────────────────────────────────────────────────
    app_name: str = Field(
        default="EEG Research Copilot",
        description="Human-readable application name (shown in OpenAPI docs).",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Semantic version string.",
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment; gates certain safety checks.",
    )
    debug: bool = Field(
        default=False,
        description="Enable FastAPI debug mode (never True in production).",
    )

    # ── Nested groups ─────────────────────────────────────────────────────────
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chroma: ChromaDBSettings = Field(default_factory=ChromaDBSettings)
    bm25: BM25Settings = Field(default_factory=BM25Settings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @model_validator(mode="after")
    def production_safety_checks(self) -> "Settings":
        if self.environment == "production":
            if self.debug:
                raise ValueError("debug must be False in production.")
            if "*" in self.api.cors_origins:
                raise ValueError(
                    "Wildcard CORS origin '*' is not permitted in production. "
                    "Set API_CORS_ORIGINS to explicit domain(s)."
                )
            if self.logging.log_llm_prompts:
                raise ValueError(
                    "log_llm_prompts must be False in production to prevent PII leakage."
                )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the process-wide Settings singleton.

    Uses lru_cache so the .env file is parsed exactly once per process.
    Call `get_settings.cache_clear()` in tests to reload between test cases.

    Example
    -------
        from app.config import get_settings
        s = get_settings()
        api_key = s.openai.api_key.get_secret_value()
    """
    return Settings()
