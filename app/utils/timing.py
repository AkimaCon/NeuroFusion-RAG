"""
app/utils/timing.py
===================
Lightweight latency tracking utilities for LangGraph nodes and API handlers.

Provides:
  - `timed_node`  – async decorator that records wall-clock ms into AgentState
  - `Timer`       – reusable async context manager for ad-hoc measurements
  - `record_latency` – helper to merge a float into AgentState["latency_ms"]

Design notes:
  - Uses `time.perf_counter()` (monotonic, sub-millisecond resolution).
  - Thread-safe: writes are local to the coroutine before merging into state.
  - No external dependencies; pure stdlib + typing only.

Example (decorator)
-------------------
    from app.utils.timing import timed_node

    @timed_node("hybrid_retrieval")
    async def run_retrieval(state: AgentState) -> AgentState:
        ...
        return state

Example (context manager)
-------------------------
    from app.utils.timing import Timer

    async with Timer("bm25_query") as t:
        results = await bm25.query(text)
    print(t.elapsed_ms)   # → e.g. 12.4
"""

from __future__ import annotations

import time
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, TypeVar

from app.schema import AgentState

F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, AgentState]])


# ──────────────────────────────────────────────────────────────────────────────
# Decorator
# ──────────────────────────────────────────────────────────────────────────────


def timed_node(node_name: str) -> Callable[[F], F]:
    """
    Async decorator that wraps a LangGraph node function.

    Measures wall-clock time of the wrapped coroutine and merges the elapsed
    milliseconds into `state["latency_ms"][node_name]`.

    The wrapped function must accept a single positional argument of type
    `AgentState` and return an `AgentState`.

    Parameters
    ----------
    node_name:
        Key written into `state["latency_ms"]`. Should match `AgentStep` values
        (e.g. "hybrid_retrieval", "reranking") for consistency.

    Returns
    -------
    Decorator that preserves the original function's signature and docstring.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(state: AgentState, *args: Any, **kwargs: Any) -> AgentState:
            t_start = time.perf_counter()
            try:
                result: AgentState = await func(state, *args, **kwargs)
            except Exception:
                # Still record partial timing before re-raising
                elapsed = (time.perf_counter() - t_start) * 1000.0
                _merge_latency(state, node_name, elapsed)
                raise
            elapsed = (time.perf_counter() - t_start) * 1000.0
            _merge_latency(result, node_name, elapsed)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


# ──────────────────────────────────────────────────────────────────────────────
# Context manager
# ──────────────────────────────────────────────────────────────────────────────


class Timer:
    """
    Async context manager for measuring elapsed wall-clock time.

    Attributes
    ----------
    label:
        Descriptive label for the timed block (used in logs).
    elapsed_ms:
        Wall-clock milliseconds elapsed inside the `async with` block.
        Available after `__aexit__` completes.

    Example
    -------
        async with Timer("chroma_query") as t:
            hits = await collection.query(...)
        log.info("chroma_query_done", elapsed_ms=t.elapsed_ms)
    """

    def __init__(self, label: str) -> None:
        self.label: str = label
        self.elapsed_ms: float = 0.0
        self._t_start: float = 0.0

    async def __aenter__(self) -> "Timer":
        self._t_start = time.perf_counter()
        return self

    async def __aexit__(self, *_: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t_start) * 1000.0


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────


def record_latency(
    state: AgentState,
    key: str,
    elapsed_ms: float,
) -> AgentState:
    """
    Immutably merge a single latency entry into AgentState.

    Creates the `latency_ms` dict if absent. Returns the same state dict
    (modified in-place – safe within a single LangGraph node).

    Parameters
    ----------
    state:
        Current LangGraph agent state.
    key:
        Label for the measurement (e.g. "bm25_query", "embedding_batch_2").
    elapsed_ms:
        Measured duration in milliseconds.
    """
    _merge_latency(state, key, elapsed_ms)
    return state


# ──────────────────────────────────────────────────────────────────────────────
# Internal
# ──────────────────────────────────────────────────────────────────────────────


def _merge_latency(state: AgentState, key: str, elapsed_ms: float) -> None:
    """Mutate `state["latency_ms"]` in place; safe within a single node."""
    latency: dict[str, float] = state.get("latency_ms", {})  # type: ignore[assignment]
    latency[key] = round(elapsed_ms, 3)
    state["latency_ms"] = latency  # type: ignore[typeddict-unknown-key]
