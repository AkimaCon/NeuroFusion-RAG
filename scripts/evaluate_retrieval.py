from __future__ import annotations

import argparse
import json
from pathlib import Path


def reciprocal_rank(ranked_ids: list[str], relevant_ids: list[str]) -> float:
    """Return reciprocal rank of the first relevant result."""
    relevant = set(relevant_ids)

    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def recall_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Recall@K = relevant retrieved in top K / total relevant.

    If there are no relevant IDs, return 0.0.
    """
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved_top_k = set(ranked_ids[:k])
    found = retrieved_top_k.intersection(relevant)

    return len(found) / len(relevant)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval runs saved as JSONL. "
            "Each line should contain: {'ranked_ids': [...], 'relevant_ids': [...]}"
        )
    )
    parser.add_argument("jsonl", help="Path to JSONL evaluation file.")
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        help="K values for Recall@K. Default: 1 3 5 10",
    )

    args = parser.parse_args()

    path = Path(args.jsonl)

    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            if "ranked_ids" not in row or "relevant_ids" not in row:
                raise ValueError(
                    f"Line {line_number} must contain 'ranked_ids' and 'relevant_ids'."
                )

            rows.append(row)

    num_queries = len(rows)

    if num_queries == 0:
        print({"queries": 0, "MRR": 0.0})
        return

    mrr = sum(
        reciprocal_rank(row["ranked_ids"], row["relevant_ids"])
        for row in rows
    ) / num_queries

    metrics = {
        "queries": num_queries,
        "MRR": round(mrr, 4),
    }

    for k in args.k:
        value = sum(
            recall_at_k(row["ranked_ids"], row["relevant_ids"], k)
            for row in rows
        ) / num_queries

        metrics[f"Recall@{k}"] = round(value, 4)

    print(metrics)


if __name__ == "__main__":
    main()
