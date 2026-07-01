"""
compile(repo_root) -> writes data/artifact.json

The site-ready artifact: active claims + edges + counts.
Deterministic: sorted by id, no wall-clock values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mole import store


def compile(repo_root: Path, run_id: str | None = None) -> dict[str, Any]:
    """
    Build and write data/artifact.json.

    Returns the artifact dict.
    """
    sources = store.load_all_sources(repo_root)
    claims = store.load_all_claims(repo_root)
    edges = store.load_all_edges(repo_root)
    tasks = store.load_all_tasks(repo_root)

    # Active claims: status != 'retired', sorted by id
    active_claims = sorted(
        [c for c in claims if c.get("status") != "retired"],
        key=lambda c: c.get("id", ""),
    )

    # Edges between active claims only, sorted by (a, b).
    # Retired claims are excluded from the artifact (SCHEMA.md), so edges
    # touching them would dangle; mirrors the active_ids filter in atlas.py.
    active_ids = {c["id"] for c in active_claims}
    sorted_edges = sorted(
        (e for e in edges if e.get("a") in active_ids and e.get("b") in active_ids),
        key=lambda e: (e.get("a", ""), e.get("b", "")),
    )

    # Task counts by kind
    task_counts: dict[str, int] = {}
    for task in tasks:
        if task.get("status") != "pending":
            continue
        kind = task.get("kind", "unknown")
        task_counts[kind] = task_counts.get(kind, 0) + 1

    # Determine generated_run: prefer the caller's authoritative run id; fall
    # back to the last run_id in append order (sources.jsonl is append-only,
    # so the last record is the chronologically latest — never use max(),
    # which sorts 'manual-*' ids above numeric ones).
    run_ids = [s.get("run_id", "") for s in sources if s.get("run_id")]
    generated_run = run_id or (run_ids[-1] if run_ids else "unknown")

    artifact: dict[str, Any] = {
        "generated_run": generated_run,
        "counts": {
            "sources": len(sources),
            "claims": len(active_claims),
            "edges": len(sorted_edges),
            "pending_tasks": task_counts,
        },
        "claims": active_claims,
        "edges": sorted_edges,
    }

    out_path = repo_root / "data" / "artifact.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, ensure_ascii=False)

    return artifact
