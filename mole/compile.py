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


def compile(repo_root: Path) -> dict[str, Any]:
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

    # All edges sorted by (a, b)
    sorted_edges = sorted(edges, key=lambda e: (e.get("a", ""), e.get("b", "")))

    # Task counts by kind
    task_counts: dict[str, int] = {}
    for task in tasks:
        kind = task.get("kind", "unknown")
        task_counts[kind] = task_counts.get(kind, 0) + 1

    # Determine generated_run: latest run_id found in sources (lexicographically last)
    run_ids = [s.get("run_id", "") for s in sources if s.get("run_id")]
    generated_run = max(run_ids) if run_ids else "unknown"

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
