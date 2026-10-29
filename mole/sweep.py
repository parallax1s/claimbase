"""Retirement sweep for junk-shaped claims already in data/claims.jsonl.

ISSUES.md ("Extractor pulls MathJax CSS fragments as claims", 2026-08-25):
the regex-era extractor emitted CSS rules / MathJax markup / raw JS from
LessWrong and AlignmentForum pages as claims. `mole/dispatch.py::is_junk_shape`
stops new ones at ingestion; this sweep applies the same shape test to the
rows already in the corpus and retires the hits through the normal
refine/retire path: the row stays in the file with ``status: retired`` (plus
``retired_reason`` / ``retired_by`` for provenance), and any pending task
whose payload names a swept claim is marked ``obsolete`` — exactly what a
fable pass does when it supersedes a source's claims.

    python -m mole sweep-junk --run-id sweep-20261029           # dry run
    python -m mole sweep-junk --run-id sweep-20261029 --apply   # rewrite

Deterministic, zero LLM calls. Run ``python -m mole compile`` afterwards so
artifact/atlas drop the retired rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mole import store
from mole.dispatch import is_junk_shape

REASON = "junk_shape"
_CLAIM_REF_KEYS = ("claim_id", "a", "b")


def find_junk(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Live (non-retired) rows whose text or quote has a junk shape."""
    return [
        c for c in claims
        if c.get("status") != "retired"
        and (is_junk_shape(c.get("text", "")) or is_junk_shape(c.get("quote", "")))
    ]


def _task_refs_claim(task: dict[str, Any], ids: set[str]) -> bool:
    payload = task.get("payload") or {}
    return any(payload.get(k) in ids for k in _CLAIM_REF_KEYS)


def sweep_junk(repo_root: Path, run_id: str, apply: bool = False) -> dict[str, Any]:
    """Retire junk-shaped live claims. Returns a summary; writes only if apply."""
    claims = store.load_all_claims(repo_root)
    junk = find_junk(claims)
    junk_ids = {c["id"] for c in junk}
    by_source: dict[str, int] = {}
    for c in junk:
        by_source[c.get("source_id", "?")] = by_source.get(c.get("source_id", "?"), 0) + 1
    pending_hit = [
        t for t in store.load_all_tasks(repo_root)
        if t.get("status") == "pending" and _task_refs_claim(t, junk_ids)
    ]
    summary: dict[str, Any] = {
        "run_id": run_id,
        "live_claims": sum(1 for c in claims if c.get("status") != "retired"),
        "junk_found": len(junk),
        "sources_touched": len(by_source),
        "pending_tasks_referencing": len(pending_hit),
        "applied": apply,
        "sample": [c["text"][:100] for c in junk[:5]],
    }
    if not apply or not junk:
        return summary
    for c in claims:
        if c["id"] in junk_ids:
            c["status"] = "retired"
            c["retired_reason"] = REASON
            c["retired_by"] = run_id
    store._rewrite_jsonl(store._claims_path(repo_root), claims)
    summary["tasks_obsoleted"] = store.update_tasks(
        repo_root,
        lambda t: t.get("status") == "pending" and _task_refs_claim(t, junk_ids),
        status="obsolete",
        note=f"claim retired as {REASON} by {run_id}",
    )
    return summary
