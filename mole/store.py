"""
JSONL read/append helpers for the mole data files.

Files:
  data/sources.jsonl
  data/claims.jsonl
  data/edges.jsonl
  queue/pending.jsonl

ID conventions:
  claims   -> clm_NNNNNN  (zero-padded 6 digits)
  tasks    -> task_NNNNNN (zero-padded 6 digits)

IDs are stable and monotonic: derived by scanning existing files, never from
the clock.  Content is never stored in sources.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _sources_path(repo_root: Path) -> Path:
    return repo_root / "data" / "sources.jsonl"


def _claims_path(repo_root: Path) -> Path:
    return repo_root / "data" / "claims.jsonl"


def _edges_path(repo_root: Path) -> Path:
    return repo_root / "data" / "edges.jsonl"


def _pending_path(repo_root: Path) -> Path:
    return repo_root / "queue" / "pending.jsonl"


def _theses_path(repo_root: Path) -> Path:
    return repo_root / "data" / "theses.jsonl"


# ---------------------------------------------------------------------------
# Low-level JSONL helpers
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed records from a JSONL file (skip blank lines)."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a single record to a JSONL file, creating parents if needed."""
    # Hard invariant: source records never store content (SCHEMA.md). Enforced
    # here so it holds for any caller and under `python -O`.
    if path.name == "sources.jsonl" and "text" in record:
        raise ValueError("content must never be stored in sources.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _rewrite_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically rewrite a JSONL file (temp file + os.replace)."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# ID assignment
# ---------------------------------------------------------------------------

def _parse_numeric_id(id_str: str, prefix: str) -> int | None:
    """Extract the numeric portion of an id like 'clm_000042' -> 42."""
    if id_str.startswith(prefix):
        try:
            return int(id_str[len(prefix):])
        except ValueError:
            pass
    return None


def next_claim_id(repo_root: Path) -> str:
    """Return the next available clm_NNNNNN id by scanning claims.jsonl."""
    max_n = 0
    for rec in _iter_jsonl(_claims_path(repo_root)):
        n = _parse_numeric_id(rec.get("id", ""), "clm_")
        if n is not None and n > max_n:
            max_n = n
    return f"clm_{max_n + 1:06d}"


def next_task_id(repo_root: Path) -> str:
    """Return the next available task_NNNNNN id by scanning pending.jsonl."""
    max_n = 0
    for rec in _iter_jsonl(_pending_path(repo_root)):
        n = _parse_numeric_id(rec.get("id", ""), "task_")
        if n is not None and n > max_n:
            max_n = n
    return f"task_{max_n + 1:06d}"


def next_thesis_id(repo_root: Path) -> str:
    """Return the next available th_NNNNNN id by scanning theses.jsonl."""
    max_n = 0
    for rec in _iter_jsonl(_theses_path(repo_root)):
        n = _parse_numeric_id(rec.get("id", ""), "th_")
        if n is not None and n > max_n:
            max_n = n
    return f"th_{max_n + 1:06d}"


# ---------------------------------------------------------------------------
# Source deduplication
# ---------------------------------------------------------------------------

def load_seen_sources(repo_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Return a mapping from (feed, item_key) -> source record for all previously
    ingested sources.
    """
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in _iter_jsonl(_sources_path(repo_root)):
        feed = rec.get("feed", "")
        # item_key is encoded in the id: src_<feed>_<item_key>
        src_id = rec.get("id", "")
        prefix = f"src_{feed}_"
        if src_id.startswith(prefix):
            item_key = src_id[len(prefix):]
        else:
            item_key = rec.get("item_key", "")
        seen[(feed, item_key)] = rec
    return seen


def is_seen_source(
    seen: dict[tuple[str, str], dict[str, Any]],
    feed: str,
    item_key: str,
) -> bool:
    return (feed, item_key) in seen


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Source append (no content stored)
# ---------------------------------------------------------------------------

def append_source(
    repo_root: Path,
    *,
    feed: str,
    item_key: str,
    url: str,
    title: str,
    author: str,
    published: str,
    sha256: str,
    claim_count: int,
    run_id: str,
) -> dict[str, Any]:
    """Append a source record. Asserts that no 'text' field is present."""
    src_id = f"src_{feed}_{item_key}"
    record: dict[str, Any] = {
        "id": src_id,
        "feed": feed,
        "url": url,
        "title": title,
        "author": author,
        "published": published,
        "content_sha256": sha256,
        "claim_count": claim_count,
        "run_id": run_id,
    }
    _append_jsonl(_sources_path(repo_root), record)
    return record


# ---------------------------------------------------------------------------
# Claim append
# ---------------------------------------------------------------------------

def make_claim_record(
    *,
    claim_id: str,
    source_id: str,
    text: str,
    claim_type: str,
    support_in_text: float,
    quote: str,
    run_id: str,
    status: str = "extracted",
    refines_claim: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": claim_id,
        "source_id": source_id,
        "text": text,
        "type": claim_type,
        "support_in_text": support_in_text,
        "quote": quote,
        "run_id": run_id,
        "status": status,
    }
    if refines_claim is not None:
        record["refines_claim"] = refines_claim
    return record


def append_claim(
    repo_root: Path,
    *,
    claim_id: str,
    source_id: str,
    text: str,
    claim_type: str,
    support_in_text: float,
    quote: str,
    run_id: str,
    status: str = "extracted",
    refines_claim: str | None = None,
) -> dict[str, Any]:
    record = make_claim_record(
        claim_id=claim_id,
        source_id=source_id,
        text=text,
        claim_type=claim_type,
        support_in_text=support_in_text,
        quote=quote,
        run_id=run_id,
        status=status,
        refines_claim=refines_claim,
    )
    _append_jsonl(_claims_path(repo_root), record)
    return record


def append_claims_batch(repo_root: Path, records: list[dict[str, Any]]) -> None:
    """Append several claim records in one file open (shrinks the crash window
    between an item's claim writes and its source record)."""
    if not records:
        return
    path = _claims_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def retire_claims_for_source(repo_root: Path, source_id: str) -> int:
    """Flip status to 'retired' for all non-retired claims of a source.

    Used when a source's content changed and its claims are re-extracted.
    Atomic rewrite; returns the number of claims retired.
    """
    path = _claims_path(repo_root)
    claims = list(_iter_jsonl(path))
    changed = 0
    for c in claims:
        if c.get("source_id") == source_id and c.get("status") != "retired":
            c["status"] = "retired"
            changed += 1
    if changed:
        _rewrite_jsonl(path, claims)
    return changed


def update_source(
    repo_root: Path,
    *,
    feed: str,
    item_key: str,
    sha256: str,
    claim_count: int,
    run_id: str,
) -> None:
    """Update an existing source line in place after a content change.

    The id stays stable (ids are never reused); only the hash, claim count,
    and run_id move forward. Atomic rewrite.
    """
    path = _sources_path(repo_root)
    records = list(_iter_jsonl(path))
    src_id = f"src_{feed}_{item_key}"
    for rec in records:
        if rec.get("id") == src_id:
            rec["content_sha256"] = sha256
            rec["claim_count"] = claim_count
            rec["run_id"] = run_id
    _rewrite_jsonl(path, records)


def prune_orphan_claims(repo_root: Path) -> int:
    """Remove claims whose source_id has no record in sources.jsonl —
    leftovers of a run interrupted between claim and source appends.

    Safe because tasks/edges are only enqueued after an item's source record
    is persisted, so orphan claims are never referenced elsewhere.
    Returns the number of claims removed.
    """
    src_ids = {rec.get("id", "") for rec in _iter_jsonl(_sources_path(repo_root))}
    claims_path = _claims_path(repo_root)
    claims = list(_iter_jsonl(claims_path))
    kept = [c for c in claims if c.get("source_id", "") in src_ids]
    removed = len(claims) - len(kept)
    if removed:
        _rewrite_jsonl(claims_path, kept)
    return removed


# ---------------------------------------------------------------------------
# Task append
# ---------------------------------------------------------------------------

def append_task(
    repo_root: Path,
    *,
    task_id: str,
    kind: str,
    payload: dict[str, Any],
    created_run: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": task_id,
        "kind": kind,
        "payload": payload,
        "created_run": created_run,
        "status": "pending",
    }
    _append_jsonl(_pending_path(repo_root), record)
    return record


# ---------------------------------------------------------------------------
# Read helpers for pipeline / compile
# ---------------------------------------------------------------------------

def load_all_claims(repo_root: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(_claims_path(repo_root)))


def load_all_sources(repo_root: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(_sources_path(repo_root)))


def load_all_edges(repo_root: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(_edges_path(repo_root)))


def load_all_tasks(repo_root: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(_pending_path(repo_root)))


def append_theses_batch(repo_root: Path, records: list[dict[str, Any]]) -> None:
    """Append thesis records (data/theses.jsonl) in one file open."""
    if not records:
        return
    path = _theses_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def load_all_theses(repo_root: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(_theses_path(repo_root)))


def update_tasks(repo_root: Path, predicate, *, status: str,
                 note: str | None = None) -> int:
    """Flip status (and optionally add a note) on every task matching
    predicate. Atomic rewrite; returns the number of tasks updated."""
    path = _pending_path(repo_root)
    tasks = list(_iter_jsonl(path))
    changed = 0
    for t in tasks:
        if predicate(t):
            t["status"] = status
            if note is not None:
                t["note"] = note
            changed += 1
    if changed:
        _rewrite_jsonl(path, tasks)
    return changed


def bump_source_extraction(repo_root: Path, *, source_id: str,
                           claim_count: int, fable_run: str) -> None:
    """Record a worker-side re-extraction on the source row WITHOUT touching
    content_sha256 (that hash belongs to the feed-cleaned text and drives the
    cron's changed-item detection — see mole/fable.py module docstring)."""
    path = _sources_path(repo_root)
    records = list(_iter_jsonl(path))
    for rec in records:
        if rec.get("id") == source_id:
            rec["claim_count"] = claim_count
            rec["fable_run"] = fable_run
    _rewrite_jsonl(path, records)


def load_live_questions(repo_root: Path) -> list[dict[str, Any]]:
    """Question-layer nodes (data/questions.jsonl), excluding merged-away ones.

    Returns [] when the question layer does not exist yet, so the mole
    degrades gracefully on repos without it."""
    path = repo_root / "data" / "questions.jsonl"
    if not path.exists():
        return []
    return [q for q in _iter_jsonl(path) if not q.get("merged_into")]
