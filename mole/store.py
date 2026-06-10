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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def _next_claim_id_from_max(current_max: int) -> str:
    return f"clm_{current_max + 1:06d}"


def next_task_id(repo_root: Path) -> str:
    """Return the next available task_NNNNNN id by scanning pending.jsonl."""
    max_n = 0
    for rec in _iter_jsonl(_pending_path(repo_root)):
        n = _parse_numeric_id(rec.get("id", ""), "task_")
        if n is not None and n > max_n:
            max_n = n
    return f"task_{max_n + 1:06d}"


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
    # Hard invariant: content is never stored
    assert "text" not in record, "content must never be stored in sources.jsonl"
    _append_jsonl(_sources_path(repo_root), record)
    return record


# ---------------------------------------------------------------------------
# Claim append
# ---------------------------------------------------------------------------

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
    _append_jsonl(_claims_path(repo_root), record)
    return record


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
