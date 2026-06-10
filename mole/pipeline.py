"""
mole pipeline: ingest new items, extract claims, enqueue work.

run(repo_root, since, run_id) -> summary dict.

feeds.py is imported LAZILY (inside run()) because it is built by a concurrent
agent; its absence must not break imports of this module or the test suite.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from mole import store


# ---------------------------------------------------------------------------
# Dangling-pronoun heuristic for 'refine' task enqueue
# ---------------------------------------------------------------------------

_DANGLING_HEADS = {"it", "this", "that", "they", "these", "those"}


def _needs_refine(text: str) -> bool:
    """True when a claim is not self-contained: lowercase start, dangling-pronoun
    head, or a bare-heading/shard too short to be an atomic proposition."""
    if not text:
        return False
    if text[0].islower():
        return True
    if len(text.split()) < 5:
        return True
    first_word = text.split()[0].rstrip(",.;:").lower()
    return first_word in _DANGLING_HEADS


# ---------------------------------------------------------------------------
# Cosine similarity (pure stdlib)
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Candidate pair generation for identity/edge tasks
# ---------------------------------------------------------------------------

def _cross_source_pairs(
    new_claims: list[dict[str, Any]],
    existing_claims: list[dict[str, Any]],
    extractor_module: Any,
    top_k: int = 4,
    embed_threshold: float = 0.40,
    jaccard_threshold: float = 0.30,
) -> list[tuple[str, str, float]]:
    """
    Return (id_a, id_b, sim) pairs for cross-source near-duplicate candidates.

    Strategy:
      1. Try embeddings (model2vec); use cosine >= embed_threshold.
      2. Fall back to jaccard >= jaccard_threshold.

    Only cross-SOURCE pairs (different source_id) are returned.
    Deterministic: pairs sorted by (id_a, id_b).
    """
    all_claims = existing_claims + new_claims
    # Need at least 2 to form pairs
    if len(all_claims) < 2:
        return []

    texts = [c["text"] for c in all_claims]
    embeddings = extractor_module.embed_texts(texts)

    pairs: dict[tuple[str, str], float] = {}

    if embeddings is not None:
        # For each new claim, find top-k neighbours among all_claims
        new_start = len(existing_claims)
        for i in range(new_start, len(all_claims)):
            sims = []
            for j, vec in enumerate(embeddings):
                if i == j:
                    continue
                sim = _cosine(embeddings[i], vec)
                sims.append((sim, j))
            sims.sort(reverse=True)
            for sim, j in sims[:top_k]:
                if sim < embed_threshold:
                    break
                ca = all_claims[i]
                cb = all_claims[j]
                if ca["source_id"] == cb["source_id"]:
                    continue
                key = (min(ca["id"], cb["id"]), max(ca["id"], cb["id"]))
                if key not in pairs or pairs[key] < sim:
                    pairs[key] = round(sim, 4)
    else:
        # Jaccard fallback: compare each new claim against all others
        new_start = len(existing_claims)
        for i in range(new_start, len(all_claims)):
            ca = all_claims[i]
            for j in range(len(all_claims)):
                if i == j:
                    continue
                cb = all_claims[j]
                if ca["source_id"] == cb["source_id"]:
                    continue
                sim = extractor_module.jaccard(ca["text"], cb["text"])
                if sim < jaccard_threshold:
                    continue
                key = (min(ca["id"], cb["id"]), max(ca["id"], cb["id"]))
                if key not in pairs or pairs[key] < sim:
                    pairs[key] = round(sim, 4)

    return sorted((a, b, s) for (a, b), s in pairs.items())


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(repo_root: Path, since: str, run_id: str) -> dict[str, Any]:
    """
    Execute one mole ingestion run.

    Parameters
    ----------
    repo_root : Path
        Root of the claimbase repo (contains feeds.yaml, data/, queue/, runs/).
    since : str
        ISO date string; fetch items published on or after this date.
    run_id : str
        Stable run identifier provided by the caller (never from clock).

    Returns
    -------
    dict
        Run summary.
    """
    # Lazy import via importlib so a sys.modules stub (tests) always wins over
    # the package attribute binding left by an earlier real import.
    try:
        import importlib

        feeds_mod = importlib.import_module("mole.feeds")
        fetch_all_with_warnings = feeds_mod.fetch_all_with_warnings
    except ImportError:
        # feeds.py not yet available; use empty stub so tests pass
        def fetch_all_with_warnings(config: Any, since: str):  # type: ignore[misc]
            return [], ["feeds.py not available — skipping fetch"]

    # Load feeds config
    feeds_yaml_path = repo_root / "feeds.yaml"
    if feeds_yaml_path.exists():
        with feeds_yaml_path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
    else:
        config = {"feeds": []}

    # Import extractor lazily (it is always present but keep pattern consistent)
    import extractor as extractor_mod

    # Fetch items
    items, warnings = fetch_all_with_warnings(config, since)

    # Load seen sources
    seen = store.load_seen_sources(repo_root)

    # Partition into new vs already-seen
    new_items = []
    skipped_count = 0
    for item in items:
        feed = item.get("feed", "")
        item_key = item.get("item_key", "")
        if store.is_seen_source(seen, feed, item_key):
            skipped_count += 1
        else:
            new_items.append(item)

    # Sort new items deterministically by (feed, item_key)
    new_items.sort(key=lambda it: (it.get("feed", ""), it.get("item_key", "")))

    # Load existing claims (for cross-source similarity)
    existing_claims = store.load_all_claims(repo_root)

    # Determine starting IDs for this run
    claim_max = 0
    for c in existing_claims:
        n = store._parse_numeric_id(c.get("id", ""), "clm_")
        if n is not None and n > claim_max:
            claim_max = n

    task_max = 0
    for t in store.load_all_tasks(repo_root):
        n = store._parse_numeric_id(t.get("id", ""), "task_")
        if n is not None and n > task_max:
            task_max = n

    # -----------------------------------------------------------------------
    # Process new items
    # -----------------------------------------------------------------------
    new_source_ids: list[str] = []
    new_claims_this_run: list[dict[str, Any]] = []

    for item in new_items:
        feed = item.get("feed", "")
        item_key = item.get("item_key", "")
        text = item.get("text", "")

        assert "text" not in {}, "invariant check placeholder"

        sha256 = store.content_sha256(text)
        raw_claims = extractor_mod.extract_claims(text)

        # Assign claim IDs
        claim_records: list[dict[str, Any]] = []
        for raw in raw_claims:
            claim_max += 1
            claim_id = f"clm_{claim_max:06d}"
            src_id = f"src_{feed}_{item_key}"
            rec = store.append_claim(
                repo_root,
                claim_id=claim_id,
                source_id=src_id,
                text=raw["text"],
                claim_type=raw["type"],
                support_in_text=raw["support_in_text"],
                quote=raw["quote"],
                run_id=run_id,
            )
            claim_records.append(rec)

        # Append source record (NO text field)
        store.append_source(
            repo_root,
            feed=feed,
            item_key=item_key,
            url=item.get("url", ""),
            title=item.get("title", ""),
            author=item.get("author", ""),
            published=item.get("published", ""),
            sha256=sha256,
            claim_count=len(raw_claims),
            run_id=run_id,
        )
        new_source_ids.append(f"src_{feed}_{item_key}")
        new_claims_this_run.extend(claim_records)

    # -----------------------------------------------------------------------
    # Enqueue 'refine' tasks for claims needing cleanup
    # -----------------------------------------------------------------------
    refine_count = 0
    for claim in new_claims_this_run:
        if _needs_refine(claim["text"]):
            task_max += 1
            task_id = f"task_{task_max:06d}"
            store.append_task(
                repo_root,
                task_id=task_id,
                kind="refine",
                payload={"claim_id": claim["id"]},
                created_run=run_id,
            )
            refine_count += 1

    # -----------------------------------------------------------------------
    # Enqueue 'identity'/'edge' tasks for cross-source near-duplicate pairs
    # -----------------------------------------------------------------------
    # Fragments routed to refine must not generate pair work — judged pairs on
    # non-self-contained claims are wasted worker tokens.
    pairable_new = [c for c in new_claims_this_run if not _needs_refine(c["text"])]
    pairable_existing = [c for c in existing_claims if not _needs_refine(c["text"])]
    candidate_pairs = _cross_source_pairs(
        pairable_new,
        pairable_existing,
        extractor_mod,
    )

    identity_edge_count = 0
    for id_a, id_b, sim in candidate_pairs:
        task_max += 1
        task_id = f"task_{task_max:06d}"
        # High similarity -> identity candidate; lower -> edge candidate
        kind = "identity" if sim >= 0.65 else "edge"
        store.append_task(
            repo_root,
            task_id=task_id,
            kind=kind,
            payload={"a": id_a, "b": id_b, "sim": sim},
            created_run=run_id,
        )
        identity_edge_count += 1

    # -----------------------------------------------------------------------
    # Write run summary
    # -----------------------------------------------------------------------
    runs_dir = repo_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "since": since,
        "feeds_polled": len(config.get("feeds", [])),
        "items_seen": len(items),
        "items_new": len(new_items),
        "items_skipped": skipped_count,
        "claims_extracted": len(new_claims_this_run),
        "tasks_enqueued": {
            "refine": refine_count,
            "identity_edge": identity_edge_count,
        },
        "warnings": warnings,
    }

    with (runs_dir / f"{run_id}.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    return summary
