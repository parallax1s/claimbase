"""Tests for the worker-side fable integration (no engine, no network)."""

from __future__ import annotations

import json
from pathlib import Path

from mole import attach, store
from mole.fable import map_artifact, resolve_thesis_claims


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


ENGINE_CLAIM = {
    "id": "c1234567890",
    "text": "The reserve fund lost 12% of its value in 2019.",
    "kind": "assertion",
    "claim_type": "statistical",
    "stance": "reported",
    "stance_source": "Dr. Reyes",
    "hedge": [0.75, 0.95],
    "spans": [{"quote": "the reserve fund lost 12% of its value", "start": 10, "end": 49}],
    "flags": ["quote_normalized_match"],
    "tier": "validated",
    "prompt_version": "propose_v2",
    "engine_version": "0.1.0",
}


def test_map_artifact_claim_fields_and_ids():
    recs, theses = map_artifact(
        claims=[ENGINE_CLAIM], theses=[], source_id="src_f_k",
        run_id="fable-t", claim_start=41, thesis_start=0, drift=True,
    )
    assert len(recs) == 1 and theses == []
    r = recs[0]
    assert r["id"] == "clm_000042"
    assert r["source_id"] == "src_f_k"
    assert r["type"] == "statistical"
    assert r["status"] == "extracted"
    assert r["tier"] == "validated"
    assert r["stance"] == "reported" and r["stance_source"] == "Dr. Reyes"
    assert r["hedge"] == [0.75, 0.95]
    assert "content_drift" in r["flags"]
    assert r["quote"].startswith("the reserve fund")


def test_map_artifact_remaps_thesis_claim_ids():
    thesis = {"id": "thX", "text": "The fund faces solvency risk.",
              "claim_ids": ["c1234567890", "unknown"], "tier": "unreviewed_ai_draft",
              "flags": []}
    recs, theses = map_artifact(
        claims=[ENGINE_CLAIM], theses=[thesis], source_id="src_f_k",
        run_id="fable-t", claim_start=0, thesis_start=7, drift=False,
    )
    assert theses[0]["id"] == "th_000008"
    assert theses[0]["claim_ids"] == [recs[0]["id"]]
    assert theses[0]["status"] == "proposed"


def test_resolve_thesis_claims_flattens_points():
    theses = [{"id": "t", "text": "x", "claim_ids": ["cA"],
               "point_ids": ["p1", "p2"]}]
    points = [{"id": "p1", "claim_ids": ["cB", "cA"]},
              {"id": "p2", "claim_ids": ["cC"]}]
    out = resolve_thesis_claims(theses, points)
    assert out[0]["claim_ids"] == ["cA", "cB", "cC"]


def test_store_thesis_ids_and_batch(tmp_path):
    assert store.next_thesis_id(tmp_path) == "th_000001"
    store.append_theses_batch(tmp_path, [{"id": "th_000001", "text": "t"}])
    assert store.next_thesis_id(tmp_path) == "th_000002"
    assert store.load_all_theses(tmp_path)[0]["text"] == "t"


def test_update_tasks_flips_matching_only(tmp_path):
    _write_jsonl(tmp_path / "queue" / "pending.jsonl", [
        {"id": "task_000001", "kind": "refine", "status": "pending",
         "payload": {"claim_id": "clm_000001"}},
        {"id": "task_000002", "kind": "refine", "status": "pending",
         "payload": {"claim_id": "clm_000009"}},
        {"id": "task_000003", "kind": "attach", "status": "pending",
         "payload": {"claim_id": "clm_000001"}},
    ])
    n = store.update_tasks(
        tmp_path,
        lambda t: t["kind"] == "refine"
        and t["payload"]["claim_id"] == "clm_000001",
        status="obsolete", note="superseded",
    )
    assert n == 1
    rows = store.load_all_tasks(tmp_path)
    assert rows[0]["status"] == "obsolete" and rows[0]["note"] == "superseded"
    assert rows[1]["status"] == "pending"
    assert rows[2]["status"] == "pending"


def test_bump_source_extraction_keeps_sha(tmp_path):
    _write_jsonl(tmp_path / "data" / "sources.jsonl", [
        {"id": "src_f_k", "feed": "f", "content_sha256": "ORIGINAL",
         "claim_count": 6, "run_id": "r0"},
    ])
    store.bump_source_extraction(tmp_path, source_id="src_f_k",
                                 claim_count=11, fable_run="fable-t")
    rec = store.load_all_sources(tmp_path)[0]
    assert rec["content_sha256"] == "ORIGINAL"
    assert rec["claim_count"] == 11
    assert rec["fable_run"] == "fable-t"


def test_candidates_hybrid_lexical_fallback(monkeypatch):
    # Force the embedding path off so hybrid degrades to lexical-only.
    import extractor
    monkeypatch.setattr(extractor, "embed_texts", lambda texts: None)
    questions = [{"id": "q_1", "text": "Is RLHF sufficient for alignment?",
                  "keywords": ["rlhf", "reward hacking"]}]
    rows = [
        {"id": "th_000001",
         "text": "RLHF and reward hacking dominate current alignment failures."},
        {"id": "th_000002",
         "text": "Bread nourishes because of custom rather than insight."},
    ]
    index = attach.build_index([r["text"] for r in rows], questions)
    hits = attach.candidates_hybrid(rows, index, questions)
    ids = [h[0] for h in hits]
    assert "th_000001" in ids
    assert "th_000002" not in ids
