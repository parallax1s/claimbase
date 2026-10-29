import json
from pathlib import Path

from mole import store
from mole.sweep import find_junk, sweep_junk


def _seed(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "queue").mkdir()
    rows = [
        {"id": "clm_000001", "source_id": "src_a", "text": "mjx-container[jax=\"CHTML\"] {\nline-height: 0;\n}",
         "type": "other", "support_in_text": 0.5, "quote": "", "run_id": "r", "status": "extracted"},
        {"id": "clm_000002", "source_id": "src_a", "text": "Interest rates fell by two points in 2024.",
         "type": "statistical", "support_in_text": 0.5, "quote": "", "run_id": "r", "status": "extracted"},
        {"id": "clm_000003", "source_id": "src_b", "text": ".foo { margin: 0; }",
         "type": "other", "support_in_text": 0.5, "quote": "", "run_id": "r", "status": "retired"},
        {"id": "clm_000004", "source_id": "src_b", "text": "A fine claim.",
         "type": "other", "support_in_text": 0.5, "quote": "var x = document.getElementById('a')",
         "run_id": "r", "status": "refined"},
    ]
    with open(tmp_path / "data" / "claims.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tasks = [
        {"id": "task_000001", "kind": "refine", "payload": {"claim_id": "clm_000001"},
         "created_run": "r", "status": "pending"},
        {"id": "task_000002", "kind": "identity", "payload": {"a": "clm_000002", "b": "clm_000004", "sim": 0.6},
         "created_run": "r", "status": "pending"},
        {"id": "task_000003", "kind": "refine", "payload": {"claim_id": "clm_000001"},
         "created_run": "r", "status": "done"},
    ]
    with open(tmp_path / "queue" / "pending.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    return tmp_path


def test_find_junk_skips_retired_and_checks_quote(tmp_path):
    root = _seed(tmp_path)
    ids = sorted(c["id"] for c in find_junk(store.load_all_claims(root)))
    assert ids == ["clm_000001", "clm_000004"]


def test_dry_run_writes_nothing(tmp_path):
    root = _seed(tmp_path)
    before = (root / "data" / "claims.jsonl").read_text()
    s = sweep_junk(root, run_id="sweep-test")
    assert s["junk_found"] == 2 and s["pending_tasks_referencing"] == 2 and not s["applied"]
    assert (root / "data" / "claims.jsonl").read_text() == before


def test_apply_retires_in_place_and_obsoletes_pending_tasks(tmp_path):
    root = _seed(tmp_path)
    s = sweep_junk(root, run_id="sweep-test", apply=True)
    assert s["junk_found"] == 2 and s["tasks_obsoleted"] == 2
    claims = {c["id"]: c for c in store.load_all_claims(root)}
    assert len(claims) == 4                       # history kept
    assert claims["clm_000001"]["status"] == "retired"
    assert claims["clm_000001"]["retired_reason"] == "junk_shape"
    assert claims["clm_000001"]["retired_by"] == "sweep-test"
    assert claims["clm_000004"]["status"] == "retired"
    assert claims["clm_000002"]["status"] == "extracted"
    assert "retired_by" not in claims["clm_000003"]  # already retired, untouched
    tasks = {t["id"]: t for t in store.load_all_tasks(root)}
    assert tasks["task_000001"]["status"] == "obsolete"
    assert tasks["task_000002"]["status"] == "obsolete"   # identity pair touching junk
    assert tasks["task_000003"]["status"] == "done"
    # idempotent
    assert sweep_junk(root, run_id="sweep-test-2", apply=True)["junk_found"] == 0
