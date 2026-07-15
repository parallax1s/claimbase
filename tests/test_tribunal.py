"""Richtschwert: marshal determinism + full trial with a mock bench."""

from __future__ import annotations

import json
from pathlib import Path

from mole import store
from mole.tribunal import docket, marshal, try_question


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _fixture(tmp_path: Path) -> Path:
    _write_jsonl(tmp_path / "data" / "questions.jsonl", [
        {"id": "q_000001", "text": "Is X detectable before deployment?",
         "status": "contested", "keywords": ["x"]},
    ])
    _write_jsonl(tmp_path / "data" / "claims.jsonl", [
        {"id": "clm_000001", "source_id": "s", "text": "Study A detected X in 3 of 5 systems.",
         "type": "empirical", "support_in_text": 0.5, "quote": "", "run_id": "r", "status": "extracted"},
        {"id": "clm_000002", "source_id": "s", "text": "Detector D was evaded by adversarial X.",
         "type": "empirical", "support_in_text": 0.5, "quote": "", "run_id": "r", "status": "extracted"},
        {"id": "clm_000003", "source_id": "s", "text": "Retired claim.", "type": "descriptive",
         "support_in_text": 0.5, "quote": "", "run_id": "r", "status": "retired"},
    ])
    _write_jsonl(tmp_path / "data" / "theses.jsonl", [
        {"id": "th_000001", "source_id": "s", "text": "Detection lags generation structurally.",
         "claim_ids": [], "tier": "unreviewed_ai_draft", "run_id": "r", "status": "proposed"},
    ])
    _write_jsonl(tmp_path / "data" / "attachments.jsonl", [
        {"question_id": "q_000001", "claim_id": "clm_000001", "stance": "bears_yes", "strength": 0.8, "note": ""},
        {"question_id": "q_000001", "claim_id": "clm_000002", "stance": "bears_no", "strength": 0.7, "note": ""},
        {"question_id": "q_000001", "claim_id": "th_000001", "stance": "bears_no", "strength": 0.6, "note": ""},
        {"question_id": "q_000001", "claim_id": "clm_000003", "stance": "informs", "strength": 0.9, "note": ""},
        {"question_id": "q_000001", "claim_id": "clm_000001", "stance": "informs", "strength": 0.2, "note": ""},
    ])
    return tmp_path


def test_marshal_buckets_floor_and_retired(tmp_path):
    m = marshal(_fixture(tmp_path), "q_000001")
    assert [e["id"] for e in m["buckets"]["bears_yes"]] == ["clm_000001"]
    # thesis text resolves; retired claim dropped even at strength 0.9
    assert [e["id"] for e in m["buckets"]["bears_no"]] == ["clm_000002", "th_000001"]
    assert m["buckets"]["informs"] == []
    assert m["evidence_watermark"] == 5  # counts ALL attachments for the question


class MockProvider:
    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, prompt, model=None):
        return self.replies.pop(0)


def test_full_trial_with_mock_bench(tmp_path):
    repo = _fixture(tmp_path)
    brief = json.dumps({"arguments": [{"point": "p", "cites": ["clm_000001"]}],
                        "key_premises": ["k"]})
    cross = json.dumps({"attacks": [{"target_premise": "k", "attack": "a", "cites": []}]})
    ruling_a = json.dumps({"credence": [0.55, 0.8], "cruxes": ["c1"],
                           "would_change_my_mind": ["w1"], "rationale": "r"})
    ruling_b = json.dumps({"credence": [0.7, 0.4], "cruxes": [],  # reversed -> normalized
                           "would_change_my_mind": [], "rationale": "r2"})
    shared = MockProvider([brief, brief, cross, cross, ruling_a])
    judges = [
        {"judge": "mock-a", "kind": "claude", "model": "m", "provider": shared},
        {"judge": "mock-b", "kind": "claude", "model": "m", "provider": MockProvider([ruling_b])},
    ]
    v = try_question(repo, "q_000001", "test-run", providers=judges,
                     on_progress=lambda m: None)
    assert v["id"] == "vrd_000001"
    assert v["credence"] == [0.475, 0.75]  # true median (midpoint on even bench)
    assert v["rulings"][1]["credence"] == [0.4, 0.7]
    assert v["evidence_cited"] == ["clm_000001"]
    assert v["status"] == "standing"

    rows = store.load_all_verdicts(repo)
    assert len(rows) == 1

    d = docket(repo)
    assert d[0]["tribunal"] == "STANDING"

    # new evidence after the verdict flags a retrial
    with (repo / "data" / "attachments.jsonl").open("a") as fh:
        fh.write(json.dumps({"question_id": "q_000001", "claim_id": "clm_000002",
                             "stance": "informs", "strength": 0.5, "note": ""}) + "\n")
    d2 = docket(repo)
    assert d2[0]["tribunal"].startswith("RETRIAL")
