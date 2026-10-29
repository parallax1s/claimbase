import json
from pathlib import Path

from mole import store
from mole.attach_drain import drain, plan, render_candidates


class FakeProvider:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt, model=None):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def _w(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _seed(root: Path):
    _w(root / "data" / "questions.jsonl", [
        {"id": "q_000001", "slug": "q1", "text": "Is X true?", "why_it_matters": "because",
         "status": "active", "merged_into": None}])
    _w(root / "data" / "sources.jsonl", [
        {"id": "src_a", "title": "Paper A", "published": "2026-09-01T00:00:00+00:00"}])
    _w(root / "data" / "claims.jsonl", [
        {"id": "clm_000001", "source_id": "src_a", "text": "X holds in setting S.", "quote": "X holds in setting S",
         "type": "empirical", "support_in_text": 0.5, "run_id": "r", "status": "extracted", "tier": "validated"},
        {"id": "clm_000002", "source_id": "src_a", "text": "Y is unrelated.", "quote": "",
         "type": "empirical", "support_in_text": 0.5, "run_id": "r", "status": "extracted", "tier": "validated"},
        {"id": "clm_000003", "source_id": "src_a", "text": "legacy regex row", "quote": "",
         "type": "other", "support_in_text": 0.5, "run_id": "r", "status": "extracted"},
        {"id": "clm_000004", "source_id": "src_a", "text": "retired row", "quote": "",
         "type": "other", "support_in_text": 0.5, "run_id": "r", "status": "retired", "tier": "validated"},
    ])
    _w(root / "data" / "theses.jsonl", [])
    _w(root / "data" / "attachments.jsonl", [
        {"question_id": "q_000001", "claim_id": "clm_000001", "stance": "informs", "strength": 0.45,
         "note": "old", "judge": "old-judge", "verified": False, "run_id": "seed"}])
    _w(root / "queue" / "pending.jsonl", [
        {"id": "task_000001", "kind": "attach", "payload": {"question_id": "q_000001", "claim_id": "clm_000001", "sim": 0.6},
         "created_run": "fable-1", "status": "pending"},
        {"id": "task_000002", "kind": "attach", "payload": {"question_id": "q_000001", "claim_id": "clm_000002", "sim": 0.5},
         "created_run": "fable-1", "status": "pending"},
        {"id": "task_000003", "kind": "attach", "payload": {"question_id": "q_000001", "claim_id": "clm_000003", "sim": 0.5},
         "created_run": "m", "status": "pending"},
        {"id": "task_000004", "kind": "attach", "payload": {"question_id": "q_000001", "claim_id": "clm_000004", "sim": 0.5},
         "created_run": "m", "status": "pending"},
        {"id": "task_000005", "kind": "refine", "payload": {"claim_id": "clm_000003"}, "created_run": "m", "status": "pending"},
    ])
    return root


def test_plan_scopes_to_validated_live_claims(tmp_path):
    p = plan(_seed(tmp_path))
    assert p["counts"] == {"validated": 2, "retired": 1, "untiered": 1, "thesis": 0, "missing": 0, "no_question": 0}
    assert len(p["batches"]) == 1 and [i["claim"]["id"] for i in p["batches"][0]["items"]] == ["clm_000001", "clm_000002"]
    assert p["retired_tasks"] == ["task_000004"]
    assert "[1] X holds in setting S." in render_candidates(p["batches"][0]["items"])


def test_drain_writes_attachments_flips_tasks_and_replaces_rejudgment(tmp_path):
    root = _seed(tmp_path)
    provider = FakeProvider(['[{"n": 1, "stance": "bears_yes", "strength": 0.72, "note": "direct evidence"},'
                             ' {"n": 2, "stance": "none"}]'])
    s = drain(root, run_id="attach-test", model="fake-model", provider=provider, workers=1, log=lambda *_: None)
    assert s["judged"] == 2 and s["attached"] == 1 and s["by_stance"] == {"bears_yes": 1}
    assert s["retired_tasks_obsoleted"] == 1 and s["errors"] == []
    assert "Is X true?" in provider.prompts[0] and "legacy regex row" not in provider.prompts[0]
    att = list(store._iter_jsonl(root / "data" / "attachments.jsonl"))
    assert len(att) == 1  # re-judgment replaced the seed row (unique on question, claim)
    assert att[0]["judge"] == "fake-model" and att[0]["strength"] == 0.72 and att[0]["run_id"] == "fable-1"
    tasks = {t["id"]: t["status"] for t in store.load_all_tasks(root)}
    assert tasks == {"task_000001": "done", "task_000002": "done", "task_000003": "pending",
                     "task_000004": "obsolete", "task_000005": "pending"}


def test_drain_leaves_batch_pending_on_unparseable_reply(tmp_path):
    root = _seed(tmp_path)
    provider = FakeProvider(["nonsense", "still nonsense"])
    s = drain(root, run_id="attach-test", model="fake-model", provider=provider, workers=1, log=lambda *_: None)
    assert s["judged"] == 0 and len(s["errors"]) == 1
    assert all(t["status"] == "pending" for t in store.load_all_tasks(root) if t["id"] in ("task_000001", "task_000002"))


def test_dry_run_touches_nothing(tmp_path):
    root = _seed(tmp_path)
    before = (root / "queue" / "pending.jsonl").read_text()
    s = drain(root, run_id="attach-test", dry_run=True)
    assert s["batches_planned"] == 1 and s["counts"]["validated"] == 2
    assert (root / "queue" / "pending.jsonl").read_text() == before
