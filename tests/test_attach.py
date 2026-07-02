"""
Tests for the attach-candidate stage (mole/attach.py + pipeline wiring).

Contract verified:
- scoring is deterministic and keyword-phrase driven
- claims matching a question's keywords produce attach tasks; unrelated
  claims and repos without a question layer produce none
- existing (question, claim) pairs are never re-enqueued
- attach tasks are well-formed per SCHEMA and counted in the run summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tests.test_pipeline import ITEM_A, _make_feeds_stub, _make_repo

QUESTION = {
    "id": "q_000001",
    "slug": "alignment-techniques-scale",
    "text": "Will current alignment techniques keep working as capabilities scale?",
    "domain": "ai-alignment",
    "type": "forecast",
    "tags": [],
    "why_it_matters": "test",
    "keywords": ["alignment problem", "interpretability", "aligned behavior"],
    "status": "open",
    "posture": "test",
    "triage": {},
    "refines_question": None,
    "merged_into": None,
    "created_run": "run-00",
    "provenance": {"extraction_method": "manual"},
}


def _write_questions(repo: Path, questions: list[dict]) -> None:
    path = repo / "data" / "questions.jsonl"
    path.write_text("".join(json.dumps(q) + "\n" for q in questions))


def _run(repo, run_id, items, monkeypatch):
    stub = _make_feeds_stub(items)
    monkeypatch.setitem(sys.modules, "mole.feeds", stub)
    from mole import pipeline

    return pipeline.run(repo_root=repo, since="2026-06-01", run_id=run_id)


class TestAttachScoring:
    def test_keyword_phrase_match_scores_above_floor(self):
        from mole import attach

        texts = [
            "The alignment problem presents fundamental challenges.",
            "Unrelated text about cooking pasta with tomatoes.",
        ]
        index = attach.build_index(texts, [QUESTION])
        hit = attach.best_question(texts[0], index)
        assert hit is not None
        qid, score = hit
        assert qid == "q_000001"
        assert score >= attach.ATTACH_FLOOR

    def test_unrelated_claim_scores_below_floor(self):
        from mole import attach

        texts = [
            "The alignment problem presents fundamental challenges.",
            "Unrelated text about cooking pasta with tomatoes.",
        ]
        index = attach.build_index(texts, [QUESTION])
        hit = attach.best_question(texts[1], index)
        assert hit is None or hit[1] < attach.ATTACH_FLOOR

    def test_deterministic(self):
        from mole import attach

        texts = ["The alignment problem presents fundamental challenges."]
        a = attach.best_question(texts[0], attach.build_index(texts, [QUESTION]))
        b = attach.best_question(texts[0], attach.build_index(texts, [QUESTION]))
        assert a == b


class TestPipelineAttach:
    def test_attach_tasks_enqueued_for_matching_claims(self, tmp_path, monkeypatch):
        from mole import store

        repo = _make_repo(tmp_path)
        _write_questions(repo, [QUESTION])
        summary = _run(repo, "run-01", [ITEM_A], monkeypatch)

        attach_tasks = [
            t for t in store.load_all_tasks(repo) if t["kind"] == "attach"
        ]
        assert summary["tasks_enqueued"]["attach"] == len(attach_tasks)
        assert attach_tasks, "ITEM_A mentions the question keywords; expected tasks"
        for t in attach_tasks:
            assert t["status"] == "pending"
            assert t["payload"]["question_id"] == "q_000001"
            assert t["payload"]["claim_id"].startswith("clm_")
            assert 0 < t["payload"]["sim"]

    def test_no_question_layer_no_attach_tasks(self, tmp_path, monkeypatch):
        from mole import store

        repo = _make_repo(tmp_path)
        summary = _run(repo, "run-01", [ITEM_A], monkeypatch)
        assert summary["tasks_enqueued"]["attach"] == 0
        assert not [t for t in store.load_all_tasks(repo) if t["kind"] == "attach"]

    def test_existing_pairs_not_reenqueued(self, tmp_path, monkeypatch):
        from mole import store

        repo = _make_repo(tmp_path)
        _write_questions(repo, [QUESTION])
        _run(repo, "run-01", [ITEM_A], monkeypatch)
        n_after_run1 = len(
            [t for t in store.load_all_tasks(repo) if t["kind"] == "attach"]
        )

        # Re-running with the same (already seen) item extracts nothing new,
        # and the attach stage must not duplicate pairs for existing claims.
        summary2 = _run(repo, "run-02", [dict(ITEM_A)], monkeypatch)
        assert summary2["tasks_enqueued"]["attach"] == 0
        n_after_run2 = len(
            [t for t in store.load_all_tasks(repo) if t["kind"] == "attach"]
        )
        assert n_after_run2 == n_after_run1

    def test_merged_questions_excluded(self, tmp_path, monkeypatch):
        from mole import store

        repo = _make_repo(tmp_path)
        merged = dict(QUESTION, merged_into="q_000009")
        _write_questions(repo, [merged])
        summary = _run(repo, "run-01", [ITEM_A], monkeypatch)
        assert summary["tasks_enqueued"]["attach"] == 0
