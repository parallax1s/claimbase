"""
Tests for mole pipeline, store, and compile.

Uses a tmp repo root with a feeds.yaml pointing at a FAKE feed kind.
Monkeypatches pipeline's feeds import and uses the real extractor.

Contract verified:
- source records contain no content text
- claim/task ids are stable and monotonic across two runs
- second run skips the seen item
- queue tasks are well-formed per SCHEMA
- compile produces the artifact with correct counts
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixture items (two items; item_b is a dup of item_a for the second run)
# ---------------------------------------------------------------------------

ITEM_A = {
    "feed": "fake-feed",
    "item_key": "post-001",
    "url": "https://example.com/post-001",
    "title": "The alignment problem is hard",
    "author": "Alice",
    "published": "2026-06-01",
    "text": (
        "The alignment problem presents fundamental challenges for AI safety. "
        "Researchers have demonstrated that current techniques are insufficient "
        "to guarantee aligned behavior at scale. "
        "This research suggests we need better interpretability tools."
    ),
}

ITEM_B = {
    "feed": "fake-feed",
    "item_key": "post-002",
    "url": "https://example.com/post-002",
    "title": "Scaling laws and safety",
    "author": "Bob",
    "published": "2026-06-02",
    "text": (
        "Scaling laws show that larger models are more capable. "
        "Safety properties do not automatically improve with scale. "
        "Evidence suggests that alignment research must accelerate significantly."
    ),
}

# ITEM_A_DUP has the same (feed, item_key) as ITEM_A — should be skipped in run 2
ITEM_A_DUP = dict(ITEM_A)  # same feed + item_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(tmp_path: Path) -> Path:
    """Set up a minimal repo structure with a fake feeds.yaml."""
    (tmp_path / "data").mkdir()
    (tmp_path / "queue").mkdir()
    (tmp_path / "runs").mkdir()

    feeds_cfg = {
        "feeds": [
            {
                "key": "fake-feed",
                "kind": "fake",
                "url": "https://fake.example.com/feed",
                "limit": 10,
            }
        ]
    }
    with (tmp_path / "feeds.yaml").open("w") as fh:
        yaml.dump(feeds_cfg, fh)

    return tmp_path


def _make_feeds_stub(items_to_return: list[dict]) -> types.ModuleType:
    """Return a fake mole.feeds module with fetch_all_with_warnings."""
    mod = types.ModuleType("mole.feeds")

    def fetch_all_with_warnings(config, since):
        return list(items_to_return), []

    mod.fetch_all_with_warnings = fetch_all_with_warnings
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStoreIds:
    """ID assignment is stable and monotonic."""

    def test_claim_ids_monotonic(self, tmp_path):
        from mole import store

        repo = _make_repo(tmp_path)
        # No existing claims -> first id is clm_000001
        id1 = store.next_claim_id(repo)
        assert id1 == "clm_000001"

        # Manually append a claim and re-query
        store.append_claim(
            repo,
            claim_id=id1,
            source_id="src_fake-feed_x",
            text="Some claim text here that is long enough.",
            claim_type="empirical",
            support_in_text=0.5,
            quote="Some claim text",
            run_id="run-01",
        )
        id2 = store.next_claim_id(repo)
        assert id2 == "clm_000002"

    def test_task_ids_monotonic(self, tmp_path):
        from mole import store

        repo = _make_repo(tmp_path)
        id1 = store.next_task_id(repo)
        assert id1 == "task_000001"

        store.append_task(
            repo,
            task_id=id1,
            kind="refine",
            payload={"claim_id": "clm_000001"},
            created_run="run-01",
        )
        id2 = store.next_task_id(repo)
        assert id2 == "task_000002"


class TestSourceNoContent:
    """Sources must never store text content."""

    def test_append_source_no_text_field(self, tmp_path):
        from mole import store

        repo = _make_repo(tmp_path)
        rec = store.append_source(
            repo,
            feed="fake-feed",
            item_key="post-001",
            url="https://example.com",
            title="Test",
            author="Author",
            published="2026-06-01",
            sha256="abc123",
            claim_count=3,
            run_id="run-01",
        )
        assert "text" not in rec

        # Also verify what was written to disk
        sources = store.load_all_sources(repo)
        assert len(sources) == 1
        assert "text" not in sources[0]

    def test_content_sha256(self):
        from mole import store

        sha = store.content_sha256("hello world")
        assert len(sha) == 64  # hex sha256
        assert store.content_sha256("hello world") == sha  # deterministic


class TestPipelineRun:
    """Full pipeline run behaviour."""

    def _run(self, repo, run_id, items, monkeypatch):
        """Run the pipeline with a stubbed feeds module."""
        stub = _make_feeds_stub(items)
        # Inject into sys.modules so the lazy import inside pipeline.run() finds it
        monkeypatch.setitem(sys.modules, "mole.feeds", stub)

        from mole import pipeline

        return pipeline.run(repo_root=repo, since="2026-06-01", run_id=run_id)

    def test_first_run_ingests_both_items(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        summary = self._run(repo, "run-01", [ITEM_A, ITEM_B], monkeypatch)

        assert summary["run_id"] == "run-01"
        assert summary["items_new"] == 2
        assert summary["items_skipped"] == 0
        assert summary["claims_extracted"] > 0

    def test_second_run_skips_seen_item(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        # Run 1: both items
        self._run(repo, "run-01", [ITEM_A, ITEM_B], monkeypatch)

        # Run 2: ITEM_A_DUP (same key as ITEM_A) + ITEM_B (same key)
        # Both should be skipped
        summary2 = self._run(repo, "run-02", [ITEM_A_DUP, ITEM_B], monkeypatch)
        assert summary2["items_skipped"] == 2
        assert summary2["items_new"] == 0
        assert summary2["claims_extracted"] == 0

    def test_second_run_partial_skip(self, tmp_path, monkeypatch):
        from mole import store

        repo = _make_repo(tmp_path)
        # Run 1: only ITEM_A
        self._run(repo, "run-01", [ITEM_A], monkeypatch)

        claims_after_run1 = store.load_all_claims(repo)
        n_claims_run1 = len(claims_after_run1)

        # Run 2: ITEM_A (dup) + ITEM_B (new)
        summary2 = self._run(repo, "run-02", [ITEM_A_DUP, ITEM_B], monkeypatch)
        assert summary2["items_skipped"] == 1
        assert summary2["items_new"] == 1

        claims_after_run2 = store.load_all_claims(repo)
        assert len(claims_after_run2) > n_claims_run1

    def test_source_records_have_no_text(self, tmp_path, monkeypatch):
        from mole import store

        repo = _make_repo(tmp_path)
        self._run(repo, "run-01", [ITEM_A], monkeypatch)

        for src in store.load_all_sources(repo):
            assert "text" not in src, f"source {src['id']} stores text content"

    def test_claim_ids_are_monotonic_across_runs(self, tmp_path, monkeypatch):
        from mole import store

        repo = _make_repo(tmp_path)
        self._run(repo, "run-01", [ITEM_A], monkeypatch)
        claims_after_run1 = store.load_all_claims(repo)
        ids_run1 = [c["id"] for c in claims_after_run1]

        self._run(repo, "run-02", [ITEM_B], monkeypatch)
        claims_after_run2 = store.load_all_claims(repo)
        # run2 accumulates all claims (run1 + run2 new ones)
        all_ids = [c["id"] for c in claims_after_run2]

        # All IDs unique across both runs
        assert len(set(all_ids)) == len(all_ids), "claim ids must be globally unique"

        # IDs are zero-padded monotonically increasing integers
        nums = [int(cid.split("_")[1]) for cid in all_ids]
        assert nums == sorted(nums), "claim ids must be monotonically increasing"

        # Run 2 added new claims beyond run 1's max
        run1_max = max(int(cid.split("_")[1]) for cid in ids_run1)
        run2_new = [c for c in claims_after_run2 if c["run_id"] == "run-02"]
        if run2_new:
            run2_min = min(int(c["id"].split("_")[1]) for c in run2_new)
            assert run2_min > run1_max, "run-02 ids must be strictly above run-01 ids"

    def test_run_summary_written_to_disk(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        self._run(repo, "run-42", [ITEM_A], monkeypatch)

        summary_path = repo / "runs" / "run-42.json"
        assert summary_path.exists()
        with summary_path.open() as fh:
            data = json.load(fh)
        assert data["run_id"] == "run-42"

    def test_queue_tasks_well_formed(self, tmp_path, monkeypatch):
        from mole import store

        repo = _make_repo(tmp_path)
        self._run(repo, "run-01", [ITEM_A], monkeypatch)

        tasks = store.load_all_tasks(repo)
        for task in tasks:
            assert "id" in task
            assert task["id"].startswith("task_")
            assert task["kind"] in {"refine", "extract", "identity", "edge", "verify"}
            assert "payload" in task
            assert "created_run" in task
            assert task["status"] == "pending"

            if task["kind"] == "extract":
                assert "source_id" in task["payload"]

            if task["kind"] == "refine":
                assert "claim_id" in task["payload"]
            elif task["kind"] in {"identity", "edge"}:
                assert "a" in task["payload"]
                assert "b" in task["payload"]
                assert "sim" in task["payload"]
                assert 0.0 <= task["payload"]["sim"] <= 1.0

    def test_task_ids_are_monotonic(self, tmp_path, monkeypatch):
        from mole import store

        repo = _make_repo(tmp_path)
        self._run(repo, "run-01", [ITEM_A, ITEM_B], monkeypatch)

        tasks = store.load_all_tasks(repo)
        nums = [int(t["id"].split("_")[1]) for t in tasks]
        assert nums == sorted(nums), "task ids must be monotonically increasing"
        assert len(set(nums)) == len(nums), "task ids must be unique"


class TestCompile:
    """compile() produces the artifact with correct structure and counts."""

    def _run_pipeline(self, repo, run_id, items, monkeypatch):
        stub = _make_feeds_stub(items)
        monkeypatch.setitem(sys.modules, "mole.feeds", stub)
        from mole import pipeline
        return pipeline.run(repo_root=repo, since="2026-06-01", run_id=run_id)

    def test_artifact_written(self, tmp_path, monkeypatch):
        from mole.compile import compile

        repo = _make_repo(tmp_path)
        self._run_pipeline(repo, "run-01", [ITEM_A, ITEM_B], monkeypatch)
        artifact = compile(repo)

        artifact_path = repo / "data" / "artifact.json"
        assert artifact_path.exists()

        with artifact_path.open() as fh:
            on_disk = json.load(fh)
        assert on_disk == artifact

    def test_artifact_counts_correct(self, tmp_path, monkeypatch):
        from mole import store
        from mole.compile import compile

        repo = _make_repo(tmp_path)
        self._run_pipeline(repo, "run-01", [ITEM_A, ITEM_B], monkeypatch)
        artifact = compile(repo)

        counts = artifact["counts"]

        # sources
        assert counts["sources"] == len(store.load_all_sources(repo))
        # active claims (non-retired)
        all_claims = store.load_all_claims(repo)
        active = [c for c in all_claims if c.get("status") != "retired"]
        assert counts["claims"] == len(active)
        # edges (none in a fresh run without workers)
        assert counts["edges"] == 0
        # pending_tasks breakdown
        tasks = store.load_all_tasks(repo)
        for task in tasks:
            kind = task["kind"]
            assert counts["pending_tasks"].get(kind, 0) > 0

    def test_artifact_claims_are_active_only(self, tmp_path, monkeypatch):
        from mole import store
        from mole.compile import compile

        repo = _make_repo(tmp_path)
        self._run_pipeline(repo, "run-01", [ITEM_A], monkeypatch)

        # Manually retire a claim by rewriting claims.jsonl
        claims = store.load_all_claims(repo)
        if not claims:
            pytest.skip("no claims extracted — nothing to retire")
        claim_path = repo / "data" / "claims.jsonl"
        with claim_path.open("w") as fh:
            for i, c in enumerate(claims):
                if i == 0:
                    c = dict(c)
                    c["status"] = "retired"
                fh.write(json.dumps(c) + "\n")

        artifact = compile(repo)
        artifact_ids = {c["id"] for c in artifact["claims"]}
        assert claims[0]["id"] not in artifact_ids
        assert artifact["counts"]["claims"] == len(claims) - 1

    def test_artifact_deterministic(self, tmp_path, monkeypatch):
        from mole.compile import compile

        repo = _make_repo(tmp_path)
        self._run_pipeline(repo, "run-01", [ITEM_A, ITEM_B], monkeypatch)

        art1 = compile(repo)
        art2 = compile(repo)
        assert art1 == art2


class TestRegressionFixes:
    """Regressions for dedup, change detection, provenance, and edge filtering."""

    def _run(self, repo, run_id, items, monkeypatch):
        stub = _make_feeds_stub(items)
        monkeypatch.setitem(sys.modules, "mole.feeds", stub)
        from mole import pipeline
        return pipeline.run(repo_root=repo, since="2026-06-01", run_id=run_id)

    def test_within_fetch_dedup(self, tmp_path, monkeypatch):
        from mole import store

        repo = _make_repo(tmp_path)
        summary = self._run(repo, "run-01", [ITEM_A, ITEM_A_DUP], monkeypatch)
        assert summary["items_new"] == 1
        assert summary["items_skipped"] == 1
        sources = store.load_all_sources(repo)
        assert len(sources) == 1
        ids = [s["id"] for s in sources]
        assert len(set(ids)) == len(ids)

    def test_content_change_retires_and_reextracts(self, tmp_path, monkeypatch):
        from mole import store

        repo = _make_repo(tmp_path)
        self._run(repo, "run-01", [ITEM_A], monkeypatch)

        changed = dict(ITEM_A)
        changed["text"] = (
            "The alignment problem was overstated in early work. "
            "New results demonstrate that current techniques scale better than expected. "
            "This evidence suggests earlier claims should be revised downward."
        )
        summary2 = self._run(repo, "run-02", [changed], monkeypatch)
        assert summary2["items_changed"] == 1
        assert summary2["items_new"] == 0
        assert summary2["claims_retired"] > 0

        claims = store.load_all_claims(repo)
        old = [c for c in claims if c["run_id"] == "run-01"]
        new = [c for c in claims if c["run_id"] == "run-02"]
        assert old and all(c["status"] == "retired" for c in old)
        assert new and all(c["status"] == "extracted" for c in new)

        # Source record updated in place: one record, new hash, stable id
        sources = store.load_all_sources(repo)
        assert len(sources) == 1
        assert sources[0]["content_sha256"] == store.content_sha256(changed["text"])
        assert sources[0]["run_id"] == "run-02"

    def test_orphan_claims_pruned(self, tmp_path, monkeypatch):
        from mole import store

        repo = _make_repo(tmp_path)
        self._run(repo, "run-01", [ITEM_A], monkeypatch)
        # Simulate an interrupted run: claim written, source record missing
        store.append_claim(
            repo,
            claim_id="clm_009999",
            source_id="src_fake-feed_never-persisted",
            text="Orphan claim from an interrupted run with enough words.",
            claim_type="empirical",
            support_in_text=0.5,
            quote="Orphan claim",
            run_id="run-crashed",
        )
        summary = self._run(repo, "run-02", [], monkeypatch)
        assert summary["orphan_claims_pruned"] == 1
        assert all(
            c["source_id"] != "src_fake-feed_never-persisted"
            for c in store.load_all_claims(repo)
        )

    def test_compile_drops_edges_to_retired_claims(self, tmp_path, monkeypatch):
        from mole import store
        from mole.compile import compile

        repo = _make_repo(tmp_path)
        self._run(repo, "run-01", [ITEM_A, ITEM_B], monkeypatch)
        claims = store.load_all_claims(repo)
        assert len(claims) >= 2

        # Add an edge, then retire one endpoint
        with (repo / "data" / "edges.jsonl").open("w") as fh:
            fh.write(json.dumps({
                "a": claims[0]["id"], "b": claims[1]["id"],
                "relation": "supports", "confidence": 0.8,
            }) + "\n")
        store.retire_claims_for_source(repo, claims[0]["source_id"])

        artifact = compile(repo)
        artifact_ids = {c["id"] for c in artifact["claims"]}
        for e in artifact["edges"]:
            assert e["a"] in artifact_ids and e["b"] in artifact_ids
        assert artifact["counts"]["edges"] == len(artifact["edges"]) == 0

    def test_generated_run_prefers_caller_then_append_order(self, tmp_path, monkeypatch):
        from mole.compile import compile

        repo = _make_repo(tmp_path)
        stub_sources = [
            ("manual-20260611", "post-m"),
            ("28498934148", "post-n"),
        ]
        from mole import store
        for run, key in stub_sources:
            store.append_source(
                repo, feed="fake-feed", item_key=key, url="https://x",
                title="t", author="a", published="2026-06-01",
                sha256="s", claim_count=0, run_id=run,
            )
        # Append-order fallback: last record wins, not lexicographic max
        assert compile(repo)["generated_run"] == "28498934148"
        # Explicit run id wins over everything
        assert compile(repo, run_id="run-explicit")["generated_run"] == "run-explicit"


class TestFeedsDeterminism:
    """No wall-clock values fabricated for missing/bad dates."""

    def test_parse_datetime_returns_none(self):
        from mole import feeds

        assert feeds._parse_datetime("") is None
        assert feeds._parse_datetime("not-a-date-at-all") is None

    def test_coerce_item_keeps_unknown_date_with_empty_published(self):
        from datetime import datetime, timezone
        from mole import feeds

        since = datetime(2026, 6, 1, tzinfo=timezone.utc)
        item = feeds._coerce_item(
            feed="f", item_key="k", title="t", author="a",
            published=None, since_dt=since, url="u",
            raw_text="<p>Some body text</p>",
        )
        assert item is not None
        assert item["published"] == ""

    def test_invalid_since_raises(self):
        from mole import feeds

        with pytest.raises(ValueError):
            feeds.fetch_all_with_warnings({"feeds": []}, "garbage")
