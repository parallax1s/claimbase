"""
Tests for the fog-of-war atlas (mole/atlas.py).

Uses tmp repo roots populated through the store helpers, like
tests/test_pipeline.py.

Contract verified:
- data/atlas.json matches the atlas contract shape (v2: unbounded
  coordinates plus a "bounds" box that covers all content)
- placement v2: anchor 0 at the origin, no two anchors within 150 units,
  kindred anchors orbit their parent, unrelated ones take the global spiral
- two builds on the same data are byte-identical (determinism)
- new data never moves existing districts or claims (stability);
  data/atlas_districts.jsonl is append-only
- embeddings-unavailable fallback groups districts by source feed
- ATLAS.md carries the header, district rows, sharpest fault, and the legend
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import extractor
from mole import store
from mole.atlas import build_atlas, write_atlas

# ---------------------------------------------------------------------------
# Fixture texts: two semantically distant topics across two feeds
# ---------------------------------------------------------------------------

ALIGNMENT_TEXTS = [
    "The alignment problem presents fundamental challenges for AI safety research.",
    "Current interpretability techniques are insufficient to audit frontier models.",
    "Scaling laws show that larger models are more capable but not safer by default.",
    "Alignment research funding must accelerate to keep pace with capability gains.",
]

HOUSING_TEXTS = [
    "Rising interest rates reduced housing affordability across European capitals.",
    "Rent control policies decreased the supply of rental apartments in several cities.",
    "Construction permit reform increased the rate of new housing completions.",
]

NOVEL_TEXT = "Medieval grain prices in Flanders rose sharply after the famine of 1315."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(tmp_path: Path) -> Path:
    """Set up a minimal repo structure (data/, queue/, runs/)."""
    (tmp_path / "data").mkdir()
    (tmp_path / "queue").mkdir()
    (tmp_path / "runs").mkdir()
    return tmp_path


def _append_edge(
    repo: Path,
    a: str,
    b: str,
    relation: str,
    confidence: float,
    note: str,
    verified: bool | None = None,
) -> None:
    record = {
        "a": a,
        "b": b,
        "relation": relation,
        "confidence": confidence,
        "note": note,
        "judge": "test-judge",
        "run_id": "run-01",
    }
    if verified is not None:
        record["verified"] = verified
        record["verifier"] = "test-verifier"
    with (repo / "data" / "edges.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _seed_repo(tmp_path: Path) -> Path:
    """Two sources on different feeds, seven claims, one tension, one supports."""
    repo = _make_repo(tmp_path)

    store.append_source(
        repo,
        feed="fake-feed",
        item_key="post-001",
        url="https://example.com/post-001",
        title="The alignment problem is hard",
        author="Alice",
        published="2026-06-01",
        sha256="a" * 64,
        claim_count=len(ALIGNMENT_TEXTS),
        run_id="run-01",
    )
    store.append_source(
        repo,
        feed="econ-feed",
        item_key="post-002",
        url="https://example.com/post-002",
        title="Housing markets under pressure",
        author="Bob",
        published="2026-06-02",
        sha256="b" * 64,
        claim_count=len(HOUSING_TEXTS),
        run_id="run-01",
    )

    n = 0
    for source_id, texts in (
        ("src_fake-feed_post-001", ALIGNMENT_TEXTS),
        ("src_econ-feed_post-002", HOUSING_TEXTS),
    ):
        for text in texts:
            n += 1
            store.append_claim(
                repo,
                claim_id=f"clm_{n:06d}",
                source_id=source_id,
                text=text,
                claim_type="empirical",
                support_in_text=0.5,
                quote=text[:100],
                run_id="run-01",
            )

    # One verified tension and one supports edge light up four claims.
    _append_edge(
        repo,
        "clm_000003",
        "clm_000006",
        "contradicts",
        0.9,
        "Synthetic test tension between scaling and supply claims.",
        verified=True,
    )
    _append_edge(repo, "clm_000001", "clm_000002", "supports", 0.7, "Synthetic supports edge.")
    return repo


def _add_claim(repo: Path, text: str) -> str:
    """Append a fresh source+claim pair; returns the new claim id."""
    claim_id = store.next_claim_id(repo)
    item_key = f"post-{claim_id}"
    store.append_source(
        repo,
        feed="fake-feed",
        item_key=item_key,
        url=f"https://example.com/{item_key}",
        title="A later arrival",
        author="Carol",
        published="2026-06-03",
        sha256="c" * 64,
        claim_count=1,
        run_id="run-02",
    )
    store.append_claim(
        repo,
        claim_id=claim_id,
        source_id=f"src_fake-feed_{item_key}",
        text=text,
        claim_type="empirical",
        support_in_text=0.5,
        quote=text[:100],
        run_id="run-02",
    )
    return claim_id


def _read(path: Path) -> bytes:
    return path.read_bytes()


def _read_anchors(repo: Path) -> list[dict]:
    path = repo / "data" / "atlas_districts.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------
# Contract shape
# ---------------------------------------------------------------------------

class TestContractShape:
    def test_top_level_keys(self, tmp_path):
        repo = _seed_repo(tmp_path)
        atlas = build_atlas(repo)

        assert set(atlas) == {
            "generated_run", "extent", "bounds", "districts", "claims", "tensions",
        }
        assert atlas["generated_run"] == "run-01"
        # "extent" is legacy/back-compat only; "bounds" is authoritative.
        assert atlas["extent"] == {"w": 1000, "h": 1000}
        bounds = atlas["bounds"]
        assert set(bounds) == {"x0", "y0", "x1", "y1"}
        assert bounds["x0"] < bounds["x1"]
        assert bounds["y0"] < bounds["y1"]

    def test_district_entries(self, tmp_path):
        repo = _seed_repo(tmp_path)
        atlas = build_atlas(repo)

        assert atlas["districts"], "expected at least one district"
        for d in atlas["districts"]:
            assert set(d) == {
                "id", "label", "x", "y", "r", "claims", "lit", "dim", "tensions",
            }
            assert d["id"].startswith("d_")
            assert 0 < d["r"] <= 170
            assert d["claims"] == d["lit"] + d["dim"]

    def test_claim_entries(self, tmp_path):
        repo = _seed_repo(tmp_path)
        atlas = build_atlas(repo)

        assert len(atlas["claims"]) == 7
        for c in atlas["claims"]:
            assert set(c) == {"id", "x", "y", "s", "d", "type", "text", "src"}
            assert c["s"] in {"lit", "dim"}
            # Labeled districts cover only multi-claim anchors; singleton
            # claims keep coordinates but reference an unlisted district.
            assert c["d"].startswith("d_")
            assert len(c["text"]) <= 140
            assert len(c["src"]) <= 40

    def test_bounds_cover_all_content(self, tmp_path):
        repo = _seed_repo(tmp_path)
        _add_claim(repo, NOVEL_TEXT)  # encourage an extra, far-away anchor
        atlas = build_atlas(repo)
        b = atlas["bounds"]

        for c in atlas["claims"]:
            assert b["x0"] <= c["x"] <= b["x1"]
            assert b["y0"] <= c["y"] <= b["y1"]
        for d in atlas["districts"]:
            assert b["x0"] <= d["x"] - d["r"] and d["x"] + d["r"] <= b["x1"]
            assert b["y0"] <= d["y"] - d["r"] and d["y"] + d["r"] <= b["y1"]

    def test_lit_means_on_a_judged_edge(self, tmp_path):
        repo = _seed_repo(tmp_path)
        atlas = build_atlas(repo)
        state = {c["id"]: c["s"] for c in atlas["claims"]}

        # Edge participants are lit (any relation counts as judged).
        for lit_id in ("clm_000001", "clm_000002", "clm_000003", "clm_000006"):
            assert state[lit_id] == "lit"
        for dim_id in ("clm_000004", "clm_000005", "clm_000007"):
            assert state[dim_id] == "dim"

    def test_tensions_are_contradicts_edges(self, tmp_path):
        repo = _seed_repo(tmp_path)
        atlas = build_atlas(repo)

        assert atlas["tensions"] == [{
            "a": "clm_000003",
            "b": "clm_000006",
            "note": "Synthetic test tension between scaling and supply claims.",
            "verified": True,
        }]
        # Each endpoint's district counts the tension.
        by_claim = {c["id"]: c["d"] for c in atlas["claims"]}
        touched = {by_claim["clm_000003"], by_claim["clm_000006"]}
        for d in atlas["districts"]:
            expected = 1 if d["id"] in touched else 0
            assert d["tensions"] == expected

    def test_retired_claims_excluded(self, tmp_path):
        repo = _seed_repo(tmp_path)
        store.append_claim(
            repo,
            claim_id=store.next_claim_id(repo),
            source_id="src_fake-feed_post-001",
            text="A retired claim that must not appear in the atlas.",
            claim_type="empirical",
            support_in_text=0.5,
            quote="A retired claim",
            run_id="run-01",
            status="retired",
        )
        atlas = build_atlas(repo)
        texts = [c["text"] for c in atlas["claims"]]
        assert "A retired claim that must not appear in the atlas." not in texts
        assert len(atlas["claims"]) == 7


# ---------------------------------------------------------------------------
# Placement v2: origin start, minimum separation, orbits and the spiral
# ---------------------------------------------------------------------------

class TestPlacement:
    def test_first_anchor_at_origin(self, tmp_path):
        repo = _seed_repo(tmp_path)
        write_atlas(repo)
        anchors = _read_anchors(repo)
        assert (anchors[0]["x"], anchors[0]["y"]) == (0.0, 0.0)

    def test_no_two_anchors_within_min_distance(self, tmp_path):
        repo = _seed_repo(tmp_path)
        _add_claim(repo, NOVEL_TEXT)  # encourage an extra anchor
        write_atlas(repo)
        anchors = _read_anchors(repo)

        assert len(anchors) >= 2, "expected several anchors"
        for i, a in enumerate(anchors):
            for b in anchors[i + 1:]:
                dist = math.dist((a["x"], a["y"]), (b["x"], b["y"]))
                assert dist >= 150.0 - 1e-6, f"{a['id']} and {b['id']} too close"

    def test_kindred_anchor_orbits_and_stranger_takes_spiral(self, tmp_path, monkeypatch):
        """Synthetic vectors steer the three placement paths directly.

        claim 1 founds anchor d_001 at the origin; claim 2 (cosine 0.25,
        between the orbit floor 0.18 and the join threshold 0.32) founds
        d_002 on d_001's first orbit ring (distance 170); claim 3
        (orthogonal, cosine 0) founds d_003 on the global spiral
        (distance 170*sqrt(g) from the origin for some slot g >= 1).
        """
        vecs = {
            "Founders write the first claim.": [1.0, 0.0, 0.0],
            "A cousin of the first claim.": [0.25, 0.9682458365518543, 0.0],
            "Something else entirely.": [0.0, 0.0, 1.0],
        }
        monkeypatch.setattr(extractor, "embed_texts", lambda texts: [vecs[t] for t in texts])

        repo = _make_repo(tmp_path)
        store.append_source(
            repo,
            feed="fake-feed",
            item_key="post-001",
            url="https://example.com/post-001",
            title="Three strangers",
            author="Alice",
            published="2026-06-01",
            sha256="a" * 64,
            claim_count=3,
            run_id="run-01",
        )
        for n, text in enumerate(vecs, start=1):
            store.append_claim(
                repo,
                claim_id=f"clm_{n:06d}",
                source_id="src_fake-feed_post-001",
                text=text,
                claim_type="empirical",
                support_in_text=0.5,
                quote=text,
                run_id="run-01",
            )
        write_atlas(repo)
        anchors = {a["id"]: a for a in _read_anchors(repo)}
        assert set(anchors) == {"d_001", "d_002", "d_003"}

        a1, a2, a3 = anchors["d_001"], anchors["d_002"], anchors["d_003"]
        assert (a1["x"], a1["y"]) == (0.0, 0.0)

        # d_002 orbits d_001 on the first ring: exactly 170 from its parent.
        orbit_dist = math.dist((a1["x"], a1["y"]), (a2["x"], a2["y"]))
        assert orbit_dist == pytest.approx(170.0, abs=0.05)

        # d_003 sits on a free global-spiral slot: 170*sqrt(g) from the
        # origin for some integer g >= 1, and clear of both other anchors.
        spiral_r = math.dist((0.0, 0.0), (a3["x"], a3["y"]))
        candidates = [170.0 * math.sqrt(g) for g in range(1, 12)]
        assert min(abs(spiral_r - r) for r in candidates) < 0.05
        assert math.dist((a1["x"], a1["y"]), (a3["x"], a3["y"])) >= 150.0 - 1e-6
        assert math.dist((a2["x"], a2["y"]), (a3["x"], a3["y"])) >= 150.0 - 1e-6


# ---------------------------------------------------------------------------
# Determinism: rebuilds on the same data are byte-identical
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_rebuild_byte_identical(self, tmp_path):
        repo = _seed_repo(tmp_path)

        write_atlas(repo)
        first = {
            name: _read(repo / name)
            for name in ("data/atlas.json", "data/atlas_districts.jsonl", "ATLAS.md")
        }

        write_atlas(repo)
        for name, blob in first.items():
            assert _read(repo / name) == blob, f"{name} changed on rebuild"


# ---------------------------------------------------------------------------
# Stability: new data never moves existing districts or claims
# ---------------------------------------------------------------------------

class TestStability:
    def test_existing_coordinates_never_move(self, tmp_path):
        repo = _seed_repo(tmp_path)
        atlas1 = write_atlas(repo)
        anchors1 = _read(repo / "data" / "atlas_districts.jsonl")
        claims1 = {c["id"]: (c["x"], c["y"], c["d"]) for c in atlas1["claims"]}
        districts1 = {d["id"]: (d["x"], d["y"]) for d in atlas1["districts"]}

        # One near-duplicate (joins an existing district) and one novel claim
        # (joins or spawns a new anchor — either way nothing else may move).
        dup_id = _add_claim(repo, ALIGNMENT_TEXTS[0])
        novel_id = _add_claim(repo, NOVEL_TEXT)

        atlas2 = write_atlas(repo)
        claims2 = {c["id"]: (c["x"], c["y"], c["d"]) for c in atlas2["claims"]}
        districts2 = {d["id"]: (d["x"], d["y"]) for d in atlas2["districts"]}

        for claim_id, placed in claims1.items():
            assert claims2[claim_id] == placed, f"{claim_id} moved"
        for district_id, centre in districts1.items():
            assert districts2[district_id] == centre, f"{district_id} moved"

        # New claims landed somewhere on the map.
        assert dup_id in claims2 and novel_id in claims2

        # The anchor file is append-only: the old content is a byte prefix.
        anchors2 = _read(repo / "data" / "atlas_districts.jsonl")
        assert anchors2.startswith(anchors1)

    def test_duplicate_claim_joins_existing_district(self, tmp_path):
        if extractor.embed_texts(["probe"]) is None:
            pytest.skip("model2vec unavailable — covered by the fallback tests")

        repo = _seed_repo(tmp_path)
        atlas1 = write_atlas(repo)
        dup_id = _add_claim(repo, ALIGNMENT_TEXTS[0])
        atlas2 = write_atlas(repo)

        by_claim1 = {c["id"]: c["d"] for c in atlas1["claims"]}
        by_claim2 = {c["id"]: c["d"] for c in atlas2["claims"]}
        # Identical text => cosine ~1.0 with the anchor clm_000001 created.
        assert by_claim2[dup_id] == by_claim1["clm_000001"]


# ---------------------------------------------------------------------------
# Embeddings-unavailable fallback: one district per source feed
# ---------------------------------------------------------------------------

class TestFallback:
    def test_one_district_per_feed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(extractor, "embed_texts", lambda texts: None)
        repo = _seed_repo(tmp_path)
        atlas = build_atlas(repo)

        assert len(atlas["districts"]) == 2
        by_claim = {c["id"]: c["d"] for c in atlas["claims"]}
        alignment_districts = {by_claim[f"clm_{i:06d}"] for i in range(1, 5)}
        housing_districts = {by_claim[f"clm_{i:06d}"] for i in range(5, 8)}
        assert len(alignment_districts) == 1
        assert len(housing_districts) == 1
        assert alignment_districts != housing_districts

    def test_fallback_deterministic_and_stable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(extractor, "embed_texts", lambda texts: None)
        repo = _seed_repo(tmp_path)

        atlas1 = write_atlas(repo)
        blob1 = _read(repo / "data" / "atlas.json")
        write_atlas(repo)
        assert _read(repo / "data" / "atlas.json") == blob1

        # A new claim on an existing feed joins that feed's district.
        new_id = _add_claim(repo, NOVEL_TEXT)
        atlas2 = write_atlas(repo)
        old = {c["id"]: (c["x"], c["y"], c["d"]) for c in atlas1["claims"]}
        new = {c["id"]: (c["x"], c["y"], c["d"]) for c in atlas2["claims"]}
        for claim_id, placed in old.items():
            assert new[claim_id] == placed
        by_claim = {c["id"]: c["d"] for c in atlas2["claims"]}
        assert by_claim[new_id] == by_claim["clm_000001"]  # same feed


# ---------------------------------------------------------------------------
# ATLAS.md — the weather report
# ---------------------------------------------------------------------------

class TestAtlasMd:
    def test_header_rows_and_legend(self, tmp_path):
        repo = _seed_repo(tmp_path)
        write_atlas(repo)
        text = (repo / "ATLAS.md").read_text(encoding="utf-8")

        # Header: run id and counts.
        assert "run run-01" in text
        assert "sources 2" in text
        assert "claims 7" in text
        assert "edges 2" in text
        assert "verified 1" in text

        # District rows: density bar glyphs, claim counts, tension marker.
        assert "░" in text or "▓" in text
        assert "claims  ⚡" in text

        # Legend.
        assert "## Legend" in text
        assert "lit" in text and "dim" in text and "darkness" in text

    def test_sharpest_fault_section(self, tmp_path):
        repo = _seed_repo(tmp_path)
        write_atlas(repo)
        text = (repo / "ATLAS.md").read_text(encoding="utf-8")

        assert "## Sharpest fault" in text
        assert "clm_000003 ⇄ clm_000006" in text
        assert "confidence 0.9" in text
        assert ALIGNMENT_TEXTS[2][:60] in text  # claim A text
        assert HOUSING_TEXTS[1][:60] in text  # claim B text
        assert "The alignment problem is hard" in text  # source title

    def test_no_verified_fault_handled(self, tmp_path):
        repo = _make_repo(tmp_path)
        store.append_source(
            repo,
            feed="fake-feed",
            item_key="post-001",
            url="https://example.com/post-001",
            title="Lonely source",
            author="Alice",
            published="2026-06-01",
            sha256="a" * 64,
            claim_count=1,
            run_id="run-01",
        )
        store.append_claim(
            repo,
            claim_id="clm_000001",
            source_id="src_fake-feed_post-001",
            text=ALIGNMENT_TEXTS[0],
            claim_type="empirical",
            support_in_text=0.5,
            quote=ALIGNMENT_TEXTS[0][:100],
            run_id="run-01",
        )
        write_atlas(repo)
        text = (repo / "ATLAS.md").read_text(encoding="utf-8")
        assert "no verified tensions yet" in text
