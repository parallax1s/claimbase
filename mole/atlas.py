"""
Fog-of-war atlas: a deterministic, stable map of the claim graph.

build_atlas(repo_root) -> the atlas contract dict (see ATLAS CONTRACT below).
write_atlas(repo_root) -> builds the atlas, writes data/atlas.json and
ATLAS.md, and appends newly created district anchors to
data/atlas_districts.jsonl.

ATLAS CONTRACT (data/atlas.json) — v2
-------------------------------------
{
  "generated_run": str,
  "extent": {"w": 1000, "h": 1000},
  "bounds": {"x0": float, "y0": float, "x1": float, "y1": float},
  "districts": [{"id", "label", "x", "y", "r", "claims", "lit", "dim", "tensions"}],
  "claims":    [{"id", "x", "y", "s": "lit"|"dim", "d", "type", "text", "src"}],
  "tensions":  [{"a", "b", "note", "verified"}]
}

Coordinates are UNBOUNDED (v2): nothing is clamped, so x/y may be negative
or arbitrarily large.  "bounds" is the bounding box of all content — claim
dots and district discs (centre ± radius) — padded on every side by 5% of
the larger content span.  Renderers must fit "bounds" to the viewport
(letterboxed).  "extent" is kept for backward compatibility only and is no
longer authoritative.

Active claims only.  "lit" = the claim participates in at least one judged
edge; "dim" = extracted, awaiting judgment.  Tensions are the `contradicts`
edges between active claims.

STABILITY BEATS OPTIMALITY
--------------------------
A rerun on the same data must be byte-identical, and new data must never move
existing districts or claims.  Three rules enforce this:

1. District anchors are FROZEN once created — never moved, never re-centered.
   data/atlas_districts.jsonl is append-only; each line is {"id", "label",
   "centroid", "x", "y", "created_run"} (fallback-mode anchors carry an extra
   "feed" key and an empty centroid).  The "label" in the file is the
   creation-time snapshot; display labels are recomputed every build.  A
   better clustering of old claims is always available; we refuse it, because
   a map that shuffles under the reader is worthless.

2. Assignment is an unlock-replay.  Claims are processed in id order (ids are
   monotonic and append-only).  A claim joins the nearest *visible* anchor
   with cosine >= 0.32; when no visible anchor matches, the next persisted
   anchor (in creation order) becomes visible and takes the claim, and once
   the persisted list is exhausted a brand-new anchor is created from the
   claim's own vector (rounded once, so replaying from the file reproduces
   the in-pass arithmetic exactly).  Replay therefore reproduces the anchor
   visibility history of every earlier build: a prior assignment can never be
   stolen by an anchor that did not exist when it was first made.

3. Geometry never looks at the clock or at mutable counts.  An anchor's
   position is computed once, at creation, from the then-existing anchor set
   and frozen in the file (placement v2):

   - anchor 0 sits at the origin (0, 0);
   - a new anchor whose centroid has cosine >= 0.18 with its most similar
     existing anchor ORBITS that anchor: rings at 170, 340, 510, ... units,
     12 candidate angles per ring starting from an angle derived from
     sha256 of the anchor's first claim id and stepping 30 degrees, taking
     the first candidate at least 150 units from every existing anchor;
   - otherwise it takes the first free slot on the global golden-angle
     spiral from the origin (r = 170*sqrt(i), theta = i*2.39996), skipping
     slots closer than 150 units to any existing anchor.

   Similar districts therefore sit next to each other, unrelated ones far
   apart, and the map grows outward without limit.  Claim jitter comes from
   sha256(claim_id) and the district radius *at the moment the claim joined*
   — the radius a district later grows to never moves the dots already
   inside it.

Retiring a claim can orphan an anchor (it stays in the file, unrendered) or
shift which claim unlocks it during replay; anchors still never move.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from mole import store

# ---------------------------------------------------------------------------
# Geometry and contract constants
# ---------------------------------------------------------------------------

EXTENT_W = 1000          # legacy "extent" contract key; no longer authoritative
EXTENT_H = 1000
_SPIRAL_STEP = 170.0     # global spiral radius = 170 * sqrt(spiral index)
_GOLDEN_ANGLE = 2.39996  # radians per spiral index
_ORBIT_STEP = 170.0      # orbit ring distance = 170 * ring number
_ORBIT_ANGLES = 12       # candidate angles per orbit ring (30 degree steps)
_ORBIT_COSINE = 0.18     # at or above this, a new anchor orbits its nearest kin
_MIN_ANCHOR_DIST = 150.0 # no anchor is ever placed within this of another
_BOUNDS_PAD = 0.05       # bounds padding fraction of the larger content span
_R_BASE = 40.0           # district radius = min(40 + 14*sqrt(members), 170)
_R_SCALE = 14.0
_R_CAP = 170.0
_DOT_PAD = 6.0           # keep dots inside the district stroke
_ASSIGN_COSINE = 0.32    # claim joins the nearest anchor at or above this
_CENTROID_DECIMALS = 4   # rounded at creation so replay == reload
_TEXT_CHARS = 140
_SRC_CHARS = 40
_NOTE_CHARS = 140
_BAR_WIDTH = 10

# ---------------------------------------------------------------------------
# Tokenisation for district labels (mirrors the extractor's content tokens)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9']+")
_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "as", "at", "by", "is", "are", "was", "were", "be", "been", "being", "that",
    "this", "it", "its", "they", "their", "them", "we", "our", "you", "your",
    "he", "she", "his", "her", "i", "not", "no", "so", "if", "then", "than",
    "there", "here", "what", "which", "who", "how", "why", "when", "from", "into",
    "out", "up", "down", "over", "about", "would", "could", "will", "can", "may",
    "might", "do", "does", "did", "have", "has", "had", "more", "most", "some",
    "any", "all", "one", "also", "such", "these", "those", "very", "just", "like",
})


def _tokens(text: str) -> list[str]:
    """Lowercase content tokens: stopwords and length<=2 tokens removed."""
    return [
        t for t in _WORD_RE.findall(text.lower())
        if len(t) > 2 and t not in _STOP_WORDS
    ]


# ---------------------------------------------------------------------------
# Cosine similarity (numpy-accelerated when available, stdlib otherwise)
# ---------------------------------------------------------------------------

def _numpy() -> Any:
    try:
        import numpy
    except ImportError:  # pragma: no cover — numpy ships with model2vec
        return None
    return numpy


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


class _AnchorIndex:
    """Nearest-anchor cosine lookup over the visible anchor centroids."""

    def __init__(self) -> None:
        self._vecs: list[list[float]] = []
        self._np = _numpy()
        self._mat: Any = None
        self._norms: Any = None

    def add(self, vec: list[float]) -> None:
        self._vecs.append([float(v) for v in vec])
        self._mat = None  # rebuilt lazily on the next query

    def nearest(self, vec: list[float]) -> tuple[int, float]:
        """Return (index, cosine) of the nearest anchor, or (-1, -1.0)."""
        if not self._vecs:
            return -1, -1.0
        np = self._np
        if np is not None:
            if self._mat is None:
                self._mat = np.asarray(self._vecs, dtype="float64")
                norms = np.linalg.norm(self._mat, axis=1)
                norms[norms == 0.0] = 1.0
                self._norms = norms
            v = np.asarray(vec, dtype="float64")
            v_norm = float(np.linalg.norm(v)) or 1.0
            sims = (self._mat @ v) / (self._norms * v_norm)
            best = int(np.argmax(sims))
            return best, float(sims[best])
        best_i, best_sim = -1, -1.0
        for i, anchor_vec in enumerate(self._vecs):
            sim = _cosine(anchor_vec, vec)
            if sim > best_sim:
                best_i, best_sim = i, sim
        return best_i, best_sim


# ---------------------------------------------------------------------------
# Deterministic geometry
# ---------------------------------------------------------------------------

def _hash_angle(claim_id: str) -> float:
    """Deterministic angle in [0, 2π) from sha256 of a claim id."""
    digest = hashlib.sha256(claim_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64 * 2.0 * math.pi


def _clear(x: float, y: float, occupied: list[tuple[float, float]]) -> bool:
    """True when (x, y) is at least _MIN_ANCHOR_DIST from every occupied point."""
    limit = _MIN_ANCHOR_DIST * _MIN_ANCHOR_DIST
    return all((x - ox) ** 2 + (y - oy) ** 2 >= limit for ox, oy in occupied)


def _spiral_position(
    start_index: int,
    occupied: list[tuple[float, float]],
) -> tuple[float, float, int]:
    """First free slot on the global golden-angle spiral from the origin.

    Scans indices start_index, start_index+1, ... (r = 170*sqrt(i),
    theta = i*2.39996), skipping slots closer than _MIN_ANCHOR_DIST to any
    existing anchor.  Returns (x, y, next start index).
    """
    g = start_index
    while True:
        r = _SPIRAL_STEP * math.sqrt(g)
        theta = g * _GOLDEN_ANGLE
        x = round(r * math.cos(theta), 2)
        y = round(r * math.sin(theta), 2)
        if _clear(x, y, occupied):
            return x, y, g + 1
        g += 1


def _orbit_position(
    first_claim_id: str,
    px: float,
    py: float,
    occupied: list[tuple[float, float]],
) -> tuple[float, float]:
    """First free slot orbiting the parent anchor at (px, py).

    Rings at 170, 340, 510, ... units; on each ring, 12 candidate angles
    starting from sha256(first_claim_id) and stepping 30 degrees.  The first
    candidate at least _MIN_ANCHOR_DIST from every existing anchor wins.
    """
    theta0 = _hash_angle(first_claim_id)
    step = 2.0 * math.pi / _ORBIT_ANGLES
    ring = 1
    while True:
        d = _ORBIT_STEP * ring
        for j in range(_ORBIT_ANGLES):
            theta = theta0 + j * step
            x = round(px + d * math.cos(theta), 2)
            y = round(py + d * math.sin(theta), 2)
            if _clear(x, y, occupied):
                return x, y
        ring += 1


def _district_radius(member_count: int) -> float:
    return min(_R_BASE + _R_SCALE * math.sqrt(member_count), _R_CAP)


def _jitter(claim_id: str, join_index: int) -> tuple[float, float]:
    """Deterministic offset inside the district disc.

    Angle and radial fraction come from sha256(claim_id); the radial fraction
    is sqrt-distributed so dots fill the disc evenly.  The disc radius is the
    district radius at the moment this claim joined (member #join_index), so
    later growth of the district never moves this dot.
    """
    digest = hashlib.sha256(claim_id.encode("utf-8")).digest()
    u_theta = int.from_bytes(digest[:8], "big") / 2.0**64
    u_rad = int.from_bytes(digest[8:16], "big") / 2.0**64
    disc = max(_district_radius(join_index) - _DOT_PAD, 0.0)
    rad = math.sqrt(u_rad) * disc
    theta = u_theta * 2.0 * math.pi
    return rad * math.cos(theta), rad * math.sin(theta)


# ---------------------------------------------------------------------------
# Anchor persistence (append-only)
# ---------------------------------------------------------------------------

def _anchors_path(repo_root: Path) -> Path:
    return repo_root / "data" / "atlas_districts.jsonl"


def _load_anchors(repo_root: Path) -> list[dict[str, Any]]:
    return list(store._iter_jsonl(_anchors_path(repo_root)))


# ---------------------------------------------------------------------------
# District assignment
# ---------------------------------------------------------------------------

def _assign_semantic(
    claims: list[dict[str, Any]],
    vectors: list[list[float]],
    persisted: list[dict[str, Any]],
    run_id: str,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Unlock-replay assignment against frozen anchors (module docstring, rule 2).

    Returns (claim_id -> anchor_id, newly created anchor records).
    """
    semantic = [a for a in persisted if a.get("centroid")]
    index = _AnchorIndex()
    visible: list[dict[str, Any]] = []
    # Every persisted anchor blocks space, visible or not: a slot taken on an
    # earlier run stays taken, so placement is append-stable across runs.
    occupied = [(float(a["x"]), float(a["y"])) for a in persisted]
    next_unlock = 0
    next_creation = len(persisted)  # anchor ids count ALL anchors ever
    spiral_idx = 0
    created: list[dict[str, Any]] = []
    assignment: dict[str, str] = {}

    for claim, vec in zip(claims, vectors):
        best, sim = index.nearest(vec)
        if best >= 0 and sim >= _ASSIGN_COSINE:
            assignment[claim["id"]] = visible[best]["id"]
            continue
        if next_unlock < len(semantic):
            anchor = semantic[next_unlock]
            next_unlock += 1
        else:
            if best >= 0 and sim >= _ORBIT_COSINE:
                parent = visible[best]
                x, y = _orbit_position(
                    claim["id"], float(parent["x"]), float(parent["y"]), occupied
                )
            else:
                x, y, spiral_idx = _spiral_position(spiral_idx, occupied)
            anchor = {
                "id": f"d_{next_creation + 1:03d}",
                "label": "",  # filled in once members are known
                "centroid": [round(float(v), _CENTROID_DECIMALS) for v in vec],
                "x": x,
                "y": y,
                "created_run": run_id,
            }
            created.append(anchor)
            occupied.append((x, y))
            next_creation += 1
        visible.append(anchor)
        index.add(anchor["centroid"])
        assignment[claim["id"]] = anchor["id"]

    return assignment, created


def _feed_of(claim: dict[str, Any], sources_by_id: dict[str, dict[str, Any]]) -> str:
    src = sources_by_id.get(claim.get("source_id", ""))
    if src and src.get("feed"):
        return src["feed"]
    src_id = claim.get("source_id", "")
    if src_id.startswith("src_"):
        return src_id[len("src_"):].split("_", 1)[0] or "unknown"
    return "unknown"


def _assign_by_feed(
    claims: list[dict[str, Any]],
    sources_by_id: dict[str, dict[str, Any]],
    persisted: list[dict[str, Any]],
    run_id: str,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Embeddings-unavailable fallback: one district per source feed.

    Feed anchors carry no centroid, so there is no similarity to orbit by:
    every new anchor takes the next free slot on the global spiral.
    """
    by_feed = {a["feed"]: a for a in persisted if a.get("feed")}
    occupied = [(float(a["x"]), float(a["y"])) for a in persisted]
    next_creation = len(persisted)
    spiral_idx = 0
    created: list[dict[str, Any]] = []
    assignment: dict[str, str] = {}

    for claim in claims:
        feed = _feed_of(claim, sources_by_id)
        anchor = by_feed.get(feed)
        if anchor is None:
            x, y, spiral_idx = _spiral_position(spiral_idx, occupied)
            anchor = {
                "id": f"d_{next_creation + 1:03d}",
                "label": "",
                "centroid": [],
                "x": x,
                "y": y,
                "created_run": run_id,
                "feed": feed,
            }
            created.append(anchor)
            occupied.append((x, y))
            by_feed[feed] = anchor
            next_creation += 1
        assignment[claim["id"]] = anchor["id"]

    return assignment, created


# ---------------------------------------------------------------------------
# District labels
# ---------------------------------------------------------------------------

def _district_label(
    member_claims: list[dict[str, Any]],
    global_counts: Counter,
    global_total: int,
) -> str:
    """Top 3 distinctive content tokens of the members, joined with ' · '."""
    counts: Counter = Counter()
    for claim in member_claims:
        counts.update(_tokens(claim.get("text", "")))
    if not counts or global_total <= 0:
        return ""

    def _score(item: tuple[str, int]) -> float:
        token, count = item
        return count * math.log(1.0 + global_total / global_counts[token])

    top = sorted(counts.items(), key=lambda item: (-_score(item), item[0]))[:3]
    return " · ".join(token for token, _ in top)


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

def _bounds(
    districts: list[dict[str, Any]],
    claims: list[dict[str, Any]],
) -> dict[str, float]:
    """Bounding box of all content (claim dots and district discs), padded.

    Every side is padded by _BOUNDS_PAD of the larger content span; an empty
    or single-point atlas falls back to a fixed pad so renderers never see a
    degenerate box.
    """
    xs: list[float] = []
    ys: list[float] = []
    for d in districts:
        xs.extend((d["x"] - d["r"], d["x"] + d["r"]))
        ys.extend((d["y"] - d["r"], d["y"] + d["r"]))
    for c in claims:
        xs.append(c["x"])
        ys.append(c["y"])
    if not xs:
        return {"x0": 0.0, "y0": 0.0, "x1": float(EXTENT_W), "y1": float(EXTENT_H)}
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad = _BOUNDS_PAD * max(x1 - x0, y1 - y0) or 50.0
    return {
        "x0": round(x0 - pad, 2),
        "y0": round(y0 - pad, 2),
        "x1": round(x1 + pad, 2),
        "y1": round(y1 + pad, 2),
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _build(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Return (atlas contract dict, ATLAS.md extras, newly created anchors)."""
    sources = store.load_all_sources(repo_root)
    all_claims = store.load_all_claims(repo_root)
    edges = store.load_all_edges(repo_root)

    active = sorted(
        (c for c in all_claims if c.get("status") != "retired"),
        key=lambda c: c.get("id", ""),
    )
    active_ids = {c["id"] for c in active}
    sources_by_id = {s.get("id", ""): s for s in sources}

    run_ids = [s.get("run_id", "") for s in sources if s.get("run_id")]
    generated_run = max(run_ids) if run_ids else "unknown"

    lit_ids: set[str] = set()
    for edge in edges:
        lit_ids.add(edge.get("a", ""))
        lit_ids.add(edge.get("b", ""))

    persisted = _load_anchors(repo_root)

    # Lazy import so a monkeypatched extractor.embed_texts always wins.
    import extractor as extractor_mod

    vectors = extractor_mod.embed_texts([c["text"] for c in active]) if active else None
    if vectors is not None:
        assignment, created = _assign_semantic(active, vectors, persisted, generated_run)
        mode = "semantic"
    else:
        assignment, created = _assign_by_feed(active, sources_by_id, persisted, generated_run)
        mode = "feed"

    anchors_by_id = {a["id"]: a for a in persisted}
    anchors_by_id.update({a["id"]: a for a in created})

    members: dict[str, list[dict[str, Any]]] = {}
    for claim in active:
        members.setdefault(assignment[claim["id"]], []).append(claim)

    # Tensions: contradicts edges between active claims
    tension_edges = sorted(
        (
            e for e in edges
            if e.get("relation") == "contradicts"
            and e.get("a") in active_ids
            and e.get("b") in active_ids
        ),
        key=lambda e: (e.get("a", ""), e.get("b", "")),
    )
    district_tensions: dict[str, int] = {}
    for edge in tension_edges:
        for aid in {assignment[edge["a"]], assignment[edge["b"]]}:
            district_tensions[aid] = district_tensions.get(aid, 0) + 1

    # Claim dots (id order; join_index per district drives the jitter disc)
    claims_out: list[dict[str, Any]] = []
    join_counts: dict[str, int] = {}
    for claim in active:
        aid = assignment[claim["id"]]
        join_counts[aid] = join_counts.get(aid, 0) + 1
        anchor = anchors_by_id[aid]
        dx, dy = _jitter(claim["id"], join_counts[aid])
        x = anchor["x"] + dx
        y = anchor["y"] + dy
        src = sources_by_id.get(claim.get("source_id", ""), {})
        claims_out.append({
            "id": claim["id"],
            "x": round(x, 2),
            "y": round(y, 2),
            "s": "lit" if claim["id"] in lit_ids else "dim",
            "d": aid,
            "type": claim.get("type", "other"),
            "text": claim.get("text", "")[:_TEXT_CHARS],
            "src": src.get("title", "")[:_SRC_CHARS],
        })

    # Labels: recomputed every build; the anchor file keeps the creation-time
    # snapshot only (new anchors get this build's label as that snapshot).
    global_counts: Counter = Counter()
    for claim in active:
        global_counts.update(_tokens(claim.get("text", "")))
    global_total = sum(global_counts.values())

    labels = {
        aid: _district_label(member_list, global_counts, global_total) or aid
        for aid, member_list in members.items()
    }
    for anchor in created:
        anchor["label"] = labels.get(anchor["id"], anchor["id"])

    districts_out: list[dict[str, Any]] = []
    for aid in sorted(members):
        member_list = members[aid]
        anchor = anchors_by_id[aid]
        lit = sum(1 for c in member_list if c["id"] in lit_ids)
        if len(member_list) < 2:
            continue
        districts_out.append({
            "id": aid,
            "label": labels[aid],
            "x": anchor["x"],
            "y": anchor["y"],
            "r": round(_district_radius(len(member_list)), 2),
            "claims": len(member_list),
            "lit": lit,
            "dim": len(member_list) - lit,
            "tensions": district_tensions.get(aid, 0),
        })

    tensions_out = [
        {
            "a": e["a"],
            "b": e["b"],
            "note": (e.get("note") or "")[:_NOTE_CHARS],
            "verified": bool(e.get("verified")),
        }
        for e in tension_edges
    ]

    atlas: dict[str, Any] = {
        "generated_run": generated_run,
        "extent": {"w": EXTENT_W, "h": EXTENT_H},
        "bounds": _bounds(districts_out, claims_out),
        "districts": districts_out,
        "claims": claims_out,
        "tensions": tensions_out,
    }

    extras: dict[str, Any] = {
        "mode": mode,
        "sources": len(sources),
        "edges": len(edges),
        "verified": sum(1 for e in edges if e.get("verified")),
        "fault": _sharpest_fault(tension_edges, {c["id"]: c for c in claims_out}),
    }
    return atlas, extras, created


def _sharpest_fault(
    tension_edges: list[dict[str, Any]],
    claims_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """The highest-confidence verified tension, or None."""
    verified = [e for e in tension_edges if e.get("verified")]
    if not verified:
        return None
    best = max(
        verified,
        key=lambda e: (e.get("confidence", 0.0), e.get("a", ""), e.get("b", "")),
    )
    a = claims_by_id[best["a"]]
    b = claims_by_id[best["b"]]
    return {
        "a": best["a"],
        "b": best["b"],
        "confidence": best.get("confidence", 0.0),
        "note": (best.get("note") or "")[:_NOTE_CHARS],
        "a_text": a["text"],
        "a_src": a["src"],
        "b_text": b["text"],
        "b_src": b["src"],
    }


# ---------------------------------------------------------------------------
# ATLAS.md — the ASCII weather report
# ---------------------------------------------------------------------------

def _render_md(atlas: dict[str, Any], extras: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Atlas")
    lines.append("")
    lines.append("Fog-of-war over the claim graph: lit where judged, dim where extracted,")
    lines.append("dark where the mole has not yet dug.")
    lines.append("")
    lines.append(
        f"run {atlas['generated_run']} · sources {extras['sources']} · "
        f"claims {len(atlas['claims'])} · edges {extras['edges']} · "
        f"verified {extras['verified']}"
    )
    if extras["mode"] == "feed" and atlas["claims"]:
        lines.append("")
        lines.append("(embeddings unavailable — districts grouped by source feed)")
    lines.append("")
    lines.append("## Districts")
    lines.append("")
    districts = sorted(atlas["districts"], key=lambda d: (-d["claims"], d["id"]))
    shown, rest = districts[:25], districts[25:]
    if shown:
        label_w = max(len(d["label"]) for d in shown)
        lines.append("```text")
        for d in shown:
            frac = d["lit"] / d["claims"] if d["claims"] else 0.0
            n_lit = int(round(_BAR_WIDTH * frac))
            bar = "▓" * n_lit + "░" * (_BAR_WIDTH - n_lit)
            lines.append(
                f"{d['label']:<{label_w}}  {bar}  {d['claims']:>5} claims  ⚡{d['tensions']}"
            )
        if rest:
            rest_claims = sum(d["claims"] for d in rest)
            lines.append(f"… and {len(rest)} smaller districts ({rest_claims} claims)")
        lines.append("```")
    else:
        lines.append("(no districts yet — the map is all darkness)")
    lines.append("")
    lines.append("## Sharpest fault")
    lines.append("")
    fault = extras["fault"]
    if fault:
        lines.append(
            f"⚡ {fault['a']} ⇄ {fault['b']} · confidence {fault['confidence']} · verified"
        )
        lines.append("")
        lines.append(f"- A: \"{fault['a_text']}\" — {fault['a_src']}")
        lines.append(f"- B: \"{fault['b_text']}\" — {fault['b_src']}")
        lines.append(f"- note: {fault['note']}")
    else:
        lines.append("(no verified tensions yet — the fault lines are still unjudged)")
    lines.append("")
    lines.append("## Legend")
    lines.append("")
    lines.append("- ▓ lit — the claim sits on at least one judged edge")
    lines.append("- ░ dim — extracted, awaiting judgment")
    lines.append("- ⚡ tension — a `contradicts` edge between two claims")
    lines.append(
        "- darkness — territory the mole has not yet reached: "
        "unfetched sources, unjudged pairs"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_atlas(repo_root: Path) -> dict[str, Any]:
    """Build and return the atlas contract dict (writes nothing)."""
    atlas, _extras, _created = _build(repo_root)
    return atlas


def write_atlas(repo_root: Path) -> dict[str, Any]:
    """Build the atlas; write data/atlas.json and ATLAS.md.

    Newly created district anchors are appended to data/atlas_districts.jsonl
    (the file is append-only: existing lines are never rewritten).
    Returns the atlas dict.
    """
    atlas, extras, created = _build(repo_root)

    if created:
        anchors_path = _anchors_path(repo_root)
        anchors_path.parent.mkdir(parents=True, exist_ok=True)
        with anchors_path.open("a", encoding="utf-8") as fh:
            for anchor in created:
                fh.write(json.dumps(anchor, ensure_ascii=False) + "\n")

    atlas_path = repo_root / "data" / "atlas.json"
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    with atlas_path.open("w", encoding="utf-8") as fh:
        json.dump(atlas, fh, indent=2, ensure_ascii=False)

    (repo_root / "ATLAS.md").write_text(_render_md(atlas, extras), encoding="utf-8")

    return atlas
