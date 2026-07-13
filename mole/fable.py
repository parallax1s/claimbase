"""
Worker-side deep extraction via the episteme-fable engine.

The mole cron enqueues `extract` tasks (one per new/changed source); an LLM
worker session drains them with `python -m mole fable ...`. This module:

  1. refetches the source text (arxiv API for arxiv-* feeds, generic
     HTML-strip otherwise; `--text-file` overrides for anything else),
  2. runs the episteme-fable pipeline (self-contained rewritten claims,
     validated deterministically; section points; document theses),
  3. retires the source's prior mole-extracted claims (superseded),
     marks their pending refine tasks obsolete,
  4. appends fable-tier claims and data/theses.jsonl rows,
  5. enqueues attach tasks for the new claims (lexical, calibrated floor)
     and theses (hybrid lexical/embedding — theses are conceptual, which
     is exactly where lexical matching fails),
  6. flips this source's `extract` tasks to done.

The source record's content_sha256 is left UNTOUCHED on refetch drift:
that hash belongs to the feed-cleaned text and is what the cron's
changed-item detection compares against. Overwriting it would make the
next cron run see every fable-extracted source as "changed" and clobber
the fable claims with regex re-extraction. Drift is recorded on the claim
rows as a `content_drift` flag instead.

Engine location: env EPISTEME_FABLE_SRC (default: the sibling
episteme-fable repo's src/). Requires the `claude` CLI on PATH.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from mole import store

DEFAULT_ENGINE_SRC = "/Users/mo/Desktop/Prj.nosync/programming/episteme-fable/src"
QUOTE_CAP = 300
SUPPORT_IN_TEXT_NEUTRAL = 0.5  # fable rows carry epistemics in stance/hedge;
                               # this legacy field is kept schema-compatible


# ---------------------------------------------------------------------------
# Engine bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_engine():
    src = os.environ.get("EPISTEME_FABLE_SRC", DEFAULT_ENGINE_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from episteme_fable.pipeline import analyze
        from episteme_fable.providers import ClaudeCLIProvider
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            f"episteme-fable engine not importable from {src!r} "
            f"(set EPISTEME_FABLE_SRC): {exc}"
        ) from exc
    return analyze, ClaudeCLIProvider


# ---------------------------------------------------------------------------
# Source refetch
# ---------------------------------------------------------------------------

_ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/([^\s?#]+)")


def refetch_text(source_rec: dict[str, Any]) -> tuple[str, bool]:
    """Return (text, drift). drift=True when the refetched text's sha does
    not match the stored content_sha256 (feed-cleaned original)."""
    from mole import feeds as feeds_mod

    url = source_rec.get("url", "")
    feed = source_rec.get("feed", "")
    m = _ARXIV_ID_RE.search(url)
    if feed.startswith("arxiv") and m:
        api = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"id_list": m.group(1), "max_results": "1"}
        )
        xml_text = feeds_mod._fetch_text(api, headers={"User-Agent": feeds_mod.USER_AGENT})
        root = ET.fromstring(xml_text)
        entry = next((n for n in root.iter() if feeds_mod._localname(n.tag) == "entry"), None)
        if entry is None:
            raise RuntimeError(f"arxiv API returned no entry for {url}")
        title = feeds_mod._first_child_text(entry, ["title"], default="")
        abstract = feeds_mod._first_child_text(entry, ["summary"], default="")
        text = f"{title}\n\n{feeds_mod._strip_html(str(abstract))}".strip()
    else:
        html = feeds_mod._fetch_text(url, headers={"User-Agent": feeds_mod.USER_AGENT})
        text = feeds_mod._strip_html(html)

    drift = store.content_sha256(text) != source_rec.get("content_sha256", "")
    return text, drift


# ---------------------------------------------------------------------------
# Artifact -> claimbase records (pure; unit-testable without the engine)
# ---------------------------------------------------------------------------

def _front_truncate(text: str, cap: int = QUOTE_CAP) -> str:
    text = " ".join(text.split())
    if len(text) <= cap:
        return text
    return "..." + text[-(cap - 3):]


def map_artifact(
    *,
    claims: list[dict[str, Any]],
    theses: list[dict[str, Any]],
    source_id: str,
    run_id: str,
    claim_start: int,
    thesis_start: int,
    drift: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map episteme-fable claim/thesis dicts onto claimbase records.

    claim_start/thesis_start are the last already-used numeric ids.
    Returns (claim_records, thesis_records); thesis claim_ids are remapped
    from engine ids to the new clm_ ids.
    """
    id_map: dict[str, str] = {}
    claim_records: list[dict[str, Any]] = []
    n = claim_start
    for c in claims:
        n += 1
        cid = f"clm_{n:06d}"
        id_map[c.get("id", "")] = cid
        quote = " | ".join(sp.get("quote", "") for sp in c.get("spans", []))
        flags = list(c.get("flags", []))
        if drift and "content_drift" not in flags:
            flags.append("content_drift")
        rec = store.make_claim_record(
            claim_id=cid,
            source_id=source_id,
            text=c.get("text", ""),
            claim_type=c.get("claim_type") or c.get("kind", "descriptive"),
            support_in_text=SUPPORT_IN_TEXT_NEUTRAL,
            quote=_front_truncate(quote),
            run_id=run_id,
            status="extracted",
        )
        rec.update({
            "tier": c.get("tier", "validated"),
            "extractor": f"episteme-fable/{c.get('engine_version', '')}",
            "prompt_version": c.get("prompt_version", ""),
            "kind": c.get("kind", "assertion"),
            "stance": c.get("stance", "author"),
            "stance_source": c.get("stance_source"),
            "hedge": c.get("hedge", [0.5, 0.9]),
            "flags": flags,
        })
        claim_records.append(rec)

    thesis_records: list[dict[str, Any]] = []
    t = thesis_start
    for th in theses:
        t += 1
        refs = [id_map[i] for i in th.get("claim_ids", []) if i in id_map]
        # theses may cite points; resolve point-cited claims transitively
        # is the engine's job — here we keep whatever claim ids survive.
        thesis_records.append({
            "id": f"th_{t:06d}",
            "source_id": source_id,
            "text": th.get("text", ""),
            "claim_ids": refs,
            "tier": th.get("tier", "unreviewed_ai_draft"),
            "run_id": run_id,
            "status": "proposed",
            "flags": list(th.get("flags", [])),
        })
    return claim_records, thesis_records


def resolve_thesis_claims(
    theses: list[dict[str, Any]],
    points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten point citations into claim ids so claimbase theses ground in
    claims directly (claimbase has no point layer)."""
    by_point = {p.get("id"): p.get("claim_ids", []) for p in points}
    out = []
    for th in theses:
        claim_ids = list(th.get("claim_ids", []))
        for pid in th.get("point_ids", []):
            claim_ids.extend(by_point.get(pid, []))
        merged = dict(th)
        merged["claim_ids"] = list(dict.fromkeys(claim_ids))
        out.append(merged)
    return out


# ---------------------------------------------------------------------------
# The worker entry
# ---------------------------------------------------------------------------

def run_fable(
    repo_root: Path,
    *,
    source_id: str,
    run_id: str,
    model: str | None = None,
    assemble_model: str | None = None,
    text: str | None = None,
) -> dict[str, Any]:
    sources = {s["id"]: s for s in store.load_all_sources(repo_root)}
    if source_id not in sources:
        raise SystemExit(f"unknown source: {source_id}")
    src = sources[source_id]

    drift = False
    if text is None:
        text, drift = refetch_text(src)
    if not text.strip():
        raise SystemExit(f"no text for {source_id}")

    analyze, ClaudeCLIProvider = _bootstrap_engine()
    provider = ClaudeCLIProvider(model=model) if model else ClaudeCLIProvider()
    art = analyze(
        text,
        doc_id=source_id,
        title=src.get("title", ""),
        provider=provider,
        model=model,
        assemble_model=assemble_model,
    )

    claim_dicts = [c.to_dict() for c in art.claims]
    thesis_dicts = resolve_thesis_claims(
        [t.to_dict() for t in art.theses],
        [p.to_dict() for p in art.points],
    )

    claim_start = int(store.next_claim_id(repo_root).split("_")[1]) - 1
    thesis_start = int(store.next_thesis_id(repo_root).split("_")[1]) - 1
    claim_records, thesis_records = map_artifact(
        claims=claim_dicts,
        theses=thesis_dicts,
        source_id=source_id,
        run_id=run_id,
        claim_start=claim_start,
        thesis_start=thesis_start,
        drift=drift,
    )

    # Supersede: retire prior claims, obsolete their pending refine tasks.
    prior_claim_ids = {
        c["id"]
        for c in store.load_all_claims(repo_root)
        if c.get("source_id") == source_id and c.get("status") != "retired"
    }
    retired = store.retire_claims_for_source(repo_root, source_id)
    obsoleted = store.update_tasks(
        repo_root,
        lambda t: (
            t.get("status") == "pending"
            and t.get("kind") == "refine"
            and t.get("payload", {}).get("claim_id") in prior_claim_ids
        ),
        status="obsolete",
        note=f"claim retired; source re-extracted by {run_id}",
    )

    store.append_claims_batch(repo_root, claim_records)
    store.append_theses_batch(repo_root, thesis_records)
    store.bump_source_extraction(
        repo_root,
        source_id=source_id,
        claim_count=len(claim_records),
        fable_run=run_id,
    )

    # Attach candidates: claims (calibrated lexical) + theses (hybrid).
    from mole import attach as attach_mod

    attach_claims = attach_theses = 0
    questions = store.load_live_questions(repo_root)
    if questions:
        live_texts = [
            c["text"] for c in store.load_all_claims(repo_root)
            if c.get("status") != "retired"
        ]
        index = attach_mod.build_index(live_texts, questions)
        already = {
            (t["payload"].get("question_id"), t["payload"].get("claim_id"))
            for t in store.load_all_tasks(repo_root)
            if t.get("kind") == "attach"
        }
        task_max = int(store.next_task_id(repo_root).split("_")[1]) - 1
        for claim_id, question_id, sim in attach_mod.candidates(claim_records, index):
            if (question_id, claim_id) in already:
                continue
            task_max += 1
            store.append_task(
                repo_root, task_id=f"task_{task_max:06d}", kind="attach",
                payload={"question_id": question_id, "claim_id": claim_id, "sim": sim},
                created_run=run_id,
            )
            attach_claims += 1
        for thesis_id, question_id, sim in attach_mod.candidates_hybrid(
            thesis_records, index, questions
        ):
            if (question_id, thesis_id) in already:
                continue
            task_max += 1
            store.append_task(
                repo_root, task_id=f"task_{task_max:06d}", kind="attach",
                payload={"question_id": question_id, "claim_id": thesis_id, "sim": sim},
                created_run=run_id,
            )
            attach_theses += 1

    # Flip this source's extract tasks.
    done = store.update_tasks(
        repo_root,
        lambda t: (
            t.get("status") == "pending"
            and t.get("kind") == "extract"
            and t.get("payload", {}).get("source_id") == source_id
        ),
        status="done",
        note=f"extracted by {run_id}: {len(claim_records)} claims, "
             f"{len(thesis_records)} theses",
    )

    return {
        "source_id": source_id,
        "run_id": run_id,
        "content_drift": drift,
        "claims_added": len(claim_records),
        "theses_added": len(thesis_records),
        "claims_retired": retired,
        "refine_tasks_obsoleted": obsoleted,
        "extract_tasks_done": done,
        "attach_enqueued": {"claims": attach_claims, "theses": attach_theses},
        "funnel": art.stats,
    }


def pending_extract_sources(repo_root: Path) -> list[str]:
    """source_ids with a pending extract task, oldest task first."""
    out: list[str] = []
    for t in store.load_all_tasks(repo_root):
        if t.get("kind") == "extract" and t.get("status") == "pending":
            sid = t.get("payload", {}).get("source_id")
            if sid and sid not in out:
                out.append(sid)
    return out
