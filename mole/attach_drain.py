"""Attach drain: judge pending attach tasks (claim -> question bearing).

The question layer's judgment joint: candidates are enqueued by the lexical/
hybrid matcher (mole/attach.py) but nothing judged them after the 2026-07-02
seed. This drain judges them with the WORKER.md attach rubric, batched per
question, through the episteme-fable provider layer (claude -p by default,
covered by the Max plan's API credits when ANTHROPIC_API_KEY is set).

Scope rules (deterministic, before any model call):
- only tasks whose claim is live AND tier == "validated" are judged;
- tasks whose claim is retired are flipped to `obsolete` (no model call);
- thesis candidates and untiered legacy claims stay `pending` (out of scope).

Outputs follow SCHEMA.md: an attachment row per relevant candidate
(strength >= STRENGTH_FLOOR), unique on (question_id, claim_id); irrelevant
candidates store no row — the task flip to `done` is the record. Every write
is a per-batch atomic rewrite, so a crash loses at most one batch.

    python -m mole attach-drain --run-id attach-20261029 --dry-run
    python -m mole attach-drain --run-id attach-20261029 --limit 2      # two batches
    python -m mole attach-drain --run-id attach-20261029                # everything in scope
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mole import store

DEFAULT_ENGINE_SRC = "/Users/mo/Desktop/Prj.nosync/programming/episteme-fable/src"
DEFAULT_MODEL = os.environ.get("ATTACH_JUDGE_MODEL", "claude-sonnet-5-5")
STRENGTH_FLOOR = 0.4
BATCH = 25
STANCES = {"bears_yes", "bears_no", "informs", "challenges_framing"}

PROMPT = """You judge whether candidate claims genuinely BEAR ON a research question.
Would a researcher tracking this question want the claim in its evidence feed?
Keyword overlap is not bearing. Be strict; most candidates from a lexical
matcher do not bear.

QUESTION ({qid}): {question}
Why it matters: {why}

For each candidate give exactly one of:
- bears_yes / bears_no — evidence or argument for the affirmative / negative pole of the question AS WORDED
- informs — genuine evidence without polarity (a measurement, a case, a mechanism)
- challenges_framing — the claim argues the question itself is confused, ill-posed, or wrongly split
- none — does not bear (irrelevant, tangential, keyword-only overlap, or a fragment)

`strength` is your credence the claim genuinely bears (relevance x evidential
weight), 0 to 1; below {floor} prefer "none". `note` is ONE sentence a reader
can check against the claim and question texts (omit for "none").

CANDIDATES:
{candidates}

Reply with ONLY a JSON array, one object per candidate, in order:
[{{"n": 1, "stance": "informs", "strength": 0.55, "note": "..."}}, {{"n": 2, "stance": "none"}}, ...]"""


def _bootstrap():
    src = os.environ.get("EPISTEME_FABLE_SRC", DEFAULT_ENGINE_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    from episteme_fable.providers import ClaudeCLIProvider, extract_json, make_provider  # type: ignore
    return ClaudeCLIProvider, extract_json, make_provider


def _ask_json(provider, extract_json, prompt: str, model: str | None):
    reply = provider.complete(prompt, model=model)
    data, err = extract_json(reply)
    if err is not None or not isinstance(data, list):
        reply = provider.complete(
            prompt + "\n\nYour previous reply was not a parseable JSON array. Reply with ONLY the JSON array.",
            model=model)
        data, err = extract_json(reply)
    if err is not None or not isinstance(data, list):
        return None
    return data


def plan(repo_root: Path, per_question_cap: int | None = None) -> dict[str, Any]:
    """Classify pending attach tasks. Returns batches (question-grouped,
    sim-descending) plus the ids of tasks whose claim is retired."""
    claims = {c["id"]: c for c in store.load_all_claims(repo_root)}
    theses = {t["id"] for t in store.load_all_theses(repo_root)}
    questions = {q["id"]: q for q in store.load_live_questions(repo_root)}
    sources = {s["id"]: s for s in store.load_all_sources(repo_root)}
    retired_tasks: list[str] = []
    by_q: dict[str, list[dict[str, Any]]] = {}
    counts = {"validated": 0, "retired": 0, "untiered": 0, "thesis": 0, "missing": 0, "no_question": 0}
    for t in store.load_all_tasks(repo_root):
        if t.get("kind") != "attach" or t.get("status") != "pending":
            continue
        cid = t["payload"].get("claim_id")
        qid = t["payload"].get("question_id")
        if cid in theses:
            counts["thesis"] += 1
            continue
        c = claims.get(cid)
        if c is None:
            counts["missing"] += 1
            continue
        if c.get("status") == "retired":
            counts["retired"] += 1
            retired_tasks.append(t["id"])
            continue
        if c.get("tier") != "validated":
            counts["untiered"] += 1
            continue
        if qid not in questions:
            counts["no_question"] += 1
            continue
        counts["validated"] += 1
        by_q.setdefault(qid, []).append({"task": t, "claim": c,
                                         "source": sources.get(c.get("source_id"), {})})
    batches = []
    for qid, items in by_q.items():
        items.sort(key=lambda x: -(x["task"]["payload"].get("sim") or 0))
        if per_question_cap:
            items = items[:per_question_cap]
        for i in range(0, len(items), BATCH):
            batches.append({"question": questions[qid], "items": items[i:i + BATCH]})
    batches.sort(key=lambda b: b["question"]["id"])
    return {"batches": batches, "retired_tasks": retired_tasks, "counts": counts,
            "per_question": {q: len(v) for q, v in by_q.items()}}


def render_candidates(items: list[dict[str, Any]]) -> str:
    lines = []
    for n, it in enumerate(items, 1):
        c, s = it["claim"], it["source"]
        meta = []
        if c.get("stance") and c["stance"] != "author":
            meta.append(f"stance={c['stance']}")
        if c.get("hedge"):
            meta.append(f"hedge={c['hedge']}")
        head = f"[{n}] {c['text']}"
        if meta:
            head += f"  ({', '.join(meta)})"
        lines.append(head)
        if s.get("title"):
            lines.append(f"    source: {s['title'][:120]} ({s.get('published', '')[:10]})")
        if c.get("quote") and c["quote"].strip() != c["text"].strip():
            lines.append(f"    quote: \"{c['quote'][:300]}\"")
    return "\n".join(lines)


def judge_batch(provider, extract_json, batch: dict[str, Any], model: str
                ) -> tuple[list[dict[str, Any]], list[str], str | None]:
    """Returns (attachment rows, judged task ids, error)."""
    q, items = batch["question"], batch["items"]
    prompt = PROMPT.format(qid=q["id"], question=q["text"], why=q.get("why_it_matters", ""),
                           floor=STRENGTH_FLOOR, candidates=render_candidates(items))
    try:
        data = _ask_json(provider, extract_json, prompt, model)
    except Exception as exc:  # provider failure: leave the batch pending
        return [], [], f"{type(exc).__name__}: {str(exc)[:200]}"
    if data is None:
        return [], [], "unparseable reply"
    verdicts: dict[int, dict[str, Any]] = {}
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("n"), int):
            verdicts[item["n"]] = item
    rows, done = [], []
    for n, it in enumerate(items, 1):
        v = verdicts.get(n)
        if v is None:
            continue  # unanswered candidate stays pending
        t = it["task"]
        done.append(t["id"])
        stance = v.get("stance")
        try:
            strength = float(v.get("strength", 0))
        except (TypeError, ValueError):
            strength = 0.0
        if stance in STANCES and strength >= STRENGTH_FLOOR:
            rows.append({
                "question_id": q["id"], "claim_id": it["claim"]["id"],
                "stance": stance, "strength": round(strength, 2),
                "note": str(v.get("note", "")).strip()[:400],
                "judge": model, "verified": False,
                "run_id": t.get("created_run", ""),
            })
    return rows, done, None


def _attachments_path(repo_root: Path) -> Path:
    return repo_root / "data" / "attachments.jsonl"


def write_batch(repo_root: Path, rows: list[dict[str, Any]], done: list[str],
                run_id: str, judge_run_note: str) -> None:
    if rows:
        path = _attachments_path(repo_root)
        existing = list(store._iter_jsonl(path)) if path.exists() else []
        keyed = {(r["question_id"], r["claim_id"]): r for r in existing}
        for r in rows:
            keyed[(r["question_id"], r["claim_id"])] = r  # re-judgment replaces
        store._rewrite_jsonl(path, list(keyed.values()))
    if done:
        ids = set(done)
        store.update_tasks(repo_root, lambda t: t.get("id") in ids, status="done",
                           note=judge_run_note)


def drain(repo_root: Path, run_id: str, model: str = DEFAULT_MODEL, limit: int | None = None,
          per_question_cap: int | None = None, workers: int = 4, dry_run: bool = False,
          provider=None, log=print) -> dict[str, Any]:
    p = plan(repo_root, per_question_cap)
    batches = p["batches"] if limit is None else p["batches"][:limit]
    summary: dict[str, Any] = {
        "run_id": run_id, "model": model, "counts": p["counts"],
        "batches_planned": len(p["batches"]), "batches_run": 0,
        "judged": 0, "attached": 0, "by_stance": {}, "errors": [],
        "retired_tasks_obsoleted": 0, "dry_run": dry_run,
    }
    if dry_run:
        summary["per_question"] = p["per_question"]
        return summary
    if p["retired_tasks"]:
        ids = set(p["retired_tasks"])
        summary["retired_tasks_obsoleted"] = store.update_tasks(
            repo_root, lambda t: t.get("id") in ids, status="obsolete",
            note=f"claim retired before attach judgment; {run_id}")
    if provider is None:
        ClaudeCLIProvider, extract_json, make_provider = _bootstrap()
        # EPF_PROVIDER=anthropic routes through the direct API provider, whose
        # usage ledger and EPF_ANTHROPIC_BUDGET_USD cap then apply; the
        # default stays the claude CLI.
        provider = make_provider(model) if os.environ.get("EPF_PROVIDER") else ClaudeCLIProvider(model=model)
    else:
        _, extract_json, _ = _bootstrap()
    note = f"judged by {model} in {run_id}"
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(judge_batch, provider, extract_json, b, model) for b in batches]
        for b, fut in zip(batches, futures):
            rows, done, err = fut.result()
            summary["batches_run"] += 1
            if err:
                summary["errors"].append(f"{b['question']['id']}: {err}")
                log(f"  {b['question']['id']}: ERROR {err}")
                continue
            write_batch(repo_root, rows, done, run_id, note)  # main thread writes
            summary["judged"] += len(done)
            summary["attached"] += len(rows)
            for r in rows:
                summary["by_stance"][r["stance"]] = summary["by_stance"].get(r["stance"], 0) + 1
            log(f"  {b['question']['id']}: {len(done)} judged, {len(rows)} attached "
                f"({summary['batches_run']}/{len(batches)}, {time.time() - t0:.0f}s)")
    return summary
