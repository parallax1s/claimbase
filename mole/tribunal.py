"""
RICHTSCHWERT — the tribunal: contested questions go to trial.

  python -m mole tribunal --docket
  python -m mole tribunal --question q_000002 --run-id tribunal-20260715

Procedure (LLM proposes, structure disposes — the house discipline):

  MARSHAL   deterministic: judged attachments (claims + theses) above the
            worker strength floor, bucketed by stance, capped per bucket.
            The trial record cites ONLY marshaled evidence, by id.
  ADVOCATES two independent briefs steelman the affirmative (PRO) and
            negative (CONTRA) poles of the question as worded.
  CROSS     each side attacks the other's load-bearing premises.
  BENCH     judges across model families (claude haiku/sonnet via CLI,
            gpt-4.1-mini via GitHub Models — keyless) return independent
            credence INTERVALS + cruxes + would-change-my-mind conditions.
            Dissent is recorded, never averaged away.
  VERDICT   data/verdicts.jsonl, git-ledgered, status='standing', with an
            evidence watermark: new attachments after the trial flag a
            RETRIAL on the docket.

Verdicts carry credences -> they are forecasts -> the calibration backtest
gets teeth. LLM transport is the episteme-fable provider layer
(EPISTEME_FABLE_SRC), same as mole/fable.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from mole import store

DEFAULT_ENGINE_SRC = "/Users/mo/Desktop/Prj.nosync/programming/episteme-fable/src"
STRENGTH_FLOOR = 0.4
BUCKET_CAP = 14

BENCH = [
    {"judge": "claude-haiku", "kind": "claude", "model": "claude-haiku-4-5-20251001"},
    {"judge": "claude-sonnet", "kind": "claude", "model": "claude-sonnet-5"},
    {"judge": "gpt-4.1-mini", "kind": "github", "model": "openai/gpt-4.1-mini"},
]


def _bootstrap():
    src = os.environ.get("EPISTEME_FABLE_SRC", DEFAULT_ENGINE_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    from episteme_fable.providers import (ClaudeCLIProvider, GitHubModelsProvider,
                                          extract_json)
    return ClaudeCLIProvider, GitHubModelsProvider, extract_json


def _github_token() -> str | None:
    if os.environ.get("GITHUB_TOKEN"):
        return os.environ["GITHUB_TOKEN"]
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def bench_providers():
    """Instantiate the bench; judges whose transport is unavailable are
    dropped with a note (an honest smaller bench beats a fake big one)."""
    ClaudeCLI, GitHubModels, _ = _bootstrap()
    judges = []
    notes = []
    for spec in BENCH:
        if spec["kind"] == "claude":
            judges.append({**spec, "provider": ClaudeCLI(model=spec["model"])})
        else:
            token = _github_token()
            if token:
                judges.append({**spec, "provider": GitHubModels(model=spec["model"], token=token)})
            else:
                notes.append(f"{spec['judge']} dropped: no GITHUB_TOKEN/gh auth")
    return judges, notes


# ---------------------------------------------------------------------------
# MARSHAL (deterministic)
# ---------------------------------------------------------------------------

def marshal(repo_root: Path, question_id: str) -> dict[str, Any]:
    questions = {q["id"]: q for q in store.load_live_questions(repo_root)}
    if question_id not in questions:
        raise SystemExit(f"unknown question: {question_id}")
    question = questions[question_id]

    texts: dict[str, str] = {}
    for c in store.load_all_claims(repo_root):
        if c.get("status") != "retired":
            texts[c["id"]] = c["text"]
    for t in store.load_all_theses(repo_root):
        if t.get("status") != "retired":
            texts[t["id"]] = t["text"]

    attachments = []
    path = repo_root / "data" / "attachments.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    attachments.append(json.loads(line))

    total_for_q = 0
    buckets: dict[str, list[dict[str, Any]]] = {
        "bears_yes": [], "bears_no": [], "informs": [], "challenges_framing": [],
    }
    for a in attachments:
        if a.get("question_id") != question_id:
            continue
        total_for_q += 1
        stance = a.get("stance")
        text = texts.get(a.get("claim_id"))
        if stance not in buckets or text is None:
            continue
        if float(a.get("strength", 0)) < STRENGTH_FLOOR:
            continue
        buckets[stance].append({
            "id": a["claim_id"],
            "text": text,
            "strength": float(a.get("strength", 0)),
            "note": a.get("note", ""),
        })
    for stance in buckets:
        buckets[stance].sort(key=lambda e: -e["strength"])
        buckets[stance] = buckets[stance][:BUCKET_CAP]

    return {
        "question": question,
        "buckets": buckets,
        "evidence_watermark": total_for_q,
    }


def _evidence_block(buckets: dict[str, list[dict[str, Any]]]) -> str:
    lines = []
    for stance, rows in buckets.items():
        if not rows:
            continue
        lines.append(f"[{stance.upper()}]")
        for e in rows:
            lines.append(f"  ({e['id']}, strength {e['strength']:.2f}) {e['text']}")
    return "\n".join(lines) or "(no evidence above the floor)"


# ---------------------------------------------------------------------------
# Trial phases (LLM)
# ---------------------------------------------------------------------------

def _ask_json(provider, prompt: str, model: str | None = None):
    _, _, extract_json = _bootstrap()
    reply = provider.complete(prompt, model=model)
    data, err = extract_json(reply)
    if err is not None:
        reply = provider.complete(
            prompt + "\n\nYour previous reply was not parseable JSON. Reply with ONLY the JSON.",
            model=model)
        data, err = extract_json(reply)
    return data, err


ADVOCATE_PROMPT = """You are the {role} advocate in a structured epistemic tribunal.

QUESTION ON TRIAL: {question}

You argue the {pole} pole of this question, exactly as worded. Build the
STRONGEST honest case — a steelman, not rhetoric. Rules:
- Every argument must cite marshaled evidence by id (e.g. clm_000123).
  Arguments without citations are struck.
- Acknowledge the strongest opposing evidence where a serious advocate must.
- 3 to 6 arguments, each one clear predication.

MARSHALED EVIDENCE:
{evidence}

Reply with ONLY JSON:
{{"arguments": [{{"point": "...", "cites": ["clm_..."]}}],
  "key_premises": ["the load-bearing assumptions of your case"]}}"""


CROSS_PROMPT = """You are the {role} advocate in an epistemic tribunal, now in
cross-examination. The opposing brief is below. Attack its 2-3 weakest
LOAD-BEARING premises — not typos, not tone; the premises whose failure
collapses the case. Cite marshaled evidence where it grounds the attack.

QUESTION: {question}

OPPOSING BRIEF:
{brief}

MARSHALED EVIDENCE:
{evidence}

Reply with ONLY JSON:
{{"attacks": [{{"target_premise": "...", "attack": "...", "cites": ["clm_..."]}}]}}"""


JUDGE_PROMPT = """You are one judge on a multi-model bench in an epistemic
tribunal. You have both briefs and the cross-examinations. Judge honestly and
independently — your credence interval will be recorded alongside dissents.

QUESTION: {question}
(Credence = probability the AFFIRMATIVE pole, as worded, is true.)

PRO BRIEF: {pro}
CONTRA BRIEF: {contra}
CROSS ON PRO: {cross_pro}
CROSS ON CONTRA: {cross_contra}

MARSHALED EVIDENCE:
{evidence}

Rules: an INTERVAL, not a point — width reflects your genuine uncertainty.
Cruxes are the sub-questions whose resolution would most move your credence.
Reply with ONLY JSON:
{{"credence": [lo, hi],
  "cruxes": ["..."],
  "would_change_my_mind": ["concrete observations that would move you"],
  "rationale": "2-3 sentences"}}"""


def try_question(repo_root: Path, question_id: str, run_id: str,
                 providers=None, on_progress=None) -> dict[str, Any]:
    """Run the full trial. providers: optional injected list (tests)."""
    progress = on_progress or (lambda m: print(f"  … {m}", flush=True))
    m = marshal(repo_root, question_id)
    q_text = m["question"]["text"]
    evidence = _evidence_block(m["buckets"])

    if providers is None:
        judges, notes = bench_providers()
    else:
        judges, notes = providers, []
    if not judges:
        raise SystemExit("no judges available (claude CLI missing and no github token)")
    # advocates use the first claude-family judge's transport (cheap + good)
    adv = judges[0]

    progress("PRO advocate drafting")
    pro, err1 = _ask_json(adv["provider"], ADVOCATE_PROMPT.format(
        role="PRO", pole="affirmative", question=q_text, evidence=evidence),
        model=adv["model"])
    progress("CONTRA advocate drafting")
    contra, err2 = _ask_json(adv["provider"], ADVOCATE_PROMPT.format(
        role="CONTRA", pole="negative", question=q_text, evidence=evidence),
        model=adv["model"])
    if err1 or err2 or not isinstance(pro, dict) or not isinstance(contra, dict):
        raise SystemExit(f"advocate phase failed: {err1 or err2}")

    progress("cross-examination")
    cross_contra, _ = _ask_json(adv["provider"], CROSS_PROMPT.format(
        role="PRO", question=q_text, brief=json.dumps(contra, ensure_ascii=False),
        evidence=evidence), model=adv["model"])
    cross_pro, _ = _ask_json(adv["provider"], CROSS_PROMPT.format(
        role="CONTRA", question=q_text, brief=json.dumps(pro, ensure_ascii=False),
        evidence=evidence), model=adv["model"])

    rulings = []
    for j in judges:
        progress(f"judge {j['judge']} deliberating")
        ruling, jerr = _ask_json(j["provider"], JUDGE_PROMPT.format(
            question=q_text,
            pro=json.dumps(pro, ensure_ascii=False),
            contra=json.dumps(contra, ensure_ascii=False),
            cross_pro=json.dumps(cross_pro or {}, ensure_ascii=False),
            cross_contra=json.dumps(cross_contra or {}, ensure_ascii=False),
            evidence=evidence), model=j["model"])
        if jerr or not isinstance(ruling, dict) or "credence" not in ruling:
            rulings.append({"judge": j["judge"], "model": j["model"],
                            "error": jerr or "malformed ruling"})
            continue
        try:
            lo, hi = float(ruling["credence"][0]), float(ruling["credence"][1])
            if lo > hi:
                lo, hi = hi, lo
        except Exception:
            rulings.append({"judge": j["judge"], "model": j["model"],
                            "error": "unparseable credence"})
            continue
        rulings.append({
            "judge": j["judge"], "model": j["model"],
            "credence": [round(max(0.0, lo), 3), round(min(1.0, hi), 3)],
            "cruxes": [str(c) for c in ruling.get("cruxes", [])][:6],
            "would_change_my_mind": [str(c) for c in ruling.get("would_change_my_mind", [])][:6],
            "rationale": str(ruling.get("rationale", ""))[:600],
        })

    good = [r for r in rulings if "credence" in r]
    if not good:
        raise SystemExit("bench failed: no valid rulings")
    from statistics import median
    agg = [round(median(r["credence"][0] for r in good), 3),
           round(median(r["credence"][1] for r in good), 3)]
    spread = max(r["credence"][1] for r in good) - min(r["credence"][0] for r in good)

    cited: set[str] = set()
    for brief in (pro, contra):
        for a in brief.get("arguments", []):
            cited.update(a.get("cites", []))

    verdict = {
        "id": store.next_verdict_id(repo_root),
        "question_id": question_id,
        "question_text": q_text,
        "credence": agg,
        "bench_spread": round(spread, 3),
        "rulings": rulings,
        "briefs": {"pro": pro, "contra": contra,
                   "cross_on_pro": cross_pro, "cross_on_contra": cross_contra},
        "evidence_cited": sorted(cited),
        "evidence_watermark": m["evidence_watermark"],
        "bench_notes": notes,
        "run_id": run_id,
        "status": "standing",
    }
    store.append_verdict(repo_root, verdict)
    return verdict


# ---------------------------------------------------------------------------
# Docket
# ---------------------------------------------------------------------------

def docket(repo_root: Path) -> list[dict[str, Any]]:
    questions = store.load_live_questions(repo_root)
    verdicts = store.load_all_verdicts(repo_root)
    standing: dict[str, dict[str, Any]] = {}
    for v in verdicts:
        if v.get("status") == "standing":
            standing[v["question_id"]] = v

    attach_counts: dict[str, int] = {}
    path = repo_root / "data" / "attachments.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    qid = json.loads(line).get("question_id")
                    attach_counts[qid] = attach_counts.get(qid, 0) + 1

    rows = []
    for q in questions:
        if q.get("status") != "contested" and q["id"] not in standing:
            continue
        v = standing.get(q["id"])
        entry = {
            "question_id": q["id"],
            "question": q["text"],
            "status": q.get("status"),
            "attachments": attach_counts.get(q["id"], 0),
        }
        if v is None:
            entry["tribunal"] = "UNTRIED"
        elif attach_counts.get(q["id"], 0) > v.get("evidence_watermark", 0):
            entry["tribunal"] = "RETRIAL (new evidence since verdict)"
            entry["verdict"] = {"id": v["id"], "credence": v["credence"]}
        else:
            entry["tribunal"] = "STANDING"
            entry["verdict"] = {"id": v["id"], "credence": v["credence"]}
        rows.append(entry)
    return rows
