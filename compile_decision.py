"""Compile data/signals.jsonl into data/decision.json for the /decision page.

Runs the vendored DirectionPrioritizer over the AI-direction taxonomy for several
beneficiary scopes, and emits a site-ready artifact with rankings, factor
breakdowns, cruxes, evidence traceability, and the honesty framing.

Honesty post-processing: a direction with zero corpus evidence is relabeled
"unevaluated" rather than the engine's default status, so the page never implies
the corpus judged something it was silent on.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from decision import (
    BeneficiaryScope,
    DirectionPrioritizer,
    iter_claim_signals_jsonl,
)
from decision_taxonomy import AI_DIRECTIONS, FACTOR_REF, ai_decision_question

REPO = Path(__file__).resolve().parent
SCOPES = [
    BeneficiaryScope.HUMANITY,
    BeneficiaryScope.FUTURE_GENERATIONS,
    BeneficiaryScope.ALL_SENTIENT_LIFE,
]
DISCLAIMER = (
    "This ranks directions by highest expected return UNDER a specific beneficiary "
    "scope, the ITN+ value model, and this corpus of AI discourse — not objectively. "
    "Signals are machine-judged from mined claims; treat them as a structured lens, "
    "not a verdict."
)


def _signals(path: Path) -> list:
    return list(iter_claim_signals_jsonl(path)) if path.exists() else []


def _claim_texts() -> dict[str, str]:
    out: dict[str, str] = {}
    path = REPO / "data" / "claims.jsonl"
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                c = json.loads(line)
                out[c["id"]] = c.get("text", "")
    return out


def compile_decision(
    signals_path: str | Path = REPO / "data" / "signals.jsonl",
    out_path: str | Path = REPO / "data" / "decision.json",
    generated_run: str = "manual",
) -> dict:
    signals = _signals(Path(signals_path))
    prioritizer = DirectionPrioritizer()
    texts = _claim_texts()

    # Claim ids backing each direction (for traceability to /live), source-deduped,
    # plus a few sample backing-claim texts (highest-strength first) for the page.
    claims_by_direction: dict[str, set[str]] = defaultdict(set)
    best_signal: dict[tuple[str, str], float] = {}
    for s in signals:
        if s.proposition_id:
            claims_by_direction[s.direction_id].add(s.proposition_id)
            key = (s.direction_id, s.proposition_id)
            best_signal[key] = max(best_signal.get(key, 0.0), s.strength * s.relevance)
    samples: dict[str, list[dict]] = {}
    for did in claims_by_direction:
        ranked_cids = sorted(
            claims_by_direction[did], key=lambda c: best_signal.get((did, c), 0.0), reverse=True
        )
        samples[did] = [
            {"id": cid, "text": texts.get(cid, "")[:200]} for cid in ranked_cids[:4]
        ]

    scopes_out: dict[str, dict] = {}
    for scope in SCOPES:
        question = ai_decision_question(scope)
        result = prioritizer.evaluate(question, signals)
        ranked = []
        for ev in result.ranked_directions:
            status = ev.status.value
            if ev.evidence_count == 0:
                status = "unevaluated"
            ranked.append(
                {
                    "direction_id": ev.direction_id,
                    "name": ev.direction_name,
                    "status": status,
                    "score": round(ev.normalized_score, 3),
                    "relative_return": round(ev.expected_return_index, 3),
                    "robust": round(ev.robust_score, 3),
                    "evidence_count": ev.evidence_count,
                    "dependency_groups": ev.dependency_group_count,
                    "factors": [
                        {
                            "factor_id": fa.factor_id,
                            "name": fa.factor_name,
                            "score": round(fa.adjusted_score, 3),
                            "evidence_count": fa.evidence_count,
                            "missing": fa.missing,
                        }
                        for fa in ev.factor_scores
                    ],
                }
            )
        # Honesty: when the top directions are within noise of each other, the
        # corpus does not decisively separate them — say so instead of crowning
        # a "winner" that an 0.007 gap produced.
        evaluated = [r for r in ranked if r["evidence_count"] > 0]
        leader_gap = (
            round(evaluated[0]["score"] - evaluated[1]["score"], 3)
            if len(evaluated) >= 2
            else None
        )
        clear_leader = leader_gap is not None and leader_gap >= 0.02
        scopes_out[scope.value] = {
            "recommended": result.recommended_direction_id,
            "clear_leader": clear_leader,
            "leader_gap": leader_gap,
            "ranked": ranked,
            "cruxes": [
                {
                    "description": c.description,
                    "value_of_information": round(c.value_of_information, 4),
                    "investigation": c.investigation,
                }
                for c in result.cruxes[:6]
            ],
            "warnings": result.warnings,
        }

    artifact = {
        "generated_run": generated_run,
        "disclaimer": DISCLAIMER,
        "corpus_note": (
            "Directions are AI sub-areas because the source corpus is AI discourse; "
            "fit/career-capital factors are mostly silent here by design."
        ),
        "signal_count": len(signals),
        "scopes": [s.value for s in SCOPES],
        "directions": [
            {"id": d.id, "name": d.name, "description": d.description} for d in AI_DIRECTIONS
        ],
        "factors": [{"id": fid, "note": note} for fid, note in FACTOR_REF],
        "claims_by_direction": {k: sorted(v) for k, v in claims_by_direction.items()},
        "samples": samples,
        "by_scope": scopes_out,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=1), encoding="utf-8")
    return {
        "signals": len(signals),
        "scopes": len(scopes_out),
        "directions_with_evidence": sum(1 for d in claims_by_direction if claims_by_direction[d]),
        "out": str(out),
    }


if __name__ == "__main__":
    import sys

    print(json.dumps(compile_decision(*sys.argv[1:])))
