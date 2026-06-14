"""Assemble full ClaimSignal records from compact LLM judgments + claim provenance.

A worker fleet judges each claim into 0..N compact records:
  {claim_id, direction_id, factor_id, polarity, strength, relevance, credence}
This joins them with data/claims.jsonl to fill provenance and the derived
evidence fields, writing data/signals.jsonl (full ClaimSignal JSON).

dependency_key is set to the claim's source_id so the prioritizer's dependency
cap treats many claims from one article/source as one correlated group.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from decision import ClaimPolarity, ClaimSignal
from decision_taxonomy import DIRECTION_IDS, FACTOR_IDS

REPO = Path(__file__).resolve().parent


def _claims_by_id() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with (REPO / "data" / "claims.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            out[c["id"]] = c
    return out


def build(judgments_path: str | Path, out_path: str | Path = REPO / "data" / "signals.jsonl") -> dict:
    claims = _claims_by_id()
    seen: set[tuple[str, str, str]] = set()
    written = 0
    skipped_unknown = 0
    skipped_dupe = 0
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with Path(judgments_path).open() as fh, out.open("w", encoding="utf-8") as w:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            cid = j.get("claim_id")
            did = j.get("direction_id")
            fid = j.get("factor_id")
            claim = claims.get(cid)
            if claim is None or did not in DIRECTION_IDS or fid not in FACTOR_IDS:
                skipped_unknown += 1
                continue
            key = (cid, did, fid)
            if key in seen:
                skipped_dupe += 1
                continue
            seen.add(key)
            support = claim.get("support_in_text")
            support = 0.5 if support is None else float(support)
            polarity = (
                ClaimPolarity.WEAKENS
                if str(j.get("polarity", "supports")).lower().startswith("weak")
                else ClaimPolarity.SUPPORTS
            )
            source_id = claim.get("source_id")
            signal = ClaimSignal(
                id=f"sig_{cid}_{did}_{fid}",
                direction_id=did,
                factor_id=fid,
                polarity=polarity,
                credence=_clamp(j.get("credence", 0.6)),
                strength=_clamp(j.get("strength", 0.5)),
                relevance=_clamp(j.get("relevance", 0.8)),
                # Extraction confidence and evidence quality come from how well the
                # claim was supported in its own source, not from the judge.
                extraction_confidence=_clamp(0.35 + 0.5 * support),
                evidence_quality=_clamp(0.3 + 0.5 * support),
                dependency_key=source_id or cid,
                source_id=source_id,
                proposition_id=cid,
                text=claim.get("text", "")[:300],
                metadata={"claim_type": claim.get("type"), "run_id": claim.get("run_id")},
            )
            w.write(json.dumps(signal.model_dump(mode="json"), ensure_ascii=False) + "\n")
            written += 1
    return {
        "signals_written": written,
        "skipped_unknown": skipped_unknown,
        "skipped_duplicate": skipped_dupe,
        "out": str(out),
    }


def _clamp(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.5
    return min(1.0, max(0.0, v))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: build_signals.py <judgments.jsonl> [out.jsonl]")
    args = sys.argv[1:]
    result = build(*args)
    print(json.dumps(result))
