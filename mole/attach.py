"""
Attach-candidate generation: score claims against the question layer.

Deterministic and LLM-free (mole-grade). For each claim, the best-matching
question above ATTACH_FLOOR becomes an `attach` task for workers to judge
(stance/strength per WORKER.md). Scoring = IDF-weighted verbatim phrase hits
on question keywords + TF-IDF cosine between claim text and question terms.

Calibrated on the attach_v0 experiment (2026-07-02, 5,885 claims x 30
questions): the best-score histogram is bimodal with a near-empty valley at
0.2-0.3, so the floor sits in a natural gap. Judged strict precision at this
floor was ~24-29% — these are CANDIDATES for worker judgment, never
auto-attachments.
"""

from __future__ import annotations

import collections
import math
import re
from typing import Any

ATTACH_FLOOR = 0.28
PHRASE_W = 1.0
TOKEN_W = 0.6
_PHRASE_CAP = 1.5

_STOP = set(
    "a an and are as at be been but by can could do for from had has have how "
    "i if in is it its more most no not of on or our so than that the their "
    "there these they this to was we what when which while who will with "
    "would you your".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'&\-]*")


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


def _norm_phrase(phrase: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", phrase.lower()) + " "


def build_index(claim_texts: list[str], questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Corpus statistics (token/phrase IDF) plus per-question vectors."""
    # Smoothed IDF: log(1 + n/(1+f)) is always positive, so the scorer stays
    # well-defined and discriminative on tiny corpora (tests, cold-start
    # repos); at corpus scale it is indistinguishable from plain log(n/(1+f)).
    n_docs = max(len(claim_texts), 1)
    df: collections.Counter = collections.Counter()
    normed = [_norm_phrase(t) for t in claim_texts]
    for text in claim_texts:
        df.update(set(_tokens(text)))
    default_idf = math.log(1 + n_docs)
    idf = {t: math.log(1 + n_docs / (1 + f)) for t, f in df.items()}

    phrases = sorted({kw.lower() for q in questions for kw in q.get("keywords", [])})
    phrase_df = dict.fromkeys(phrases, 0)
    for text in normed:
        for p in phrases:
            if _norm_phrase(p) in text:
                phrase_df[p] += 1
    phrase_idf = {p: math.log(1 + n_docs / (1 + f)) for p, f in phrase_df.items()}

    q_vecs: dict[str, dict[str, float]] = {}
    for q in questions:
        vec: collections.Counter = collections.Counter()
        for kw in q.get("keywords", []):
            for t in _tokens(kw):
                vec[t] += 3.0
        for t in _tokens(q["text"]):
            vec[t] += 1.0
        weighted = {t: w * idf.get(t, default_idf) for t, w in vec.items()}
        norm = math.sqrt(sum(w * w for w in weighted.values())) or 1.0
        q_vecs[q["id"]] = {t: w / norm for t, w in weighted.items()}

    return {
        "idf": idf,
        "phrase_idf": phrase_idf,
        "max_phrase_idf": default_idf,
        "q_vecs": q_vecs,
        "questions": questions,
    }


def best_question(claim_text: str, index: dict[str, Any]) -> tuple[str, float] | None:
    """Best (question_id, score) for a claim, or None when nothing scores."""
    toks = _tokens(claim_text)
    if not toks:
        return None
    tf = collections.Counter(toks)
    weighted = {t: (1 + math.log(f)) * index["idf"].get(t, 0.0) for t, f in tf.items()}
    norm = math.sqrt(sum(w * w for w in weighted.values())) or 1.0
    cvec = {t: w / norm for t, w in weighted.items()}
    normed_text = _norm_phrase(claim_text)

    best: tuple[float, str] | None = None
    for q in index["questions"]:
        qid = q["id"]
        qvec = index["q_vecs"][qid]
        cos = sum(w * qvec.get(t, 0.0) for t, w in cvec.items())
        ph = 0.0
        for raw_kw in q.get("keywords", []):
            if _norm_phrase(raw_kw) in normed_text:
                ph += index["phrase_idf"][raw_kw.lower()] / index["max_phrase_idf"]
        score = TOKEN_W * cos + PHRASE_W * min(ph, _PHRASE_CAP)
        if score > 0 and (best is None or score > best[0]):
            best = (score, qid)
    if best is None:
        return None
    return best[1], round(best[0], 4)


def candidates(
    claims: list[dict[str, Any]],
    index: dict[str, Any],
    floor: float = ATTACH_FLOOR,
) -> list[tuple[str, str, float]]:
    """(claim_id, question_id, sim) for each claim whose best score >= floor."""
    out: list[tuple[str, str, float]] = []
    for claim in claims:
        hit = best_question(claim["text"], index)
        if hit is not None and hit[1] >= floor:
            out.append((claim["id"], hit[0], hit[1]))
    return out


# ---------------------------------------------------------------------------
# Hybrid scoring for theses
# ---------------------------------------------------------------------------
# Thesis-level statements are conceptual: they routinely restate a question's
# subject matter without its jargon tokens, which is exactly where lexical
# TF-IDF scores zero (measured 2026-07: normative claims at 0.01-0.04 against
# a 0.28 floor). For theses we therefore also score embedding cosine via the
# extractor's optional model2vec backend and take a candidate on EITHER
# signal. Degrades to lexical-only when embeddings are unavailable.

EMB_FLOOR = 0.45


def candidates_hybrid(
    rows: list[dict[str, Any]],
    index: dict[str, Any],
    questions: list[dict[str, Any]],
    lex_floor: float = ATTACH_FLOOR,
    emb_floor: float = EMB_FLOOR,
) -> list[tuple[str, str, float]]:
    """(row_id, question_id, sim) where sim = max(lexical, embedding-cosine)
    and at least one of the two clears its floor."""
    out: list[tuple[str, str, float]] = []
    if not rows or not questions:
        return out

    emb_by_q: dict[str, list[float]] | None = None
    row_vecs: list[list[float]] | None = None
    try:
        from extractor import embed_texts

        q_texts = [f"{q['text']} {' '.join(q.get('keywords', []))}" for q in questions]
        q_vecs = embed_texts(q_texts)
        r_vecs = embed_texts([r["text"] for r in rows])
        if q_vecs is not None and r_vecs is not None:
            emb_by_q = {q["id"]: v for q, v in zip(questions, q_vecs)}
            row_vecs = r_vecs
    except Exception:
        emb_by_q = None

    import math

    def _cos(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    for i, row in enumerate(rows):
        best_id: str | None = None
        best_sim = 0.0
        lex = best_question(row["text"], index)
        if lex is not None and lex[1] >= lex_floor:
            best_id, best_sim = lex[0], lex[1]
        if emb_by_q is not None and row_vecs is not None:
            for q in questions:
                c = _cos(row_vecs[i], emb_by_q[q["id"]])
                if c >= emb_floor and c > best_sim:
                    best_id, best_sim = q["id"], round(c, 4)
        if best_id is not None:
            out.append((row["id"], best_id, best_sim))
    return out
