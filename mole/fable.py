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
import threading
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from mole import store

DEFAULT_ENGINE_SRC = "/Users/mo/Desktop/Prj.nosync/programming/episteme-fable/src"
# Serialises the write phase (id allocation -> task flips) so a drain can run
# several sources' LLM work concurrently; every store write is read-modify-write.
_WRITE_LOCK = threading.Lock()
_INDEX_CACHE: dict[str, Any] = {"n": -1, "index": None}
_INDEX_STALE_AFTER = 250  # rebuild the attach index when this many live claims were added
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
        from episteme_fable.providers import make_provider
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            f"episteme-fable engine not importable from {src!r} "
            f"(set EPISTEME_FABLE_SRC): {exc}"
        ) from exc
    return analyze, make_provider


# ---------------------------------------------------------------------------
# Source refetch
# ---------------------------------------------------------------------------

_ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/([^\s?#]+)")
# arXiv's export API asks for one request every 3 s per client and answers
# bursts with HTTP 429; a drain with N workers must share one clock.
_ARXIV_MIN_INTERVAL = float(os.environ.get("ARXIV_MIN_INTERVAL", "3.0"))
_ARXIV_CLOCK = {"next": 0.0, "banned_until": 0.0}
_ARXIV_LOCK = threading.Lock()
_ARXIV_BAN_SECONDS = float(os.environ.get("ARXIV_BAN_SECONDS", "900"))


def _arxiv_banned() -> bool:
    import time as _time
    return _time.monotonic() < _ARXIV_CLOCK["banned_until"]


def _arxiv_fetch(api_url: str) -> str:
    """Rate-limited, 429-tolerant GET of an arXiv API query."""
    import time as _time
    import urllib.error
    from mole import feeds as feeds_mod

    if _arxiv_banned():
        raise urllib.error.HTTPError(api_url, 429, "arxiv client banned (local flag)", {}, None)
    for attempt in range(2):
        with _ARXIV_LOCK:
            wait = _ARXIV_CLOCK["next"] - _time.monotonic()
            if wait > 0:
                _time.sleep(wait)
            _ARXIV_CLOCK["next"] = _time.monotonic() + _ARXIV_MIN_INTERVAL
        try:
            return feeds_mod._fetch_text(api_url, headers={"User-Agent": feeds_mod.USER_AGENT})
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 1:
                with _ARXIV_LOCK:  # back everyone off, not just this thread
                    _ARXIV_CLOCK["next"] = _time.monotonic() + 10.0
                continue
            if e.code == 429:
                # Two 429s in a row: arXiv has banned this client for a while.
                # Flag it so every worker goes straight to the fallbacks
                # instead of burning the back-off per source.
                with _ARXIV_LOCK:
                    _ARXIV_CLOCK["banned_until"] = _time.monotonic() + _ARXIV_BAN_SECONDS
            raise
        except OSError:
            if attempt < 3:
                _time.sleep(5.0 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")
_LW_POST_RE = re.compile(r"https?://(www\.)?(lesswrong\.com|alignmentforum\.org)/posts/([A-Za-z0-9]+)")


def _lesswrong_post_html(host: str, post_id: str) -> tuple[str, str]:
    """(title, htmlBody) of a LessWrong / Alignment Forum post via the forum's
    GraphQL API. The HTML page itself is a Next.js bundle (hundreds of KB of
    JS + escaped HTML); stripping tags from it yields script text as 'prose'
    (2026-10-30: two AF posts produced 1,021 junk claims that way)."""
    import urllib.request
    from mole import feeds as feeds_mod

    query = ('{post(input:{selector:{_id:"%s"}}){result{title htmlBody}}}' % post_id)
    body = json.dumps({"query": query}).encode("utf-8")
    # Alignment Forum posts live in the LessWrong database under the same id;
    # alignmentforum.org/graphql answers with a 308 redirect, so always ask LW.
    req = urllib.request.Request(
        "https://www.lesswrong.com/graphql", data=body, method="POST",
        headers={"Content-Type": "application/json", "User-Agent": feeds_mod.USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    result = ((data.get("data") or {}).get("post") or {}).get("result") or {}
    if not result.get("htmlBody"):
        raise RuntimeError(f"lesswrong graphql returned no htmlBody for {post_id}: {str(data)[:200]}")
    return result.get("title") or "", result["htmlBody"]


def _openalex_arxiv(arxiv_id: str) -> tuple[str, str] | None:
    """(title, abstract) from OpenAlex for an arXiv id, or None. Keyless,
    polite pool via mailto; abstracts come as an inverted index."""
    import urllib.error
    import urllib.request
    from mole import feeds as feeds_mod

    bare = re.sub(r"v\d+$", "", arxiv_id)
    url = ("https://api.openalex.org/works/https://doi.org/10.48550/arXiv." + bare
           + "?mailto=" + urllib.parse.quote(os.environ.get("OPENALEX_MAILTO", "claimbase@example.org")))
    req = urllib.request.Request(url, headers={"User-Agent": feeds_mod.USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, OSError, json.JSONDecodeError):
        return None
    inv = data.get("abstract_inverted_index") or {}
    if not inv:
        return None
    positions: list[tuple[int, str]] = [(p, w) for w, ps in inv.items() for p in ps]
    abstract = " ".join(w for _, w in sorted(positions))
    return (data.get("title") or "", abstract)


def _arxiv_abs_page(url: str) -> tuple[str, str] | None:
    """(title, abstract) scraped from the arxiv.org/abs page meta tags."""
    import urllib.error
    from mole import feeds as feeds_mod

    try:
        html = _arxiv_fetch(url.replace("http://", "https://"))
    except (urllib.error.HTTPError, OSError, RuntimeError):
        return None
    t = re.search(r'<meta name="citation_title" content="([^"]*)"', html)
    a = re.search(r'<meta property="og:description" content="([^"]*)"', html, re.S)
    if not (t and a):
        return None
    return (feeds_mod._strip_html(t.group(1)), feeds_mod._strip_html(a.group(1)))


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
        title = abstract = None
        try:
            xml_text = _arxiv_fetch(api)
            root = ET.fromstring(xml_text)
            entry = next((n for n in root.iter() if feeds_mod._localname(n.tag) == "entry"), None)
            if entry is None:
                raise RuntimeError(f"arxiv API returned no entry for {url}")
            title = feeds_mod._first_child_text(entry, ["title"], default="")
            abstract = feeds_mod._strip_html(str(feeds_mod._first_child_text(entry, ["summary"], default="")))
        except (urllib.error.HTTPError, OSError, RuntimeError, ET.ParseError) as exc:
            # export.arxiv.org bans a client for a while after a burst (HTTP
            # 429 on every call, 2026-10-30). Fall back to OpenAlex, then the
            # abs page, before giving up on the source.
            alt = _openalex_arxiv(m.group(1)) or _arxiv_abs_page(url)
            if alt is None:
                raise RuntimeError(f"arxiv unavailable ({exc}); no fallback for {url}") from exc
            title, abstract = alt
        text = f"{title}\n\n{abstract}".strip()
    elif _LW_POST_RE.search(url):
        lw = _LW_POST_RE.search(url)
        title, html_body = _lesswrong_post_html(lw.group(2), lw.group(3))
        text = f"{title}\n\n{feeds_mod._strip_html(html_body)}".strip()
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

    analyze, make_provider = _bootstrap_engine()
    provider = make_provider(model)  # EPF_PROVIDER: claude (default) | github
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

    with _WRITE_LOCK:
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
            if (_INDEX_CACHE["index"] is None
                    or len(live_texts) - _INDEX_CACHE["n"] >= _INDEX_STALE_AFTER):
                _INDEX_CACHE["index"] = attach_mod.build_index(live_texts, questions)
                _INDEX_CACHE["n"] = len(live_texts)
            index = _INDEX_CACHE["index"]
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
