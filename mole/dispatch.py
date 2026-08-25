"""
Scale-up extraction dispatcher.

Fans claim extraction over MANY documents (candidates pulled from the free
OpenAlex API) across N parallel LLM endpoints, with rate governance and
importance-first ordering, writing results in claimbase's EXISTING formats
(data/sources.jsonl, data/claims.jsonl, queue/pending.jsonl) via mole.store
and mole.attach -- the same modules mole/fable.py uses. This module does not
modify any of those, feeds.yaml, or the cron workflows; see ../DISPATCH.md.

Subcommands:
  plan    -- query OpenAlex, rank by cited_by_count (importance prior), keep
             only OA items, write dispatch/worklist.jsonl. No fulltext fetch.
  run     -- process the top-K unprocessed worklist entries with N worker
             threads, per-provider token-bucket rate limiting, and a single
             writer thread for all file I/O (crash-safe, append-only).
  status  -- progress report + ETA at the current rate.

v0 scope: content = OpenAlex title + abstract (reconstructed from
abstract_inverted_index already returned by `plan`'s query -- no extra
fetch). Real fulltext PDF/HTML extraction is out of scope for v0; records are
labeled content_kind="abstract"|"fulltext"|"none" so a later version can add
a fulltext fetcher without touching the record shape.

Providers are pluggable: claude-cli (subprocess `claude -p ... --output-format
json`) and openrouter (stdlib urllib POST). Both speak the same minimal JSON
claim contract: [{"text":..., "kind":..., "claim_type":..., "quote":...}, ...].

stdlib-only except PyYAML, which is already a claimbase dependency
(pyproject.toml: PyYAML>=6.0) -- used only for the human-edited config file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue as queue_mod
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from mole import store

try:
    import yaml  # PyYAML -- already a claimbase dependency (pyproject.toml)
except ImportError:  # pragma: no cover - environment issue
    yaml = None


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = MODULE_DIR.parent
DISPATCH_SUBDIR = "dispatch"
DEFAULT_CONFIG_NAME = "dispatch_config.yaml"

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OPENROUTER_URL_DEFAULT = "https://openrouter.ai/api/v1/chat/completions"

QUOTE_CAP = 300
SUPPORT_IN_TEXT_NEUTRAL = 0.5  # same convention as mole/fable.py: v0 does not
                               # localize claims within source text; epistemics
                               # (when present) live in kind/claim_type instead.

_VALID_CLAIM_TYPES = {
    "empirical", "statistical", "causal", "predictive", "normative",
    "definitional", "descriptive", "historical", "other",
}
_VALID_KINDS = {"assertion", "question", "goal", "plan", "inference", "definition"}


def dispatch_dir(root: Path) -> Path:
    return root / DISPATCH_SUBDIR


def worklist_path(repo_root: Path) -> Path:
    # Candidate discovery (`plan`) is not data-dir-scoped: the worklist is
    # shared regardless of where a `run` later writes extracted records.
    return dispatch_dir(repo_root) / "worklist.jsonl"


def done_path(data_root: Path) -> Path:
    # Scoped to data_root (== --data-dir), NOT repo_root: a sandboxed `run`
    # (--data-dir dispatch/sandbox_data) must get its OWN done-marker file,
    # so sandbox verification runs can never cause a later real run to skip
    # items as "already done". Default data_root is repo_root, so real runs
    # keep using dispatch/done.jsonl at the repo root as before.
    return dispatch_dir(data_root) / "done.jsonl"


def rejections_path(data_root: Path) -> Path:
    return dispatch_dir(data_root) / "rejections.jsonl"


# ---------------------------------------------------------------------------
# JSONL helpers (mirrors mole/store.py's house style: single-writer append,
# temp-then-replace for full rewrites; dispatch's own bookkeeping files live
# under dispatch/, never touching data/ or queue/ directly -- that only
# happens through mole.store, imported unmodified above)
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _append_jsonl_line(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    """write-temp-then-rename: used for worklist.jsonl, which `plan` fully
    regenerates each run (never partially written even if plan is killed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "query": {
        "search": "",
        "concepts": [],
        "venues": [],
        "from_publication_date": None,
        "max_works": 50,
    },
    "openalex": {
        # OpenAlex asks for a real contact email for their "polite pool"
        # (faster, more reliable responses). Fill this in before running
        # `plan` for real -- we deliberately do NOT default this to any
        # actual person's address.
        "mailto": "you@example.com",
        "per_page": 100,
    },
    "providers": {
        "claude-cli": {
            "model": "haiku",
            "timeout_s": 120,
            "bare": False,
            "requests_per_min": 20,
            "tokens_per_min": 40000,
        },
        "openrouter": {
            "model": "meta-llama/llama-3.1-8b-instruct",
            "api_key_env": "OPENROUTER_API_KEY",
            "base_url": OPENROUTER_URL_DEFAULT,
            "timeout_s": 60,
            "requests_per_min": 20,
            "tokens_per_min": 60000,
        },
    },
    "run": {
        "default_provider": "claude-cli",
        "default_workers": 4,
        "default_limit": 20,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | None) -> dict[str, Any]:
    """Load dispatch_config.yaml (or .json). Missing file -> defaults only,
    so `plan`/`run` are still runnable (e.g. in tests) without a config."""
    if path is None or not path.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError(
                f"{path} is YAML but PyYAML is not importable; use a .json "
                "config or install PyYAML (claimbase already depends on it)"
            )
        user_cfg = yaml.safe_load(text) or {}
    else:
        user_cfg = json.loads(text) if text.strip() else {}
    return _deep_merge(DEFAULT_CONFIG, user_cfg)


# ---------------------------------------------------------------------------
# OpenAlex: plan
# ---------------------------------------------------------------------------

def _reconstruct_abstract(inv_index: dict[str, list[int]] | None) -> str:
    """OpenAlex ships abstracts as an inverted index (word -> positions) to
    dodge publisher copyright on raw abstract text. Reassembling it from data
    already returned by the SAME query `plan` issues for ranking is not an
    extra fetch -- it is why `plan` requests abstract_inverted_index in its
    `select` param up front."""
    if not inv_index:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inv_index.items():
        for i in idxs:
            positions[i] = word
    if not positions:
        return ""
    return " ".join(positions[i] for i in sorted(positions))


def _build_openalex_filter(query_cfg: dict[str, Any]) -> str:
    parts = []
    if query_cfg.get("concepts"):
        parts.append("concepts.id:" + "|".join(query_cfg["concepts"]))
    if query_cfg.get("venues"):
        parts.append("primary_location.source.id:" + "|".join(query_cfg["venues"]))
    if query_cfg.get("from_publication_date"):
        parts.append("from_publication_date:" + query_cfg["from_publication_date"])
    return ",".join(parts)


_OPENALEX_SELECT = (
    "id,doi,display_name,cited_by_count,open_access,best_oa_location,"
    "primary_location,abstract_inverted_index,publication_date,authorships"
)


def fetch_openalex_works(
    query_cfg: dict[str, Any],
    *,
    mailto: str,
    per_page: int = 100,
    max_works: int = 50,
    _urlopen=None,
) -> list[dict[str, Any]]:
    """Cursor-paged, server-side sorted (cited_by_count desc) OpenAlex works
    query. No fulltext is fetched here -- only the metadata + abstract needed
    to rank and, later, to run a v0 abstract-only extraction."""
    urlopen = _urlopen or urllib.request.urlopen
    filt = _build_openalex_filter(query_cfg)
    base_params: dict[str, str] = {
        "sort": "cited_by_count:desc",
        "per-page": str(max(1, min(per_page, 200))),
        "mailto": mailto,
        "select": _OPENALEX_SELECT,
    }
    if filt:
        base_params["filter"] = filt
    if query_cfg.get("search"):
        base_params["search"] = query_cfg["search"]

    out: list[dict[str, Any]] = []
    cursor = "*"
    while len(out) < max_works and cursor:
        params = dict(base_params)
        params["cursor"] = cursor
        url = OPENALEX_WORKS_URL + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url, headers={"User-Agent": f"claimbase-dispatch/0.1 (mailto:{mailto})"}
        )
        with urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        results = body.get("results", [])
        if not results:
            break
        out.extend(results)
        cursor = (body.get("meta") or {}).get("next_cursor")
    return out[:max_works]


def _work_to_worklist_row(work: dict[str, Any], priority: int) -> dict[str, Any] | None:
    oa = work.get("open_access") or {}
    best = work.get("best_oa_location") or work.get("primary_location") or {}
    has_oa = bool(oa.get("is_oa")) or bool(work.get("best_oa_location"))
    if not has_oa:
        return None  # "filter to items with an OA fulltext/abstract location"

    raw_id = work.get("id") or ""
    openalex_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
    authors = [
        a.get("author", {}).get("display_name", "")
        for a in work.get("authorships", [])[:3]
        if a.get("author", {}).get("display_name")
    ]
    return {
        "openalex_id": openalex_id,
        "doi": work.get("doi") or "",
        "title": work.get("display_name") or "",
        "cited_by_count": work.get("cited_by_count") or 0,
        "oa_url": best.get("pdf_url") or best.get("landing_page_url") or "",
        "priority": priority,
        # extra fields beyond the minimum spec, filled in from data `plan`
        # already fetched (no second network round-trip needed by `run`):
        "abstract": abstract,
        "content_kind": "abstract" if abstract else "none",
        "publication_date": work.get("publication_date") or "",
        "authors": authors,
    }


def cmd_plan(args: argparse.Namespace, config: dict[str, Any], repo_root: Path) -> None:
    query_cfg = dict(config["query"])
    if args.max_works is not None:
        query_cfg["max_works"] = args.max_works
    if args.search is not None:
        query_cfg["search"] = args.search

    mailto = config["openalex"]["mailto"]
    if mailto == DEFAULT_CONFIG["openalex"]["mailto"]:
        print(
            f"warning: openalex.mailto is still the placeholder "
            f"({mailto!r}); set a real contact address in the config for "
            f"OpenAlex's polite pool before doing large runs",
            file=sys.stderr,
        )

    works = fetch_openalex_works(
        query_cfg,
        mailto=mailto,
        per_page=config["openalex"]["per_page"],
        max_works=query_cfg["max_works"],
    )

    rows: list[dict[str, Any]] = []
    for w in works:
        row = _work_to_worklist_row(w, priority=0)
        if row is not None:
            rows.append(row)
    # already sorted by cited_by_count desc from the API; re-sort defensively
    # (filtering above cannot reorder, but this keeps `plan` correct even if
    # a future query_cfg drops the server-side sort) and assign final rank.
    rows.sort(key=lambda r: r["cited_by_count"], reverse=True)
    for i, row in enumerate(rows, start=1):
        row["priority"] = i

    out_path = worklist_path(repo_root)
    _write_jsonl_atomic(out_path, rows)
    print(
        f"plan: fetched {len(works)} works, kept {len(rows)} with an OA "
        f"location -> {out_path}"
    )


# ---------------------------------------------------------------------------
# Extraction prompt + validation (dispatch's own minimal contract -- distinct
# from episteme-fable's windowed propose_v2 contract, which assumes
# in-document sequential state (glossary/prev-tail) this dispatcher's
# per-document abstract-only extraction does not need)
# ---------------------------------------------------------------------------

def build_prompt(*, title: str, content_kind: str, text: str) -> str:
    kind_note = "the full document text" if content_kind == "fulltext" else "the abstract only (full text was not available/fetched)"
    return f"""You are extracting atomic claims from a research document for a claim graph.

DOCUMENT TITLE: {title}
CONTENT: {kind_note}

TEXT:
{text}

Extract the substantive claims this text makes. Reply with a JSON array ONLY
-- no markdown fences, no commentary before or after. Each element:
{{
  "text": "self-contained rewrite of the claim, understandable without the source",
  "kind": "assertion",              // assertion|question|goal|plan|inference|definition
  "claim_type": "empirical",        // empirical|statistical|causal|predictive|normative|definitional|descriptive|historical|other
  "quote": "verbatim substring of TEXT supporting this claim, <=300 chars"
}}
Return [] if there are no substantive claims. Do not invent claims TEXT does not support.
"""


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def extract_json_array(reply: str) -> tuple[list | None, str | None]:
    reply = reply.strip()
    if reply.startswith("```"):
        reply = re.sub(r"^```[a-zA-Z]*\n?", "", reply)
        reply = re.sub(r"\n?```$", "", reply).strip()
    try:
        data = json.loads(reply)
    except json.JSONDecodeError:
        m = _JSON_ARRAY_RE.search(reply)
        if not m:
            return None, "no JSON array found in reply"
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return None, f"invalid JSON: {e}"
    if not isinstance(data, list):
        return None, "top-level JSON is not an array"
    return data, None


def validate_claim(raw: Any) -> tuple[dict[str, str] | None, str | None]:
    if not isinstance(raw, dict):
        return None, "element is not a JSON object"
    text = str(raw.get("text", "")).strip()
    if not text:
        return None, "empty/missing text"
    quote = str(raw.get("quote", "")).strip()[:QUOTE_CAP]
    kind = raw.get("kind") if raw.get("kind") in _VALID_KINDS else "assertion"
    claim_type = raw.get("claim_type") if raw.get("claim_type") in _VALID_CLAIM_TYPES else "descriptive"
    return {"text": text, "kind": kind, "claim_type": claim_type, "quote": quote}, None


# ---- Ingestion-time junk filter (ISSUES.md: "Extractor pulls MathJax CSS
# fragments as claims", found 2026-08-25, 2,326 junk rows, no fix landed yet
# anywhere in this codebase as of this dispatcher). ----------------------

_JUNK_PATTERNS = [
    re.compile(r"[.#]?[\w-]+\s*\{[^{}]{0,400}:[^{}]{0,400};[^{}]{0,400}\}"),  # CSS rule shape
    re.compile(r"\bmjx-|MathJax|\\displaystyle|\\begin\{|\\end\{|\\frac\{"),   # MathJax
    re.compile(r"\bfunction\s*\(|=>\s*\{|\bvar\s+\w+\s*=|\bconst\s+\w+\s*=|\bdocument\.getElementById"),  # JS
    re.compile(r"<style[\s>]|<script[\s>]|</style>|</script>"),               # raw tags
]

_SYMBOL_RE = re.compile(r"[^A-Za-z0-9\s]")


def is_junk_shape(text: str) -> bool:
    if not text:
        return False
    for pat in _JUNK_PATTERNS:
        if pat.search(text):
            return True
    # generic minified-CSS/JS heuristic: dense punctuation/symbols with no
    # sentence-like structure
    if len(text) >= 24:
        symbol_ratio = len(_SYMBOL_RE.findall(text)) / len(text)
        if symbol_ratio > 0.4:
            return True
    return False


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

class ProviderError(RuntimeError):
    pass


class Provider:
    name = "base"

    def extract(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class ClaudeCLIProvider(Provider):
    """Subprocess `claude -p ... --output-format json`, like mole/fable.py's
    engine ultimately shells out to (mole/fable.py delegates to the sibling
    episteme-fable package's ClaudeCLIProvider; this dispatcher is
    self-contained/stdlib-only, so it drives the `claude` CLI directly)."""

    name = "claude-cli"

    def __init__(self, *, model: str = "haiku", timeout_s: int = 120, bare: bool = False):
        self.model = model
        self.timeout_s = timeout_s
        self.bare = bare

    def build_cmd(self, prompt: str) -> list[str]:
        cmd = ["claude", "-p", prompt, "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        if self.bare:
            cmd.append("--bare")
        return cmd

    def extract(self, prompt: str) -> str:
        cmd = self.build_cmd(prompt)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout_s
            )
        except FileNotFoundError as exc:
            raise ProviderError(f"claude CLI not found on PATH: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(f"claude CLI timed out after {self.timeout_s}s") from exc
        if proc.returncode != 0:
            raise ProviderError(
                f"claude CLI exit {proc.returncode}: {proc.stderr.strip()[:500]}"
            )
        try:
            outer = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"claude CLI produced non-JSON stdout: {exc}") from exc
        if outer.get("is_error"):
            raise ProviderError(f"claude CLI error result: {outer.get('result')!r}")
        return outer.get("result", "")


class OpenRouterProvider(Provider):
    """Plain urllib POST to OpenRouter's chat-completions endpoint. 429s get
    exponential backoff. `build_request` is split out from `extract` so it
    can be unit-tested (request construction only) without a real API key or
    network call -- see DISPATCH.md verification (d)."""

    name = "openrouter"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str = OPENROUTER_URL_DEFAULT,
        timeout_s: int = 60,
        max_retries: int = 5,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def build_request(self, prompt: str) -> urllib.request.Request:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        data = json.dumps(body).encode("utf-8")
        return urllib.request.Request(
            self.base_url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key or ''}",
            },
        )

    def extract(self, prompt: str, *, _urlopen=None) -> str:
        urlopen = _urlopen or urllib.request.urlopen
        req = self.build_request(prompt)
        backoff = 1.0
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urlopen(req, timeout=self.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                last_err = exc
                if exc.code == 429 and attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise ProviderError(
                    f"openrouter HTTP {exc.code}: {exc.read()[:300]!r}"
                ) from exc
            except urllib.error.URLError as exc:
                raise ProviderError(f"openrouter connection error: {exc}") from exc
        raise ProviderError(f"openrouter: exhausted retries on 429s: {last_err}")


def make_provider(name: str, config: dict[str, Any], *, model_override: str | None = None) -> Provider:
    pcfg = config["providers"].get(name, {})
    if name == "claude-cli":
        return ClaudeCLIProvider(
            model=model_override or pcfg.get("model", "haiku"),
            timeout_s=pcfg.get("timeout_s", 120),
            bare=pcfg.get("bare", False),
        )
    if name == "openrouter":
        api_key = os.environ.get(pcfg.get("api_key_env", "OPENROUTER_API_KEY"))
        return OpenRouterProvider(
            model=model_override or pcfg.get("model", "meta-llama/llama-3.1-8b-instruct"),
            api_key=api_key,
            base_url=pcfg.get("base_url", OPENROUTER_URL_DEFAULT),
            timeout_s=pcfg.get("timeout_s", 60),
        )
    raise ValueError(f"unknown provider: {name!r}")


def ask_claims(provider: Provider, prompt: str) -> tuple[list | None, str | None, str]:
    """One retry with a "reply with ONLY the JSON" correction, mirroring
    mole/tribunal.py's _ask_json. Returns (data, error, last_raw_reply)."""
    reply = provider.extract(prompt)
    data, err = extract_json_array(reply)
    if err is not None:
        reply2 = provider.extract(
            prompt + "\n\nYour previous reply was not parseable JSON. "
            "Reply with ONLY the JSON array, no commentary, no markdown fences."
        )
        data2, err2 = extract_json_array(reply2)
        if err2 is None:
            return data2, None, reply2
        return None, err2, reply2
    return data, None, reply


# ---------------------------------------------------------------------------
# Rate limiting: per-provider token bucket (requests/min AND tokens/min)
# ---------------------------------------------------------------------------

class TokenBucket:
    """Continuous-refill token bucket. acquire() blocks until `amount` tokens
    are available, sleeping in short slices so multiple threads interleave
    fairly instead of one thread hogging the whole deficit wait."""

    def __init__(self, rate_per_sec: float, capacity: float):
        self.rate = max(rate_per_sec, 1e-9)
        self.capacity = capacity
        self.tokens = capacity
        self._lock = threading.Lock()
        self._last = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

    def acquire(self, amount: float = 1.0) -> None:
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                deficit = amount - self.tokens
                wait = deficit / self.rate
            time.sleep(min(wait, 0.5))

    def snapshot(self) -> float:
        with self._lock:
            self._refill()
            return self.tokens


class RateLimiter:
    """Wraps a requests-bucket and a tokens-bucket for one provider."""

    def __init__(self, *, requests_per_min: float, tokens_per_min: float):
        self.requests_per_min = requests_per_min
        self.tokens_per_min = tokens_per_min
        self._req_bucket = TokenBucket(requests_per_min / 60.0, max(requests_per_min, 1))
        self._tok_bucket = TokenBucket(tokens_per_min / 60.0, max(tokens_per_min, 1))

    def acquire(self, estimated_tokens: int = 500) -> None:
        self._req_bucket.acquire(1.0)
        self._tok_bucket.acquire(float(estimated_tokens))


def _estimate_tokens(prompt: str) -> int:
    # rough, provider-agnostic estimate (chars/4); good enough for governance,
    # not billing. See ../corpuscle/docs/claim-bytes.md sec.7 for the
    # ~25-26 tokens/claim figure used to size tokens_per_min in the config.
    return max(len(prompt) // 4, 1)


# ---------------------------------------------------------------------------
# run: worker pool + single writer thread
# ---------------------------------------------------------------------------

@dataclass
class WorkItem:
    row: dict[str, Any]


@dataclass
class RunStats:
    extracted: int = 0
    rejected: int = 0
    skipped_no_content: int = 0
    errors: int = 0
    claims_added: int = 0
    claims_dropped_junk: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def bump(self, **kw: int) -> None:
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)


def _writer_loop(
    q: "queue_mod.Queue[tuple | None]",
    *,
    data_root: Path,
) -> None:
    """The single writer thread. Every mutation to data/, queue/, and
    dispatch/*.jsonl happens here, one item at a time -- this is what makes
    concurrent worker threads crash-safe without file locks, matching
    mole/store.py's single-writer-append assumption."""
    from mole import attach as attach_mod  # local import: only the writer thread touches it

    while True:
        item = q.get()
        if item is None:
            q.task_done()
            return
        kind = item[0]
        if kind == "extracted":
            _, source_kwargs, claim_records, done_rec = item
            store.append_source(data_root, **source_kwargs)
            store.append_claims_batch(data_root, claim_records)
            if claim_records:
                questions = store.load_live_questions(data_root)
                if questions:
                    live_texts = [
                        c["text"] for c in store.load_all_claims(data_root)
                        if c.get("status") != "retired"
                    ]
                    index = attach_mod.build_index(live_texts, questions)
                    already = {
                        (t["payload"].get("question_id"), t["payload"].get("claim_id"))
                        for t in store.load_all_tasks(data_root)
                        if t.get("kind") == "attach"
                    }
                    task_max = int(store.next_task_id(data_root).split("_")[1]) - 1
                    for claim_id, question_id, sim in attach_mod.candidates(claim_records, index):
                        if (question_id, claim_id) in already:
                            continue
                        task_max += 1
                        store.append_task(
                            data_root, task_id=f"task_{task_max:06d}", kind="attach",
                            payload={"question_id": question_id, "claim_id": claim_id, "sim": sim},
                            created_run=source_kwargs["run_id"],
                        )
            _append_jsonl_line(done_path(data_root), done_rec)
        elif kind == "rejected":
            _, rejection_rec, done_rec = item
            _append_jsonl_line(rejections_path(data_root), rejection_rec)
            _append_jsonl_line(done_path(data_root), done_rec)
        elif kind == "skipped":
            _, done_rec = item
            _append_jsonl_line(done_path(data_root), done_rec)
        q.task_done()


def _process_one(
    row: dict[str, Any],
    *,
    provider: Provider,
    limiter: RateLimiter,
    run_id: str,
    provider_name: str,
    model: str,
    claim_id_counter: "list[int]",
    counter_lock: threading.Lock,
    write_q: "queue_mod.Queue",
    stats: RunStats,
) -> None:
    openalex_id = row["openalex_id"]
    source_id = f"src_openalex-dispatch_{openalex_id}"
    content_kind = row.get("content_kind", "none")
    text = row.get("abstract", "") or ""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if not text.strip():
        stats.bump(skipped_no_content=1)
        write_q.put((
            "skipped",
            {
                "openalex_id": openalex_id, "source_id": source_id,
                "status": "skipped_no_content", "provider": provider_name,
                "run_id": run_id, "ts": ts,
            },
        ))
        return

    prompt = build_prompt(title=row.get("title", ""), content_kind=content_kind, text=text)
    limiter.acquire(_estimate_tokens(prompt))

    t0 = time.monotonic()
    try:
        data, err, raw_reply = ask_claims(provider, prompt)
    except ProviderError as exc:
        stats.bump(errors=1)
        # Deliberately NOT written to done.jsonl: transient provider errors
        # (timeout, CLI missing, connection) should be retried by a later
        # `run` invocation, unlike permanent JSON-shape rejections below.
        print(f"error: {openalex_id}: {exc}", file=sys.stderr)
        return
    duration_s = time.monotonic() - t0

    if err is not None:
        stats.bump(rejected=1)
        write_q.put((
            "rejected",
            {
                "openalex_id": openalex_id, "source_id": source_id,
                "raw_reply": (raw_reply or "")[:2000], "error": err,
                "provider": provider_name, "run_id": run_id, "ts": ts,
            },
            {
                "openalex_id": openalex_id, "source_id": source_id,
                "status": "rejected", "provider": provider_name,
                "run_id": run_id, "ts": ts,
            },
        ))
        return

    kept: list[dict[str, str]] = []
    dropped = 0
    for raw_claim in data:
        validated, verr = validate_claim(raw_claim)
        if validated is None:
            dropped += 1
            continue
        if is_junk_shape(validated["text"]) or is_junk_shape(validated["quote"]):
            dropped += 1
            continue
        kept.append(validated)

    with counter_lock:
        start = claim_id_counter[0]
        claim_id_counter[0] += len(kept)

    claim_records = []
    for i, c in enumerate(kept):
        cid = f"clm_{start + i + 1:06d}"
        rec = store.make_claim_record(
            claim_id=cid,
            source_id=source_id,
            text=c["text"],
            claim_type=c["claim_type"],
            support_in_text=SUPPORT_IN_TEXT_NEUTRAL,
            quote=c["quote"],
            run_id=run_id,
            status="extracted",
        )
        rec.update({
            "tier": "dispatch-abstract",  # distinct from fable's "validated":
                                          # not run through episteme-fable's
                                          # deterministic checks. See DISPATCH.md.
            "extractor": f"claimbase-dispatch/0.1.0+{provider_name}:{model}",
            "prompt_version": "dispatch_v1",
            "kind": c["kind"],
            "content_kind": content_kind,
        })
        claim_records.append(rec)

    authors = row.get("authors") or []
    source_kwargs = dict(
        feed="openalex-dispatch",
        item_key=openalex_id,
        url=row.get("oa_url", ""),
        title=row.get("title", ""),
        author="; ".join(authors),
        published=row.get("publication_date", ""),
        sha256=store.content_sha256(text),
        claim_count=len(claim_records),
        run_id=run_id,
    )
    done_rec = {
        "openalex_id": openalex_id, "source_id": source_id, "status": "extracted",
        "claims_added": len(claim_records), "claims_dropped_junk": dropped,
        "provider": provider_name, "model": model, "content_kind": content_kind,
        "run_id": run_id, "ts": ts, "duration_s": round(duration_s, 2),
    }
    stats.bump(extracted=1, claims_added=len(claim_records), claims_dropped_junk=dropped)
    write_q.put(("extracted", source_kwargs, claim_records, done_rec))


def _load_done_ids(data_root: Path) -> set[str]:
    return {rec["openalex_id"] for rec in _iter_jsonl(done_path(data_root)) if "openalex_id" in rec}


def cmd_run(args: argparse.Namespace, config: dict[str, Any], repo_root: Path) -> None:
    worklist = list(_iter_jsonl(worklist_path(repo_root)))
    if not worklist:
        print(f"run: {worklist_path(repo_root)} is empty or missing -- run `plan` first", file=sys.stderr)
        return

    data_root = Path(args.data_dir) if args.data_dir else repo_root
    # done.jsonl/rejections.jsonl are scoped to data_root, not repo_root: a
    # sandboxed run (--data-dir dispatch/sandbox_data) gets its own markers
    # so it can never cause a later REAL run to skip items as "done".
    done_ids = _load_done_ids(data_root)
    pending = [r for r in worklist if r["openalex_id"] not in done_ids]
    pending.sort(key=lambda r: r.get("priority", 1 << 30))

    provider_name = args.provider or config["run"]["default_provider"]
    workers = args.workers or config["run"]["default_workers"]
    limit = args.limit if args.limit is not None else config["run"]["default_limit"]
    batch = pending[:limit]

    pcfg = config["providers"].get(provider_name, {})
    limiter = RateLimiter(
        requests_per_min=pcfg.get("requests_per_min", 20),
        tokens_per_min=pcfg.get("tokens_per_min", 40000),
    )
    provider = make_provider(provider_name, config, model_override=args.model)
    model = getattr(provider, "model", "?")
    run_id = args.run_id or f"dispatch-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"

    if not batch:
        print("run: nothing to do (worklist exhausted or all items already in done.jsonl)")
        return

    print(
        f"run: {len(batch)} items (of {len(pending)} pending, {len(worklist)} total) "
        f"provider={provider_name} model={model} workers={workers} data_dir={data_root}"
    )

    claim_start = int(store.next_claim_id(data_root).split("_")[1]) - 1
    claim_id_counter = [claim_start]
    counter_lock = threading.Lock()
    stats = RunStats()

    write_q: "queue_mod.Queue" = queue_mod.Queue()
    writer = threading.Thread(
        target=_writer_loop, args=(write_q,),
        kwargs={"data_root": data_root},
        daemon=True,
    )
    writer.start()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _process_one, row,
                provider=provider, limiter=limiter, run_id=run_id,
                provider_name=provider_name, model=model,
                claim_id_counter=claim_id_counter, counter_lock=counter_lock,
                write_q=write_q, stats=stats,
            )
            for row in batch
        ]
        for f in concurrent.futures.as_completed(futures):
            f.result()  # re-raise unexpected exceptions (bugs, not ProviderError)

    write_q.put(None)
    write_q.join()

    print(
        f"run done: extracted={stats.extracted} rejected={stats.rejected} "
        f"skipped_no_content={stats.skipped_no_content} errors={stats.errors} "
        f"claims_added={stats.claims_added} claims_dropped_junk={stats.claims_dropped_junk}"
    )


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace, config: dict[str, Any], repo_root: Path) -> None:
    data_root = Path(args.data_dir) if args.data_dir else repo_root
    worklist = list(_iter_jsonl(worklist_path(repo_root)))
    done_recs = list(_iter_jsonl(done_path(data_root)))
    rejections = list(_iter_jsonl(rejections_path(data_root)))

    by_status: dict[str, int] = {}
    by_provider: dict[str, int] = {}
    claims_total = 0
    for rec in done_recs:
        by_status[rec.get("status", "?")] = by_status.get(rec.get("status", "?"), 0) + 1
        prov = rec.get("provider")
        if prov:
            by_provider[prov] = by_provider.get(prov, 0) + 1
        claims_total += rec.get("claims_added", 0)

    processed = len(done_recs)
    remaining = max(len(worklist) - processed, 0)

    provider_name = args.provider or config["run"]["default_provider"]
    rpm = config["providers"].get(provider_name, {}).get("requests_per_min", 0)
    eta_min = (remaining / rpm) if rpm else float("inf")

    print(f"worklist size:      {len(worklist)}")
    print(f"processed (done):   {processed}")
    print(f"  by status:        {by_status}")
    print(f"remaining:          {remaining}")
    print(f"claims yielded:     {claims_total}")
    print(f"rejections logged:  {len(rejections)}")
    print(f"requests by provider: {by_provider}")
    print(
        f"ETA at {provider_name} rate ({rpm} req/min): "
        f"{'n/a (0 rpm configured)' if rpm == 0 else f'{eta_min:.1f} min'}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_default_config(repo_root: Path) -> Path | None:
    for name in (DEFAULT_CONFIG_NAME, "dispatch_config.json"):
        p = repo_root / name
        if p.exists():
            return p
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mole.dispatch")
    parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    parser.add_argument("--config", default=None, help="path to dispatch_config.yaml/.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="query OpenAlex, rank, write dispatch/worklist.jsonl")
    p_plan.add_argument("--max-works", type=int, default=None)
    p_plan.add_argument("--search", default=None, help="override query.search from config")

    p_run = sub.add_parser("run", help="process top-K unprocessed worklist entries")
    p_run.add_argument("--workers", type=int, default=None)
    p_run.add_argument("--provider", choices=["claude-cli", "openrouter"], default=None)
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--model", default=None)
    p_run.add_argument("--run-id", default=None)
    p_run.add_argument(
        "--data-dir", default=None,
        help="repo-root-like path under which data/ and queue/ live; "
             "defaults to --repo-root (the REAL claimbase data). Pass a "
             "sandbox path (e.g. dispatch/sandbox_data) for dry runs.",
    )

    p_status = sub.add_parser("status", help="progress report + ETA")
    p_status.add_argument("--provider", default=None)
    p_status.add_argument(
        "--data-dir", default=None,
        help="same meaning as `run --data-dir`; report on that data dir's "
             "done.jsonl/rejections.jsonl (defaults to --repo-root)",
    )

    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config) if args.config else _find_default_config(repo_root)
    config = load_config(config_path)

    if args.cmd == "plan":
        cmd_plan(args, config, repo_root)
    elif args.cmd == "run":
        cmd_run(args, config, repo_root)
    elif args.cmd == "status":
        cmd_status(args, config, repo_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
