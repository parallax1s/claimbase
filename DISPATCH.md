# DISPATCH — scale-up extraction dispatcher

`mole/dispatch.py` fans claim extraction over MANY documents (candidates
pulled from the free OpenAlex API) across N parallel LLM endpoints, with
rate governance and importance-first ordering. It is a standalone, additive
tool: a bulk-import front door that writes into claimbase's **existing**
formats by calling the **existing**, unmodified `mole.store` and
`mole.attach` modules — the same modules `mole/fable.py` uses.

**Not wired into cron.** `.github/workflows/mole.yml` and `.github/workflows/
fable.yml` are untouched. Nothing about this tool runs automatically; it is
invoked by hand (`python -m mole.dispatch ...`) when someone wants to seed
the graph from a large batch of OpenAlex works, e.g. before or alongside the
"Phase 2 real-corpus" experiment described in `../corpuscle/HANDOFF.md` §5.

## How it composes with the existing mole/queue

- **Same data files, same shapes.** `dispatch run` calls `store.append_source`,
  `store.append_claims_batch`, `store.next_claim_id`, etc. directly — it does
  not reimplement id allocation, hashing, or the append format. Every record
  it writes validates against SCHEMA.md's `data/sources.jsonl` and
  `data/claims.jsonl` contracts field-for-field (verified below).
- **Attach candidates flow into `queue/pending.jsonl` exactly like a fable
  pass.** After appending a batch of claims, the writer thread calls
  `mole.attach.build_index` / `candidates` over the live corpus and enqueues
  `attach` tasks with the standard `{"question_id", "claim_id", "sim"}`
  payload — the same lexical, calibrated-floor (0.28) scorer WORKER.md
  already documents. **Nothing about the judgment rubric changes**: these
  are candidates for a worker to judge stance/strength on via the normal
  `attach` task flow, never auto-attachments, exactly as WORKER.md specifies.
- **Known nondeterminism (benign):** because `run` processes items
  concurrently, the "live claims corpus" a given item's attach-scoring sees
  depends on which other items the writer thread has already flushed at that
  moment (TF-IDF/IDF weights shift slightly with corpus size) — this is the
  same live-corpus scoring fable.py itself does, just under concurrent
  ordering instead of fable's one-source-at-a-time sequence. Practical
  effect: a few borderline candidates may be missed on a given run. Since
  claimbase already ships `python -m mole attach-backfill --run-id <id>`
  (unmodified, in `mole/__main__.py`) to enqueue attach tasks for any live
  claim not yet scored, running that after a big `dispatch run` batch closes
  the gap — no new tooling needed.
- **New optional fields, old readers unaffected.** Claim records get the
  same *shape* of extra fields fable-tier claims already carry
  (`tier`/`extractor`/`prompt_version`/`kind`), plus one new field,
  `content_kind` (`"abstract"|"fulltext"|"none"`), recording what was
  actually available to extract from. `tier` is set to `"dispatch-abstract"`
  rather than fable's `"validated"` — these claims have NOT passed
  episteme-fable's deterministic validator, so reusing `"validated"` would
  overclaim. `compile.py`/`pipeline.py`/`atlas.py` read known fields via
  `.get()` and ignore unknown ones, so this is additive and safe.
- **`feed` value is new, not in `feeds.yaml`.** Dispatched sources use
  `feed="openalex-dispatch"`, which is intentionally not a `feeds.yaml`
  entry (nothing in the codebase requires `feed` to resolve there at
  runtime — it is a descriptive label). This keeps `feeds.yaml` itself
  untouched, as required.
- **v0 has no fulltext fetcher.** `run` extracts from the OpenAlex abstract
  (reconstructed from `abstract_inverted_index`, which `plan` already fetched
  for ranking — no second network call). Items with no abstract are recorded
  as `content_kind="none"` in the worklist and skipped at run time
  (`status="skipped_no_content"` in `dispatch/done.jsonl`) rather than
  silently dropped, so a later version that adds a PDF/HTML fulltext fetcher
  can pick them back up without re-running `plan`.

## Commands

```
python -m mole.dispatch plan   [--max-works N] [--search "..."]
python -m mole.dispatch run    [--workers N] [--provider claude-cli|openrouter]
                                [--limit K] [--model M] [--run-id ID]
                                [--data-dir PATH]
python -m mole.dispatch status [--provider NAME] [--data-dir PATH]
```

Global: `--repo-root PATH` (default: the claimbase checkout containing this
file), `--config PATH` (default: `dispatch_config.yaml` at the repo root, if
present; otherwise built-in defaults — both subcommands run even with zero
config file, useful for tests).

### `plan`

Queries `https://api.openalex.org/works` with cursor paging
(`meta.next_cursor`) and server-side sort (`sort=cited_by_count:desc` — the
importance prior), applies the configured filter (`search`/`concepts`/
`venues`/`from_publication_date`, ANDed), keeps only works with an OA
location (`open_access.is_oa` or a `best_oa_location`), and writes
`dispatch/worklist.jsonl` (full rewrite each run, temp-then-`os.replace`).
No fulltext is fetched — the `select` param already asked OpenAlex for
`abstract_inverted_index`, so the worklist also carries the reconstructed
abstract and a `content_kind` hint, sparing `run` a second network round
trip per document.

Worklist row: `{openalex_id, doi, title, cited_by_count, oa_url, priority,
abstract, content_kind, publication_date, authors}`.

### `run`

Loads the worklist, drops anything already in **this data-dir's**
`dispatch/done.jsonl` (see Crash safety below), sorts the remainder by
`priority` ascending (1 = most cited = most important), takes the first
`--limit` (default from config `run.default_limit`), and fans them across
`--workers` threads (default `run.default_workers`). Each worker: rate-limit
acquire → build the extraction prompt → call the provider → validate JSON →
junk-filter → hand the result to a single writer thread. Prints a one-line
summary (`extracted`/`rejected`/`skipped_no_content`/`errors`/
`claims_added`/`claims_dropped_junk`).

### `status`

Reports worklist size, processed count (by status), claims yielded,
per-provider request counts, and an ETA (`remaining / requests_per_min` for
the given `--provider`'s configured rate).

## Sandbox verification mode (`--data-dir`)

`run` and `status` take `--data-dir PATH`. It is the root `mole.store`
functions treat as `repo_root` for `data/sources.jsonl`, `data/claims.jsonl`,
and `queue/pending.jsonl` — **and** the root dispatch's own
`dispatch/done.jsonl` / `dispatch/rejections.jsonl` live under. Default is
the real claimbase root (real `data/`, real `queue/`). Passing
`--data-dir dispatch/sandbox_data` writes everything under
`dispatch/sandbox_data/{data,queue,dispatch}/` instead — fully isolated:

- A sandbox `run` can never write into the real `data/`/`queue/`.
- A sandbox `run`'s `done.jsonl` markers are scoped to the sandbox dir, so
  running a sandbox verification batch can never cause a **later real**
  `run` to skip those same items as "already done" (this was caught and
  fixed during verification — see `git`-free history: `done_path`/
  `rejections_path` take `data_root`, not `repo_root`).
- `dispatch/worklist.jsonl` itself is NOT data-dir-scoped (candidate
  discovery is shared regardless of where you later choose to write
  extracted records).

## Config reference (`dispatch_config.yaml`)

PyYAML is used because claimbase already depends on it
(`pyproject.toml: PyYAML>=6.0`); a `.json` config with the same shape works
too (checked by file extension). See the file for full comments; keys:

- `query.search` / `query.concepts` / `query.venues` /
  `query.from_publication_date` / `query.max_works` — OpenAlex filter(s).
- `openalex.mailto` — **fill in a real contact address** before large runs
  (OpenAlex's "polite pool"); `plan` warns if it's still the placeholder.
  Deliberately not auto-filled with anyone's real address.
- `openalex.per_page` — OpenAlex page size (max 200).
- `providers.claude-cli.{model,timeout_s,bare,requests_per_min,tokens_per_min}`
  — `bare: true` requires `ANTHROPIC_API_KEY` set (verified: without it,
  `--bare` fails auth even when the interactive CLI is logged in via
  OAuth/keychain, since `--bare` strictly requires `ANTHROPIC_API_KEY` or
  `apiKeyHelper`). Default model `haiku` — verified at ~$0.02/call vs.
  ~$0.17/call for the session's default model, for a short abstract-only
  extraction prompt.
- `providers.openrouter.{model,api_key_env,base_url,timeout_s,
  requests_per_min,tokens_per_min}` — reads the key from the named env var
  (default `OPENROUTER_API_KEY`); 429s get exponential backoff, up to 5
  attempts.
- `run.default_provider` / `run.default_workers` / `run.default_limit`.

## Scaling notes: workers vs. rate caps

- `--workers` controls *concurrency* (how many extraction calls are in
  flight); `requests_per_min`/`tokens_per_min` control *throughput ceiling*
  per provider, enforced by a token-bucket (`RateLimiter`) shared across all
  worker threads for that run. Raising `--workers` past what the rate caps
  allow just means more threads blocked in `limiter.acquire()` — safe, but
  pointless; a good rule of thumb is `workers ≈ requests_per_min / 60 ×
  avg_latency_s` (little's-law-ish) so most workers are usually mid-flight
  rather than idle-blocked.
- Rate caps are per **provider config**, not per run — running two `dispatch
  run` processes against the same provider concurrently will each open a
  fresh bucket and can jointly exceed the intended ceiling; do not do that
  against a real API key without accounting for it.
- Token estimate for `tokens_per_min` governance is a rough `len(prompt)//4`
  heuristic (governance only, not billing). For sizing, `../corpuscle/docs/
  claim-bytes.md` §7 gives ~25-26 tokens/claim as a context-budget
  reference (gpt2/tiktoken and an 8k-vocab BPE trained on this corpus agree
  closely); a whole abstract+prompt is dominated by the ~150-300 word
  abstract itself (roughly 200-400 tokens), not by claim count, since the
  claim JSON is the *output*, not the input, side of the request.
- OpenAlex itself has no documented hard rate limit for the anonymous free
  tier beyond "be polite" (hence the `mailto` param); `plan`'s cursor-paged
  fetch of a few hundred works completes in a handful of requests regardless
  of `--workers` (that concurrency knob only applies to `run`, not `plan`,
  which is a single-threaded, one-shot query).
- claimbase's own concurrency precedent (`../corpuscle/HANDOFF.md` §3, the
  "Kaggle fleet") treats slot exhaustion as instant rejection + re-push,
  not a queue. This dispatcher instead blocks in the rate limiter rather
  than rejecting — appropriate for HTTP/CLI providers with soft per-minute
  caps rather than hard concurrent-slot ceilings.

## Ingestion-time junk filter

`mole/dispatch.py::is_junk_shape` drops claims whose `text` or `quote` look
like CSS rules, MathJax markup, or raw JS (`ISSUES.md`: "Extractor pulls
MathJax CSS fragments as claims", found 2026-08-25, 2,326 junk rows in the
existing corpus — **no fix has landed anywhere in claimbase as of this
writing**; this dispatcher's filter is new code, not a reuse of an existing
pattern, since none exists yet). It also uses a generic symbol-density
heuristic (>40% non-alphanumeric characters in a ≥24-char string) to catch
minified-looking blobs the specific patterns miss. This only prevents *new*
junk from `dispatch run`; it does not retire the 2,326 rows already in
`data/claims.jsonl` — that cleanup is the separate "retirement sweep via the
normal refine/retire path" ISSUES.md already recommends, out of scope here.

## Crash safety

- `dispatch/worklist.jsonl`: `plan` always does write-temp-then-
  `os.replace` — never a partially written worklist.
- `dispatch/done.jsonl` / `dispatch/rejections.jsonl`: single writer thread,
  one `open(...,'a').write(json+'\n')` per record — the same no-lock,
  single-writer-append assumption `mole/store.py` already uses. Re-running
  `run` skips any `openalex_id` already present in `done.jsonl` for that
  data-dir.
- `done.jsonl` only records **terminal** outcomes: `"extracted"`,
  `"rejected"` (malformed JSON survived one correction retry — logged to
  `rejections.jsonl` with the raw reply, capped at 2000 chars), or
  `"skipped_no_content"`. Transient provider errors (timeout, `claude` not
  on PATH, connection failure) are **not** written to `done.jsonl` — they
  print to stderr and the item stays pending, so the next `run` retries it
  automatically.
- `data/sources.jsonl` / `data/claims.jsonl` / `queue/pending.jsonl` writes
  go through unmodified `mole.store` functions, inheriting its existing
  atomicity guarantees (append-only for these three files; no in-place
  rewrite happens during a dispatch run since retirement/re-extraction logic
  is deliberately not implemented here — a dispatched source is always new).

## Verification performed (no money-losing runs against real data)

1. **`plan` for real**, `max_works=50`, `query.search="AI alignment"` — 50
   works fetched, all 50 kept (all had an OA location), top-10 by
   `cited_by_count` written to `dispatch/worklist.jsonl` (topped by the
   AlphaFold paper at 46,706 citations — OpenAlex's plain `search` param is
   a broad fulltext-contains search, so a generic two-word query pulls in
   adjacent high-citation ML/bio work; `concepts`/`venues` filters narrow
   this for real campaigns).
2. **`run --limit 2 --provider claude-cli --model haiku
   --data-dir dispatch/sandbox_data`** — real `claude` CLI calls, real
   OpenAlex abstracts, 2/2 extracted, 18 claims total, 0 rejections.
   Resulting `data/sources.jsonl` and `data/claims.jsonl` records checked
   field-for-field against SCHEMA.md's required key set — all present,
   correctly typed, no unexpected keys beyond the documented fable-tier-style
   extras. Separately verified the attach-task integration end-to-end (not
   exercised by the 2-item run above, since claimbase's live questions are
   AI-alignment-specific and these sample works are biomedical/ML papers
   lexically unrelated to them): seeded a throwaway sandbox with a copy of
   the real `data/questions.jsonl` and confirmed `queue/pending.jsonl` gets
   correctly shaped `attach` tasks (`task_NNNNNN`, `payload.question_id`/
   `claim_id`/`sim`), then deleted the throwaway sandbox.
3. **Rate limiter unit check** — a 6-req/min bucket given a burst of 9 fake
   `acquire()` calls let the first 6 through near-instantly (using the
   bucket's full starting capacity) and spaced the remaining 3 exactly
   ~10s apart (60s / 6 = 10s/token) — bucket behaves correctly.
4. **OpenRouter dry run** — `OpenRouterProvider.build_request()` inspected
   without any network call or API key: POST to
   `https://openrouter.ai/api/v1/chat/completions`, correct
   `Authorization: Bearer <key-or-empty>` header, JSON body with
   `model`/`messages`/`temperature`, prompt correctly embedded in
   `messages[0].content`.

## Uncertainties / deliberately out of scope for v0

- No fulltext PDF/HTML extraction — v0 is abstract-only, as specified.
  `oa_url` is captured in the worklist for a future fulltext fetcher to use.
- OpenAlex's plain `search` filter is broad (fulltext-contains); tighter
  targeting needs `concepts`/`venues` IDs, which the config supports but the
  shipped `dispatch_config.yaml` doesn't populate (left as an example
  search-only config).
- Attach-candidate recall under concurrent `run` is corpus-order-dependent
  (see "Known nondeterminism" above) — mitigated by the existing
  `mole attach-backfill` command, not by new code here.
- `openrouter` provider was only dry-run tested (request construction) per
  the task's "no money assumed" constraint — never exercised against a real
  key/endpoint.
- `claude-cli` provider cost scales with which Claude model your `claude`
  CLI session defaults to unless `--model`/config overrides it; `haiku` was
  used for verification specifically to keep it cheap.
