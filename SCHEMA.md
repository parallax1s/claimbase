# SCHEMA — the record contract

All data files are JSONL (one JSON object per line, UTF-8, append-mostly).
IDs are stable and never reused. All timestamps are UTC ISO-8601 dates or
datetimes. Every file is deterministic given the same inputs: stable ordering,
no wall-clock values except the run id passed in by the scheduler.

## data/sources.jsonl

One line per ingested source item. **Content is never stored.**

```json
{
  "id": "src_lw_abc123",            // "src_" + feed key + item key
  "feed": "lesswrong-af",            // key into feeds.yaml
  "url": "https://...",
  "title": "...",
  "author": "...",
  "published": "2026-06-01",
  "content_sha256": "…",            // hash of fetched text, for change detection
  "claim_count": 14,
  "run_id": "20260611"
}
```

## data/claims.jsonl

```json
{
  "id": "clm_000001",                // zero-padded, monotonically assigned
  "source_id": "src_lw_abc123",
  "text": "…",                       // the claim, self-contained
  "type": "empirical",               // empirical|statistical|causal|predictive|normative|definitional|descriptive|historical|other
  "support_in_text": 0.42,           // engine's in-source support estimate, 0-1
  "quote": "…",                      // ≤ 300 chars of source context, for verification
  "run_id": "20260611",
  "status": "extracted"              // extracted | refined | retired
}
```

`retired` claims stay in the file (history matters); they are excluded from
artifacts. A refinement that rewrites a claim appends a NEW claim with
`"refines_claim": "clm_000001"` and retires the old one.

## data/edges.jsonl

```json
{
  "a": "clm_000001",
  "b": "clm_000042",
  "relation": "contradicts",         // supports|contradicts|refines|same
  "confidence": 0.78,                // judge's credence in the relation, 0-1
  "note": "one sentence a reader can check",
  "judge": "claude-sonnet-4-6",      // model identity of the judging worker
  "verified": true,                  // present iff an adversarial reviewer confirmed
  "verifier": "claude-fable-5",      // model identity of the reviewer, if verified
  "run_id": "20260611"
}
```

Edges are undirected for `same`/`contradicts`, directed a→b for
`supports` (a supports b) and `refines` (a refines b). The pair (a,b) plus
relation is unique; a re-judgment replaces the line.

## queue/pending.jsonl

```json
{
  "id": "task_000123",
  "kind": "identity",                // refine | identity | edge | verify
  "payload": { "a": "clm_000001", "b": "clm_000042", "sim": 0.61 },
  "created_run": "20260611",
  "status": "pending"                // pending | done | skipped
}
```

Workers mark tasks `done` in place and append results to the data files in the
same commit. A worker batch must be atomic: one commit containing both the
status flips and the produced records.

## runs/<run_id>.json

Mole run summary: feeds polled, items seen/new, claims extracted, tasks
enqueued, warnings. Diagnostic only — not part of the graph.

## feeds.yaml

```yaml
feeds:
  - key: lesswrong-af
    kind: graphql-forum            # LessWrong/Alignment Forum API
    url: https://www.alignmentforum.org/graphql
    view: new                      # newest posts
    limit: 25
  - key: arxiv-csai
    kind: arxiv
    query: cat:cs.AI AND (alignment OR "AI safety")
    limit: 25
  - key: blog-rss
    kind: rss
    url: https://thezvi.substack.com/feed
    limit: 10
```
