# claimbase

A living, versioned claim graph — claims extracted from public sources, judged
and cross-linked by AI workers, with every update a git commit.

**The bet:** arguments should be data. When claims are typed records with
provenance, you can ask questions prose can't answer — where do these sources
actually collide, which assumptions are load-bearing, what new evidence touches
what conclusions, and where is the frontier worth working on.

The pilot domain is the **AI alignment-tractability debate**.

## How it works

```
feeds.yaml ──► mole (cron, no LLM) ──► data/*.jsonl + queue/pending.jsonl
                                              │
        workers (any subscription-backed agent) drain the queue
                                              │
                            judged edges, identities, confidences
                                              │
                              compiled artifacts ──► site views
```

- **The mole** fetches sources *into memory only*, extracts claims with a
  deterministic offline engine, persists claims + provenance, and discards the
  content. No source documents are stored in this repository, ever — only
  claim-level extracts with citation (url, content hash, short quote).
- **The queue** holds judgment work: claim refinement, identity matching, edge
  relations, adversarial verification.
- **Workers** are stateless. Any agent that can read this repo can claim a
  batch, judge it per [WORKER.md](WORKER.md), and commit results. Contributing
  compute means *running a worker on your own subscription* — no keys change
  hands, and every judgment is signed with the model that made it and is
  subject to adversarial re-review.
- **History is the audit log.** Every belief update is a commit. Diffs between
  weeks are the "what moved" digest.

## Layout

| Path | What |
|---|---|
| `data/sources.jsonl` | one record per ingested source (no content) |
| `data/claims.jsonl` | extracted claims with provenance |
| `data/edges.jsonl` | judged relations between claims |
| `queue/pending.jsonl` | judgment work awaiting a worker |
| `runs/` | mole run summaries |
| `feeds.yaml` | what the mole watches |
| `SCHEMA.md` | record formats (the contract) |
| `WORKER.md` | the judgment protocol and rubric |

## Running the mole locally

```bash
python -m pip install -e '.[all]'
python -m mole run --since 2026-06-01 --run-id manual-$(date +%Y%m%d)
pytest -q
```

The scheduled run lives in `.github/workflows/mole.yml` (daily, tokenless —
the mole needs no LLM).

## Provenance & copyright posture

Sources are fetched transiently, never redistributed, never retained. What is
stored: the claim text (a transformative extract), source URL, author, title,
publication date, a content hash for change detection, and at most a short
quotation for verification. Feeds are limited to publicly accessible sources.

## Lineage

The extraction engine in `extractor/` is a minimal vendored derivative of
[Episteme](https://github.com/parallax1s/Episteme) (MIT). The cross-source
consolidation and adversarial-verification patterns were developed there.
