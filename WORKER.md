# WORKER — the judgment protocol

A worker is any LLM agent (Claude, Codex, or other) that drains
`queue/pending.jsonl`. Workers are stateless and interchangeable; what keeps
the graph honest is the protocol, not the worker.

## Protocol

1. Read `queue/pending.jsonl`; take the first N `pending` tasks of one kind
   (N ≤ 50). Read the referenced claims from `data/claims.jsonl`.
2. Judge each task per the rubric below. **When in doubt, the weaker label
   wins** (unrelated > refines > supports; never escalate to `contradicts` or
   `same` under uncertainty).
3. Append results to the data files; flip the tasks to `done`; for every new
   `contradicts` or `same` edge, enqueue a `verify` task.
4. Commit everything as ONE commit:
   `worker(<kind>): <n> tasks — <model-id>`. Stamp `judge` (or `verifier`)
   with your real model identity. Never commit partial batches.
5. `verify` tasks must be taken by a DIFFERENT model family than the original
   judge whenever possible (cross-model review beats same-model review).

## Task kinds: `identity` vs `edge`

Both are judged with the relation rubric below. `identity` tasks are
high-similarity pairs where `same` is plausible; `edge` tasks are looser
candidates. The kind sets expectations only — any relation may be returned for
either. **`unrelated` judgments are NOT appended to edges.jsonl** — flip the
task to `done` and store nothing.

## Rubric — relations between claim A and claim B

Judge PROPOSITIONS, not topics.

- **same** — paraphrases: evidence for one is evidence for the other. Topical
  overlap is not enough.
- **supports** — if A is true, B becomes meaningfully more likely (premise,
  instance, mechanism). Mark direction a→b.
- **contradicts** — both cannot be true as stated, or they take opposing sides
  of one question. Genuine collision a reader should see, not different
  emphasis.
- **refines** — same subject; A narrows, conditions, or qualifies B without
  opposing it.
- **unrelated** — topical neighbors without a real logical relation. Default
  here when uncertain. Sparse honest edges beat dense noisy ones.

`confidence` is your credence in the chosen relation (0–1). `note` is one
sentence a reader can check against the two claim texts.

## Rubric — refine tasks

A claim is refined when it is not self-contained (dangling pronouns, missing
subject), fuses several assertions, or is too vague to ever collide with
anything. Rewrite it as one or more atomic, operationalized claims; keep
meaning, never add content the source doesn't carry. Append new claims with
`refines_claim` set and retire the original (see SCHEMA.md).

## Rubric — extract tasks

An `extract` task asks for worker-side deep extraction of one source with the
episteme-fable engine — self-contained rewritten claims (validated
deterministically) plus document theses. Do NOT extract by hand; run:

```
python -m mole fable --source <source_id> --run-id <your-run-id>
# or drain the queue oldest-first:
python -m mole fable --drain 5 --run-id <your-run-id>
```

Requirements: the episteme-fable repo present locally (env
`EPISTEME_FABLE_SRC` if not at the default sibling path) and the `claude`
CLI on PATH. The command refetches the source, supersedes its regex-era
claims (retire + obsolete their refine tasks), appends fable-tier claims and
`data/theses.jsonl` rows, enqueues attach candidates (theses via the hybrid
matcher), and flips the extract task — commit everything it changed as one
batch. If the refetch fails (dead URL, paywall), flip the task `skipped`
with a note; if the engine is unavailable, leave the task pending.

## Rubric — verify tasks

You are an adversarial reviewer of someone else's `contradicts`/`same` edge.
Re-read both claim texts. Try to REFUTE the label. Default to overturning when
uncertain. Confirmed edges get `verified: true` and your model id as
`verifier`; overturned edges get the corrected relation (often `refines` or
removal).

## Rubric — attach tasks

You judge whether a claim genuinely BEARS ON a question: would a researcher
tracking this question want this claim in its evidence feed? Keyword overlap
is not bearing — "AI control" in a medication-safety paper does not bear on
the AI-control-sufficiency question. Irrelevant candidates get no attachment;
the task flip is the record. Relevant ones get a stance:

- **bears_yes / bears_no** — the claim is evidence or argument for the
  affirmative / negative pole of the question as worded.
- **informs** — genuine evidence without polarity (a measurement, a case,
  a mechanism).
- **challenges_framing** — the claim argues the question itself is confused,
  ill-posed, or wrongly split.

`strength` is your credence the claim genuinely bears (relevance × evidential
weight); below 0.4, prefer no attachment. Extraction fragments are judged
irrelevant and enqueued for `refine`, same as in edge tasks. `note` is one
sentence a reader can check against the claim and question texts.

## Record conventions

- `run_id` on worker-produced records carries the task's `created_run` (the
  mole run that generated the work), so a graph state is traceable to its
  ingestion wave.
- A `verify` task payload is `{"a", "b", "relation", "edge_judge"}` — the edge
  to re-examine and the model that judged it (verifiers must differ in model
  family when possible).
- Claims that are obvious extraction fragments (bare headings, sentence
  shards) should be judged `unrelated` and additionally enqueued as a `refine`
  task if one does not already exist for that claim.

## Task id allocation (mole/worker races)

Task ids are a single numeric sequence and are never reused, but the mole
(daily cron) and workers (anywhere) both append to it — so allocate
pessimistically:

1. Allocate ids by scanning pending.jsonl at commit time, not at claim time.
2. If your push is rejected because the remote moved, `git pull --rebase`,
   then renumber every task YOU created to continue from the new remote
   maximum before completing the rebase. Your task ids are yours to move
   until they are pushed; ids that have reached origin are frozen.
3. Records that reference task ids do not exist (attachments/edges reference
   claim and question ids), so renumbering is always safe.

This convention was first exercised in commit 6dc53ca (worker range collided
with mole run 28569379761 and was shifted +875).

## Budget conduct

Quota exhaustion is normal, not an error. Stop cleanly at a batch boundary;
the queue persists; the next worker resumes. Never leave tasks half-judged.

## Rubric — tribunal (Richtschwert)

Contested questions go to trial: `python -m mole tribunal --docket` lists
the docket (UNTRIED / STANDING / RETRIAL — new evidence past a verdict's
watermark triggers retrial); `--question q_NNNNNN --run-id <rid>` runs one:
deterministic evidence marshal → PRO/CONTRA steelman briefs (citations
required) → cross-examination → multi-family bench (claude + GitHub Models)
returning credence INTERVALS + cruxes + would-change-my-mind. Verdicts land
in data/verdicts.jsonl (status `standing`), dissents recorded verbatim.
Needs the episteme-fable repo locally (EPISTEME_FABLE_SRC) and claude CLI;
the GitHub judge self-drops without a token. Commit the verdict with the
run. Verdict credences are forecasts: never edit a standing verdict — retry
the question and let the ledger show the revision.
