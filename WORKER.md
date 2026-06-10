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

## Rubric — verify tasks

You are an adversarial reviewer of someone else's `contradicts`/`same` edge.
Re-read both claim texts. Try to REFUTE the label. Default to overturning when
uncertain. Confirmed edges get `verified: true` and your model id as
`verifier`; overturned edges get the corrected relation (often `refines` or
removal).

## Budget conduct

Quota exhaustion is normal, not an error. Stop cleanly at a batch boundary;
the queue persists; the next worker resumes. Never leave tasks half-judged.
