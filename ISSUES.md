# Known issues

## Extractor pulls MathJax CSS fragments as claims (found 2026-08-25)

A compression benchmark over `data/claims.jsonl` (see
`../corpuscle/docs/claim-bytes.md`) found the extractor emits MathJax/CSS
fragments from LessWrong/AlignmentForum posts as fake "claims": 2,326 junk
rows overall; 57.6% of exact-duplicate *active* claims are this junk.
Suggested fix: ingestion-time shape filter (CSS/JS token patterns) in
`extractor/engine.py` + a retirement sweep for existing junk rows via the
normal refine/retire path. Genuine active-claim dup rate is only ~3-5%;
a normalized-text uniqueness index at ingestion captures most of it.

**Status 2026-10-29:** both halves landed. Ingestion-time filter:
`mole/dispatch.py::is_junk_shape` (CSS-rule / MathJax / JS / raw-tag
patterns + a >40% symbol-density heuristic), applied to `text` and `quote`.
Retirement sweep: `python -m mole sweep-junk --run-id <id> [--apply]`
(`mole/sweep.py`) retires live rows failing the same test in place
(`status: retired`, `retired_reason: junk_shape`, `retired_by: <run>`) and
marks pending tasks that reference them `obsolete`. First sweep
(`sweep-20261029`): 2,693 live junk rows retired across 31 sources
(2,626 on text, 67 on quote only, 5 heuristic-only — equation/table
fragments), 6,104 pending identity/attach/refine tasks obsoleted.
Still open: the ~3-5% genuine duplicate rate (uniqueness index).
