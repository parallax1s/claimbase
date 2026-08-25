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
