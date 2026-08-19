# Corpuscle (downstream consumer)

[corpuscle](../corpuscle) (github.com/parallax1s/corpuscle) tests whether the
claim graph can serve as the *canonical substrate for LLM training data*:
compress sources to verified claims, then re-expand them into stylistically
varied text — fresh each epoch — and pretrain on the expansion. Full research
brief: `../corpuscle/docs/brief.html` (gap map + 60-paper bibliography,
Aug 2026 survey).

What corpuscle takes from claimbase:

- **Sources.** Phase 0 (round-trip fidelity) samples documents from
  `data/sources.jsonl` and re-fetches them by URL; claim text and quotes in
  `data/claims.jsonl` serve as a cross-check on fresh extraction.
- **The store itself, eventually.** If Phase 1 vindicates the claim layer,
  claimbase's schema (tiers, edges, adversarial verification, git-native
  audit) is the canonical store the training expansion mints from.

What corpuscle may feed back:

- A **GenRM-style generative truth gate** on claim admission — directly aimed
  at the `queue/pending.jsonl` backlog, where extraction throughput outruns
  judgment ~40×.
- A **credence layer**: BTProp-style belief propagation over `data/edges.jsonl`
  to give claims propagated truth probabilities (leaf credences from
  verifiers/literature; the calculus, not the LLM, does the arithmetic).

For workers: nothing in `WORKER.md` changes. Corpuscle is read-only toward
this repo's data; any future write path will arrive as ordinary queue tasks
under the existing rubric.

**Status 2026-08-20:** Phase 1 concluded — claim-formatted text is ~2.5x more
valuable per token for extractable knowledge (B 25.8% vs raw 0.9%, 7-condition
matrix, 2-3 seeds each). Full results and agent handoff: `../corpuscle/HANDOFF.md`.
