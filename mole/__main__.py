"""
CLI entry point: python -m mole <command> [options]

Commands:
  run     Fetch new items, extract claims, enqueue tasks, rebuild artifacts.
  compile Build data/artifact.json, data/atlas.json, and ATLAS.md.
  map     Print the fog-of-war atlas (ATLAS.md), building it if missing.
  attach-backfill
          Enqueue 'attach' tasks for live claims the question layer has not
          seen yet (one-off after adding/expanding data/questions.jsonl).
  fable   Worker-side deep extraction for one source or pending extract
          tasks (requires the episteme-fable engine + claude CLI).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_run(args: argparse.Namespace) -> None:
    from mole.atlas import write_atlas
    from mole.compile import compile
    from mole.pipeline import run

    repo_root = Path(args.repo_root).resolve()
    summary = run(repo_root=repo_root, since=args.since, run_id=args.run_id)
    # Rebuild the site artifact and the fog-of-war atlas so the daily cron
    # commits fresh data/artifact.json, data/atlas.json, and ATLAS.md.
    compile(repo_root, run_id=args.run_id)
    write_atlas(repo_root, run_id=args.run_id)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Feed failures must surface, not rot in runs/*.json: echo them to stderr,
    # and if most feeds died, fail the run (artifacts are already written) so
    # the scheduled workflow goes red instead of silently going arxiv-only.
    warnings = summary.get("warnings", [])
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    feeds_polled = summary.get("feeds_polled", 0)
    if feeds_polled and len(warnings) > feeds_polled / 2:
        print(
            f"error: {len(warnings)} of {feeds_polled} feeds failed",
            file=sys.stderr,
        )
        sys.exit(3)


def _cmd_compile(args: argparse.Namespace) -> None:
    from mole.atlas import write_atlas
    from mole.compile import compile

    repo_root = Path(args.repo_root).resolve()
    artifact = compile(repo_root)
    atlas = write_atlas(repo_root)
    # Print a brief summary (the full artifacts are written to disk)
    print(
        json.dumps(
            {
                "generated_run": artifact["generated_run"],
                "counts": artifact["counts"],
                "atlas": {
                    "districts": len(atlas["districts"]),
                    "claims": len(atlas["claims"]),
                    "tensions": len(atlas["tensions"]),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _cmd_map(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    md_path = repo_root / "ATLAS.md"
    if not md_path.exists():
        from mole.atlas import write_atlas

        write_atlas(repo_root)
    print(md_path.read_text(encoding="utf-8"), end="")


def _cmd_attach_backfill(args: argparse.Namespace) -> None:
    from mole import attach as attach_mod
    from mole import store
    from mole.pipeline import _needs_refine

    repo_root = Path(args.repo_root).resolve()
    questions = store.load_live_questions(repo_root)
    if not questions:
        print("no data/questions.jsonl — nothing to backfill", file=sys.stderr)
        sys.exit(1)

    claims = [c for c in store.load_all_claims(repo_root) if c.get("status") != "retired"]
    already = {
        (t["payload"].get("question_id"), t["payload"].get("claim_id"))
        for t in store.load_all_tasks(repo_root)
        if t.get("kind") == "attach"
    }
    claims_with_task = {c for _, c in already}
    fresh = [
        c for c in claims
        if c["id"] not in claims_with_task and not _needs_refine(c["text"])
    ]
    index = attach_mod.build_index([c["text"] for c in claims], questions)
    task_max = int(store.next_task_id(repo_root).split("_")[1]) - 1
    enqueued = 0
    for claim_id, question_id, sim in attach_mod.candidates(fresh, index):
        if (question_id, claim_id) in already:
            continue
        task_max += 1
        store.append_task(
            repo_root,
            task_id=f"task_{task_max:06d}",
            kind="attach",
            payload={"question_id": question_id, "claim_id": claim_id, "sim": sim},
            created_run=args.run_id,
        )
        enqueued += 1
    print(json.dumps({
        "claims_scanned": len(fresh),
        "attach_enqueued": enqueued,
        "questions": len(questions),
    }, indent=2))


def _cmd_fable(args: argparse.Namespace) -> None:
    from mole import store
    from mole.fable import pending_extract_sources, run_fable

    repo_root = Path(args.repo_root).resolve()
    drain_mode = not args.source
    if args.source:
        source_ids = [args.source]
    else:
        source_ids = pending_extract_sources(repo_root)[: args.drain]
        if not source_ids:
            print("no pending extract tasks", file=sys.stderr)
            return

    text = Path(args.text_file).read_text(encoding="utf-8") if args.text_file else None
    if text is not None and len(source_ids) != 1:
        print("--text-file requires a single --source", file=sys.stderr)
        sys.exit(2)

    for sid in source_ids:
        try:
            summary = run_fable(
                repo_root,
                source_id=sid,
                run_id=args.run_id,
                model=args.model,
                assemble_model=args.assemble_model,
                text=text,
            )
        except Exception as exc:  # noqa: BLE001 - drain must survive bad sources
            if not drain_mode:
                raise
            # Refetch/engine failure on one source must not stall the queue:
            # flip its extract tasks skipped with the reason (WORKER.md rubric)
            # and move on.
            store.update_tasks(
                repo_root,
                lambda t, s=sid: (
                    t.get("status") == "pending"
                    and t.get("kind") == "extract"
                    and t.get("payload", {}).get("source_id") == s
                ),
                status="skipped",
                note=f"fable failed: {str(exc)[:160]}",
            )
            print(json.dumps({"source_id": sid, "skipped": str(exc)[:200]}))
            continue
        print(json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_tribunal(args: argparse.Namespace) -> None:
    from mole.tribunal import docket, try_question

    repo_root = Path(args.repo_root).resolve()
    if args.docket or not args.question:
        print(json.dumps(docket(repo_root), indent=2, ensure_ascii=False))
        return
    verdict = try_question(repo_root, args.question, args.run_id or "tribunal-manual")
    slim = {k: verdict[k] for k in
            ("id", "question_id", "credence", "bench_spread", "evidence_watermark", "status")}
    slim["rulings"] = [
        {k: r.get(k) for k in ("judge", "credence", "error") if k in r}
        for r in verdict["rulings"]
    ]
    print(json.dumps(slim, indent=2, ensure_ascii=False))


def _cmd_extract_backfill(args: argparse.Namespace) -> None:
    from mole import store

    repo_root = Path(args.repo_root).resolve()
    sources = store.load_all_sources(repo_root)
    tasked = {
        t["payload"].get("source_id")
        for t in store.load_all_tasks(repo_root)
        if t.get("kind") == "extract"
    }
    task_max = int(store.next_task_id(repo_root).split("_")[1]) - 1
    enqueued = 0
    for src in sources:
        sid = src.get("id", "")
        if args.feed_prefix and not src.get("feed", "").startswith(args.feed_prefix):
            continue
        if sid in tasked or src.get("fable_run"):
            continue
        task_max += 1
        store.append_task(
            repo_root,
            task_id=f"task_{task_max:06d}",
            kind="extract",
            payload={"source_id": sid},
            created_run=args.run_id,
        )
        enqueued += 1
    print(json.dumps({
        "sources": len(sources),
        "extract_enqueued": enqueued,
        "feed_prefix": args.feed_prefix,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mole",
        description="The mole: claim extraction and compilation for claimbase.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- run ----
    run_p = sub.add_parser("run", help="Ingest new items and extract claims.")
    run_p.add_argument(
        "--since",
        required=True,
        metavar="YYYY-MM-DD",
        help="Fetch items published on or after this date.",
    )
    run_p.add_argument(
        "--run-id",
        required=True,
        dest="run_id",
        help="Stable run identifier (e.g. 20260611).",
    )
    run_p.add_argument(
        "--repo-root",
        default=".",
        dest="repo_root",
        help="Path to the claimbase repo root (default: current directory).",
    )

    # ---- compile ----
    compile_p = sub.add_parser(
        "compile", help="Build data/artifact.json, data/atlas.json, and ATLAS.md."
    )
    compile_p.add_argument(
        "--repo-root",
        default=".",
        dest="repo_root",
        help="Path to the claimbase repo root (default: current directory).",
    )

    # ---- map ----
    map_p = sub.add_parser(
        "map", help="Print the fog-of-war atlas (ATLAS.md), building it if missing."
    )
    map_p.add_argument(
        "--repo-root",
        default=".",
        dest="repo_root",
        help="Path to the claimbase repo root (default: current directory).",
    )

    # ---- attach-backfill ----
    ab_p = sub.add_parser(
        "attach-backfill",
        help="Enqueue attach tasks for live claims not yet scored against the question layer.",
    )
    ab_p.add_argument("--run-id", required=True, dest="run_id",
                      help="Run id to stamp on enqueued tasks.")
    ab_p.add_argument("--repo-root", default=".", dest="repo_root",
                      help="Path to the claimbase repo root (default: current directory).")

    # ---- fable ----
    fable_p = sub.add_parser(
        "fable",
        help="Worker-side deep extraction (episteme-fable engine).",
    )
    fable_p.add_argument("--source", default=None,
                         help="Source id (src_...). Omit to drain pending extract tasks.")
    fable_p.add_argument("--drain", type=int, default=1,
                         help="How many pending extract tasks to process when --source is omitted (default 1).")
    fable_p.add_argument("--run-id", required=True, dest="run_id",
                         help="Run id to stamp on records and tasks (e.g. fable-20260713).")
    fable_p.add_argument("--model", default=None,
                         help="Propose model override (default: engine default, haiku).")
    fable_p.add_argument("--assemble-model", default=None, dest="assemble_model",
                         help="Assemble model override (default: engine default, sonnet).")
    fable_p.add_argument("--text-file", default=None, dest="text_file",
                         help="Read source text from a file instead of refetching (single --source only).")
    fable_p.add_argument("--repo-root", default=".", dest="repo_root",
                         help="Path to the claimbase repo root (default: current directory).")

    # ---- tribunal ----
    tr_p = sub.add_parser(
        "tribunal",
        help="Richtschwert: try a contested question (advocates -> bench -> verdict).",
    )
    tr_p.add_argument("--question", default=None, help="question id (q_...); omit for --docket view")
    tr_p.add_argument("--docket", action="store_true", help="list contested questions + verdict state")
    tr_p.add_argument("--run-id", default=None, dest="run_id")
    tr_p.add_argument("--repo-root", default=".", dest="repo_root")

    # ---- extract-backfill ----
    eb_p = sub.add_parser(
        "extract-backfill",
        help="Enqueue extract tasks for existing sources not yet deep-extracted.",
    )
    eb_p.add_argument("--run-id", required=True, dest="run_id",
                      help="Run id to stamp on enqueued tasks.")
    eb_p.add_argument("--feed-prefix", default=None, dest="feed_prefix",
                      help="Only sources whose feed starts with this prefix (e.g. arxiv).")
    eb_p.add_argument("--repo-root", default=".", dest="repo_root",
                      help="Path to the claimbase repo root (default: current directory).")

    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "compile":
        _cmd_compile(args)
    elif args.command == "map":
        _cmd_map(args)
    elif args.command == "attach-backfill":
        _cmd_attach_backfill(args)
    elif args.command == "fable":
        _cmd_fable(args)
    elif args.command == "extract-backfill":
        _cmd_extract_backfill(args)
    elif args.command == "tribunal":
        _cmd_tribunal(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
