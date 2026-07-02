"""
CLI entry point: python -m mole <command> [options]

Commands:
  run     Fetch new items, extract claims, enqueue tasks, rebuild artifacts.
  compile Build data/artifact.json, data/atlas.json, and ATLAS.md.
  map     Print the fog-of-war atlas (ATLAS.md), building it if missing.
  attach-backfill
          Enqueue 'attach' tasks for live claims the question layer has not
          seen yet (one-off after adding/expanding data/questions.jsonl).
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

    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "compile":
        _cmd_compile(args)
    elif args.command == "map":
        _cmd_map(args)
    elif args.command == "attach-backfill":
        _cmd_attach_backfill(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
