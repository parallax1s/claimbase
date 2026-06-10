"""
CLI entry point: python -m mole <command> [options]

Commands:
  run     Fetch new items, extract claims, enqueue tasks.
  compile Build data/artifact.json from current data files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_run(args: argparse.Namespace) -> None:
    from mole.pipeline import run

    repo_root = Path(args.repo_root).resolve()
    summary = run(repo_root=repo_root, since=args.since, run_id=args.run_id)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_compile(args: argparse.Namespace) -> None:
    from mole.compile import compile

    repo_root = Path(args.repo_root).resolve()
    artifact = compile(repo_root)
    # Print a brief summary (the full artifact is written to disk)
    print(
        json.dumps(
            {
                "generated_run": artifact["generated_run"],
                "counts": artifact["counts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


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
    compile_p = sub.add_parser("compile", help="Build data/artifact.json.")
    compile_p.add_argument(
        "--repo-root",
        default=".",
        dest="repo_root",
        help="Path to the claimbase repo root (default: current directory).",
    )

    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "compile":
        _cmd_compile(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
