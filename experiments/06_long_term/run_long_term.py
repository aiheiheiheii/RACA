"""PyCharm entry point for Group 06: seed 2026, one repetition.

Importing this file never runs an experiment. Plot-only mode imports only the
saved-result reader, never simulation or method implementations.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Group 06: seed 2026 once; 5 scenarios, 3 methods, 15 trajectories, 3000 method-round records")
    parser.add_argument("--plots-only", action="store_true", help="Read saved summary; no algorithms")
    parser.add_argument("--force", action="store_true", help="Recompute the current 15 single-run checkpoints; previous run manifests are preserved")
    parser.add_argument("--legacy", action="store_true", help="With --plots-only, read archived adapted results")
    parser.add_argument("--result-root", type=Path, help="With --plots-only, select a saved result directory")
    args = parser.parse_args(argv)
    if args.plots_only and args.force:
        parser.error("--force cannot be combined with --plots-only")
    if not args.plots_only and (args.legacy or args.result_root is not None):
        parser.error("--legacy and --result-root require --plots-only")
    if args.plots_only:
        module = importlib.import_module("experiments.06_long_term.plot_long_term")
        plot_args = ["--legacy"] if args.legacy else []
        if args.result_root is not None:
            plot_args.extend(["--result-root", str(args.result_root)])
        module.main(plot_args)
    else:
        module = importlib.import_module("experiments.06_long_term.long_term_experiment")
        module.run_experiment(force=args.force)


if __name__ == "__main__":
    main()
