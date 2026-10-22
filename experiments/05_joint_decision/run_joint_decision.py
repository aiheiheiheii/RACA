"""PyCharm entry: resume matching runs; archive and restart changed versions."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plots-only", action="store_true", help="Read saved CSVs and redraw J1/J2")
    mode.add_argument("--force", action="store_true", help="Archive previous exports and start a new full run")
    args = parser.parse_args()
    if args.plots_only:
        module = importlib.import_module("experiments.05_joint_decision.plot_joint_decision")
        module.plot_joint_decision()
    else:
        module = importlib.import_module("experiments.05_joint_decision.joint_experiment")
        module.run_experiment(force=args.force)


if __name__ == "__main__":
    main()
