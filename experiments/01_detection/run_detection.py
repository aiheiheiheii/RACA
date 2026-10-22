"""Right-click Run in PyCharm: the complete 65-configuration, 30-seed experiment."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
# Prevent imports from creating any files inside the immutable dataset directory.
sys.dont_write_bytecode = True
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="TNSE Group 01 corruption detection")
    parser.add_argument("--output-dir", type=Path, help="Optional output root for this run")
    parser.add_argument("--plot-kind", choices=("bar", "line"), default="bar",
                        help="Main-figure style; bar follows the comparable TMC experiments")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir or PROJECT_ROOT / "results" / "01_detection"
    try:
        module = importlib.import_module("experiments.01_detection.detection_experiment")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing experiment dependency: {exc.name}. In this project's interpreter run "
            f'python -m pip install -r "{PROJECT_ROOT / "requirements.txt"}"'
        ) from exc
    module.run_experiment(output_dir=output, plot_kind=args.plot_kind)


if __name__ == "__main__":
    main()
