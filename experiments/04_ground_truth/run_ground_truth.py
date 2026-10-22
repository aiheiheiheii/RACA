"""Right-click Run in PyCharm to execute all 55000 formal Group 04 scenes."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULT_ROOT = PROJECT_ROOT / "results" / "04_ground_truth"
sys.dont_write_bytecode = True
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Full Ground-Truth Construction Correctness and Robustness experiment")
    parser.add_argument("--plots-only", action="store_true", help="Replot saved CSVs and manifest without GT execution")
    args = parser.parse_args(argv)
    result_root = RESULT_ROOT.resolve()
    started = time.perf_counter()
    if not args.plots_only:
        experiment = importlib.import_module("experiments.04_ground_truth.ground_truth_experiment")
        experiment.run_experiment(result_root, DATASET_ROOT)
    plotter = importlib.import_module("experiments.04_ground_truth.plot_ground_truth")
    storage = importlib.import_module("experiments.04_ground_truth.ground_truth_storage")
    path = result_root / "logs" / "run_manifest.json"
    try:
        figure = plotter.plot_ground_truth()
    except BaseException as exc:
        # Complete computation evidence can be rendered again later.
        if path.is_file():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest.update(figures_status="failed", figure_error={"type": type(exc).__name__, "message": str(exc)})
            storage.atomic_json(path, manifest)
        raise
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(figures_status="complete", figures=figure)
    manifest.pop("figure_error", None)
    if not args.plots_only:
        manifest["total_entry_elapsed_seconds"] = time.perf_counter() - started
    storage.atomic_json(path, manifest)
    print(f"Saved Group 04 result evidence at {result_root}", flush=True)
    print(f"Saved {figure['figure_name']} PDF/PNG at {figure['figure_output_root']}", flush=True)


if __name__ == "__main__":
    main()
