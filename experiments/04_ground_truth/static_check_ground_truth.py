"""Non-executing source checks: stdlib only, no experiment imports or calls."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUP = PROJECT_ROOT / "experiments" / "04_ground_truth"


def _literal_assignments(tree: ast.Module) -> dict:
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


def _calls(tree: ast.AST) -> set[str]:
    return {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))}


def check_sources() -> dict:
    paths = sorted(GROUP.glob("*.py")) + [
        PROJECT_ROOT / "experiments/01_detection/proposed_detection.py",
        PROJECT_ROOT / "experiments/common/plotting_style.py",
        PROJECT_ROOT / "experiments/common/dataset_adapter.py",
    ]
    trees = {path: ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path)) for path in paths}
    constants = _literal_assignments(trees[GROUP / "config_ground_truth.py"])
    expected = {"N": 3000, "GLOBAL_CR_PERCENT": 3.0, "D": 30, "H": 0.5,
                "INITIAL_K": 50, "GT_SIZE_M": 25, "DELTA_K": 25,
                "MAX_CANDIDATE_POOL_SIZE": 100, "H0": 0.10,
                "TAU_LOWER": 0.997, "TAU_UPPER": 1.0, "REPETITIONS": 1000,
                "GT_QUORUM_NUMERATOR": 2, "GT_QUORUM_DENOMINATOR": 3,
                "GT_SELECTION_POLICY": "history_consensus",
                "K_VALUES": (50, 75, 100),
                "METHODS": ("RACA-HIM", "Fixed-K Majority", "Top-M", "Random-M")}
    if any(constants.get(name) != value for name, value in expected.items()):
        raise ValueError("Static protocol literals differ from the supplied formal settings")
    for filename in ("run_ground_truth.py", "plot_ground_truth.py"):
        tree = trees[GROUP / filename]
        if not any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body):
            raise ValueError(f"Missing direct-run main function: {filename}")
        if not any(isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                   and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
                   for node in tree.body):
            raise ValueError(f"Missing direct-run guard: {filename}")
    kernel = trees[PROJECT_ROOT / "experiments/01_detection/proposed_detection.py"]
    function = next(node for node in kernel.body if isinstance(node, ast.FunctionDef)
                    and node.name == "build_ground_truth")
    if tuple(argument.arg for argument in function.args.kwonlyargs) != (
            "responder_available", "pairwise_cache", "allow_expansion", "support_quorum", "selection_policy"):
        raise ValueError("Shared GT optional interface is inconsistent")
    forbidden = {"build_ground_truth", "run_method", "run_experiment", "load_group", "draw_candidate_set",
                 "detect_proposed", "cosine_pairwise", "fill_between", "interp", "interp1d"}
    for filename in ("plot_ground_truth.py", "ground_truth_storage.py"):
        if _calls(trees[GROUP / filename]) & forbidden:
            raise ValueError(f"Saved-result path contains experiment or smoothing calls: {filename}")
    methods = trees[GROUP / "ground_truth_methods.py"]
    entry = next(node for node in methods.body if isinstance(node, ast.FunctionDef) and node.name == "run_method")
    if any(arg.arg in {"labels", "binary_label", "subtype_label", "prototype", "scene", "sample_id"}
           for arg in entry.args.args):
        raise ValueError("Method entry exposes a prohibited evaluation input")
    config = trees[GROUP / "config_ground_truth.py"]
    ranges = {tuple(ast.literal_eval(arg) for arg in node.args) for node in ast.walk(config)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range"
              and all(isinstance(arg, ast.Constant) for arg in node.args)}
    if not {(0, 51, 5), (0, 21, 5), (2026, 3026)} <= ranges:
        raise ValueError("Formal q/failure/seed source grids are incomplete")
    return {"status": "passed", "kind": "AST and source-only consistency",
            "files_checked": [path.relative_to(PROJECT_ROOT).as_posix() for path in paths],
            "experiment_executed": False, "tests_executed": False, "plotting_executed": False,
            "limits": "Does not establish runtime, numerical, regression, or rendered-layout correctness"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(check_sources(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
