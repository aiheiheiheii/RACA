"""AST/source checks only; never import or execute an experiment module.

This intentionally does not establish runtime correctness or a proof of causal
isolation.  Compilation creates a code object and immediately discards it.
"""
from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = (
    "run_long_term.py", "config_long_term.py", "long_term_experiment.py",
    "dynamic_simulator.py", "long_term_methods.py", "long_term_metrics.py",
    "plot_long_term.py", "config_native_long_term.py", "native_iedi.py", "native_edi_v.py",
)
FORBIDDEN_POLICY_FIELDS = {
    "corrupted_mask", "event_model_ids", "onset_round", "recovery_round",
    "fail_uniform", "prototype", "normal_X", "corrupted_event_X",
    "binary_label", "subtype_label", "sample_id", "cosine_to_prototype",
}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _call_name(node.value) + "." + node.attr
    return ""


def main() -> None:
    missing = [name for name in REQUIRED_FILES if not (SOURCE_ROOT / name).is_file()]
    if missing:
        raise ValueError(f"Missing required Group 06 source files: {missing}")
    paths = sorted(SOURCE_ROOT.glob("*.py"))
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        compile(tree, str(path), "exec")
        if path.name == "long_term_experiment.py":
            source_list = next((node.value for node in tree.body
                                if isinstance(node, ast.Assign)
                                and any(isinstance(target, ast.Name) and target.id == "SOURCE_FILES"
                                        for target in node.targets)), None)
            if source_list is None:
                raise ValueError("Native driver must declare its scientific source closure")
            scientific_sources = set(ast.literal_eval(source_list))
            if not {"experiments/06_long_term/config_native_long_term.py",
                    "experiments/06_long_term/native_iedi.py",
                    "experiments/06_long_term/native_edi_v.py"}.issubset(scientific_sources):
                raise ValueError("Native protocol/config sources are absent from computational fingerprints")
        if path.name in {"run_long_term.py", "plot_long_term.py"}:
            if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
                       for node in tree.body):
                raise ValueError(f"Missing direct-entry main function: {path.name}")
            if not any(isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                       and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
                       for node in tree.body):
                raise ValueError(f"Missing direct-entry __main__ guard: {path.name}")
        for node in ast.walk(tree):
            location = f"{path.name}:{getattr(node, 'lineno', '?')}"
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                sibling = SOURCE_ROOT / (node.module.split(".")[0] + ".py")
                if not sibling.is_file():
                    raise ValueError(f"Missing sibling import at {location}")
            elif isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None:
                for alias in node.names:
                    if not (SOURCE_ROOT / (alias.name + ".py")).is_file():
                        raise ValueError(f"Missing sibling imported module at {location}: {alias.name}")
            if path.name == "long_term_methods.py":
                if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                    if node.slice.value in FORBIDDEN_POLICY_FIELDS:
                        raise ValueError(f"Forbidden event/truth field access in policy at {location}")
                if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_POLICY_FIELDS:
                    raise ValueError(f"Forbidden event/truth attribute in policy at {location}")
            if path.name == "plot_long_term.py":
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                if any(any(token in module for token in (
                        "load_dataset", "long_term_experiment", "dynamic_simulator",
                        "long_term_methods", "long_term_metrics", "joint_methods", "ground_truth"))
                       for module in modules):
                    raise ValueError(f"Plot-only source imports experiment execution at {location}")
                if isinstance(node, ast.Call):
                    name = _call_name(node.func)
                    if name.endswith((".fill_between", ".fill_betweenx", ".errorbar")):
                        raise ValueError(f"Figure L1 must retain mean curves without CI bands at {location}")
                    if name.endswith(".import_module"):
                        if (not node.args or not isinstance(node.args[0], ast.Constant)
                                or node.args[0].value not in {
                                    "experiments.06_long_term.config_long_term",
                                    "experiments.06_long_term.config_native_long_term"}):
                            raise ValueError(f"Plot-only dynamic import must be declarative configuration at {location}")
    print(f"Static AST/compile/source checks completed for {len(paths)} Group 06 Python files. "
          "No project module was imported; no algorithm, test or figure was executed.")


if __name__ == "__main__":
    main()
