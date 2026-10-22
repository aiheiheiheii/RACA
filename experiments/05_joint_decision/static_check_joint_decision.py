"""Parse Group 05 source without importing project code or running algorithms."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FORBIDDEN_FIELDS = {"binary_label", "subtype_label", "prototype", "cosine_to_prototype", "sample_id"}


def main() -> None:
    paths = sorted(ROOT.glob("*.py"))
    for path in paths:
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        compile(tree, str(path), "exec")  # Syntax checking only; the code object is never executed.
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                sibling = ROOT / (node.module.split(".")[0] + ".py")
                if not sibling.is_file():
                    raise ValueError(f"Missing relative source module: {path.name}:{node.lineno}")
            if path.name in {"historical_state.py", "joint_inputs.py", "joint_methods.py", "joint_objective.py"}:
                if (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                        and node.slice.value in FORBIDDEN_FIELDS):
                    raise ValueError(f"Forbidden dataset field access: {path.name}:{node.lineno}")
            if path.name == "plot_joint_decision.py":
                if isinstance(node, ast.ImportFrom) and any(name in (node.module or "") for name in (
                        "joint_experiment", "joint_methods", "joint_inputs", "load_dataset")):
                    raise ValueError("The plot-only entry must not import algorithms or dataset execution")
    print(f"Static AST/compile checks completed for {len(paths)} source files; no project module was imported.")


if __name__ == "__main__":
    main()
