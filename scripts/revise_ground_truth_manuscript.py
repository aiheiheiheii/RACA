"""Prepare the user-authorized GT-method manuscript revision inside the workspace.

Text processing only. No experiment, GT code, tests, or plotting is imported.
The original attached manuscript is never written by this staging script.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("C:/Users/MengKe/Desktop/Tutu/TNSE/Latest version of the paper.md")
REVISION = PROJECT_ROOT / "reports/04_ground_truth_method_revision.md"
DESTINATION = PROJECT_ROOT / "reports/Latest_version_of_paper_GT_v2.md"


def _between(text: str, beginning: str, ending: str) -> str:
    if text.count(beginning) != 1 or text.count(ending) != 1:
        raise ValueError(f"Ambiguous manuscript section markers: {beginning} / {ending}")
    return text.split(beginning, 1)[1].split(ending, 1)[0].strip()


def revised_manuscript(source: str, revision: str) -> str:
    section = _between(revision, "## 可替换稿件 Section IV-A 的内容", "## 可替换稿件定理 1 与证明的内容")
    theorem = _between(revision, "## 可替换稿件定理 1 与证明的内容", "## 复杂度补充")
    complexity = _between(revision, "## 复杂度补充", "## 版本与后续实验")
    first, second = "### Collaborative Ground-Truth Determination", "### Model Verification Risk Estimation"
    _between(source, first, second)
    # Slicing by positions preserves unrelated manuscript whitespace/content.
    start, stop = source.index(first), source.index(second)
    output = source[:start] + section + "\n\n" + source[stop:]
    theorem_start = output.index("**定理 1.**")
    theorem_stop = output.index("令 $J^{(k)}(t)", theorem_start)
    output = output[:theorem_start] + theorem + "\n\n" + output[theorem_stop:]
    old_complexity_marker = ("对于规模为 $K$ 的 Ground-Truth candidate set" if "对于规模为 $K$ 的 Ground-Truth candidate set" in output
                             else "候选之间仍至多执行 $K(K-1)$")
    old_complexity_start = output.index(old_complexity_marker)
    next_complexity_start = output.index("对于 Joint Verification Decision", old_complexity_start)
    output = output[:old_complexity_start] + complexity + "\n\n" + output[next_complexity_start:]
    output = output.replace("在 Ground-Truth determination 中通过多数一致性检验", "在 Ground-Truth determination 中通过三分之二支持检验")
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Stage the revised GT manuscript without changing the original")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--revision", type=Path, default=REVISION)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args(argv)
    destination = args.destination.resolve()
    if not destination.is_relative_to(PROJECT_ROOT):
        raise ValueError("Stage the reviewable manuscript inside the project workspace")
    source = args.source.read_text(encoding="utf-8-sig")
    revision = args.revision.read_text(encoding="utf-8-sig")
    output = revised_manuscript(source, revision)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(output, encoding="utf-8")
    print(f"Staged manuscript: {destination}")
    print(f"SHA256: {hashlib.sha256(destination.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
