"""Read-only audit of the supplied descriptor banks and all 729 lazy groups.

This deliberately does not import or execute dataset/validate_dataset.py,
whose ordinary entry point rewrites the dataset validation report and manifest.
All outputs here are written to the project reports directory instead.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import sys
import time

# Set before any project imports, including when run without the recommended -B.
sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from threadpoolctl import threadpool_limits

from dataset.load_dataset import BANK_COUNTS, SUBTYPES, enumerate_configs, load_group
from experiments.common.reproducibility import dataset_snapshot, verify_manifest


def _half_up(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def validate(dataset_root: Path, report_root: Path) -> dict:
    """Validate stored bytes and logical views without modifying the source tree."""

    started = time.perf_counter()
    dataset_root = dataset_root.resolve()
    before = dataset_snapshot(dataset_root)
    manifest_before = verify_manifest(dataset_root)
    outcomes: list[str] = []
    failures: list[str] = []
    banks: list[dict] = []
    groups: list[dict] = []

    def check(condition, description: str) -> None:
        (outcomes if bool(condition) else failures).append(description)

    check(manifest_before["verified"] == 30, "Original manifest verifies all 30 listed files")
    check(not manifest_before["missing"] and not manifest_before["mismatch"],
          "Original manifest has no missing files or hash mismatches")
    config = json.loads((dataset_root / "config.json").read_text(encoding="utf-8"))
    check(config["expected_virtual_groups"] == 729, "Dataset advertises 729 lazy configurations")
    with (dataset_root / "dataset_summary.csv").open(encoding="utf-8", newline="") as file:
        indexed = list(csv.DictReader(file))
    check(len(indexed) == 729, "Dataset summary has 729 configurations")
    check(len({row["group_id"] for row in indexed}) == 729, "Summary group identifiers are unique")
    check(all(row["status"] == "bank_view" for row in indexed), "Summary records are lazy bank views")

    with threadpool_limits(limits=1):
        for hindex in range(11):
            path = dataset_root / "banks" / f"bank_H_{hindex:02d}.npz"
            with np.load(path, allow_pickle=False) as bank:
                X = bank["X_50"]
                binary = bank["binary_label"]
                subtype = bank["subtype_label"]
                check(X.shape == (9000, 50), f"Bank H={hindex/10:.1f}: shape 9000x50")
                check(X.dtype == np.float32, f"Bank H={hindex/10:.1f}: float32 descriptors")
                check(np.isfinite(X).all(), f"Bank H={hindex/10:.1f}: finite descriptors")
                norms = np.linalg.norm(X.astype(np.float64), axis=1)
                check(np.all(norms > 0), f"Bank H={hindex/10:.1f}: nonzero descriptors")
                norm_error = float(np.max(np.abs(norms - 1)))
                check(norm_error < 4e-6, f"Bank H={hindex/10:.1f}: unit-normalized descriptors")
                check(binary.shape == (9000,) and subtype.shape == (9000,),
                      f"Bank H={hindex/10:.1f}: label dimensions")
                check(np.isin(binary, (0, 1)).all(), f"Bank H={hindex/10:.1f}: binary encoding")
                check(np.isin(subtype, (0, 1, 2, 3)).all(), f"Bank H={hindex/10:.1f}: subtype encoding")
                check(np.array_equal(binary, (subtype >= 2).astype(binary.dtype)),
                      f"Bank H={hindex/10:.1f}: binary/subtype consistency")
                counts = np.bincount(subtype.astype(np.int64), minlength=4)
                check(tuple(counts.tolist()) == BANK_COUNTS,
                      f"Bank H={hindex/10:.1f}: subtype bank quotas")
                banks.append({"H": hindex / 10, "shape": list(X.shape), "dtype": str(X.dtype),
                              "max_unit_norm_error": norm_error, "minimum_norm": float(norms.min()),
                              "subtype_counts": dict(zip(SUBTYPES, counts.tolist()))})

        configs = list(enumerate_configs())
        check(len(configs) == 729 and len(set(configs)) == 729, "Loader enumerates 729 distinct groups")
        total_rows = 0
        max_norm_error = 0.0
        for position, (n, cr, dim) in enumerate(configs, 1):
            group = load_group(dataset_root, n, cr, dim, H=0.5, seed=2026)
            X = group["X"]
            binary = group["binary_label"]
            subtype = group["subtype_label"]
            tag = f"N={n},cr={cr:.1f},d={dim},H=0.5,seed=2026"
            check(X.shape == (n, dim), f"Group {tag}: requested shape")
            check(X.dtype == np.float32, f"Group {tag}: float32 descriptors")
            check(np.isfinite(X).all(), f"Group {tag}: finite descriptors")
            norms = np.linalg.norm(X.astype(np.float64), axis=1)
            check(np.all(norms > 0), f"Group {tag}: nonzero descriptors")
            norm_error = float(np.max(np.abs(norms - 1)))
            max_norm_error = max(max_norm_error, norm_error)
            check(norm_error < 4e-6, f"Group {tag}: unit-normalized descriptors")
            check(binary.shape == (n,) and subtype.shape == (n,), f"Group {tag}: label dimensions")
            check(np.isin(binary, (0, 1)).all(), f"Group {tag}: binary encoding")
            check(np.isin(subtype, (0, 1, 2, 3)).all(), f"Group {tag}: subtype encoding")
            check(np.array_equal(binary, (subtype >= 2).astype(binary.dtype)),
                  f"Group {tag}: binary/subtype consistency")
            corrupt = _half_up(Decimal(n) * Decimal(str(cr)) / Decimal(100))
            normal = n - corrupt
            high = _half_up(Decimal(normal) * Decimal("0.20"))
            mild = _half_up(Decimal(corrupt) * Decimal("0.20"))
            expected = (normal - high, high, corrupt - mild, mild)
            actual = tuple(np.bincount(subtype.astype(np.int64), minlength=4).tolist())
            check(int(binary.sum()) == corrupt, f"Group {tag}: half-up corruption quota")
            check(actual == expected, f"Group {tag}: half-up subtype quotas")
            check(group["counts"] == dict(zip(SUBTYPES, expected)), f"Group {tag}: returned counts agree")
            ids = group["sample_id"]
            check(ids.shape == (n,) and len(np.unique(ids)) == n,
                  f"Group {tag}: unique source samples within logical group")
            check(np.all((ids >= 0) & (ids < 9000)), f"Group {tag}: source sample bounds")
            check(group["tau_lower"] == 0.997 and group["tau_upper"] == 1.0,
                  f"Group {tag}: default similarity interval")
            groups.append({"N": n, "cr": cr, "d": dim, "H": 0.5, "seed": 2026,
                           "corrupt_count": corrupt, "realized_cr_percent": 100 * corrupt / n,
                           "subtype_counts": dict(zip(SUBTYPES, actual)),
                           "max_unit_norm_error": norm_error})
            total_rows += n
            if position % 81 == 0:
                print(f"Read-only dataset audit: {position}/729 lazy groups checked", flush=True)

        first = load_group(dataset_root, 3000, 3.0, 30, H=0.5, seed=2026)
        repeated = load_group(dataset_root, 3000, 3.0, 30, H=0.5, seed=2026)
        array_keys = ("X", "binary_label", "subtype_label", "sample_id", "prototype", "cosine_to_prototype")
        check(all(first[key].dtype == repeated[key].dtype and
                  first[key].shape == repeated[key].shape and
                  first[key].tobytes() == repeated[key].tobytes() for key in array_keys),
              "Same seed and parameters return byte-identical descriptor and metadata arrays")
        different = load_group(dataset_root, 3000, 3.0, 30, H=0.5, seed=2027)
        check(not np.array_equal(first["sample_id"], different["sample_id"]),
              "Changing seed changes fixed-bank sampling")

    manifest_after = verify_manifest(dataset_root)
    after = dataset_snapshot(dataset_root)
    check(manifest_after["verified"] == 30 and not manifest_after["missing"] and not manifest_after["mismatch"],
          "Final manifest still verifies all 30 listed files")
    check(before == after, "Complete dataset file tree and every SHA256 are unchanged")
    changes = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    payload = {
        "result": "PASS" if not failures else "FAIL", "read_only": True,
        "dataset_root": str(dataset_root), "banks_checked": len(banks),
        "canonical_bank_rows": sum(row["shape"][0] for row in banks),
        "lazy_groups_checked": len(groups), "lazy_group_rows_checked": total_rows,
        "lazy_grid_H": 0.5, "lazy_grid_seed": 2026,
        "assertions_passed": len(outcomes), "assertions_failed": len(failures),
        "max_lazy_group_unit_norm_error": max_norm_error,
        "elapsed_seconds": time.perf_counter() - started,
        "manifest_before": manifest_before, "manifest_after": manifest_after,
        "dataset_file_count_before": len(before), "dataset_file_count_after": len(after),
        "all_hashes_unchanged": before == after, "changed_paths": changes,
        "hashes_before": before, "hashes_after": after,
        "bank_diagnostics": banks, "group_diagnostics": groups,
        "passed_assertions": outcomes, "failed_assertions": failures,
        "limitations": [
            "The data are synthetic descriptors, not extracted pretrained/quantized models.",
            "The 99,000 stored H-indexed records represent 9,000 paired source identities.",
            "729 lazy views and their aggregate rows are not independent datasets or samples.",
            "This checks data integrity, not algorithm classification accuracy or GT theorem assumptions.",
            "Original dataset/validate_dataset.py is not executed; no source dataset files are written.",
        ],
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "01_detection_dataset_checks.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# 第一组数据集只读检查", "", f"结果：**{payload['result']}**。",
             f"通过 {len(outcomes)} 项断言，失败 {len(failures)} 项。", "",
             "- 11 个真实存储的 bank：每个 9000×50、float32、有限、非零、单位范数；标签编码及四类配额一致。",
             "- 729 个逻辑配置：9N×9cr×9d，固定 H=0.5、seed=2026；逐个检查形状、精度、有限值、非零、范数及标签。",
             f"- 总共检查 {total_rows:,} 个逻辑组数据行；这些行存在重复来源，不能当作独立样本。",
             "- 损坏及 subtype 数量依据 ROUND_HALF_UP 重算，全部与返回标签、counts 一致。",
             "- 相同配置与种子的描述符、标签、来源 ID 及诊断数组逐字节一致；不同 seed 改变抽样。",
             f"- 最大逻辑组单位范数误差：{max_norm_error:.3e}，要求 <4e-6。",
             f"- MANIFEST 前后各通过 {manifest_before['verified']} / {manifest_after['verified']} 个文件校验。",
             f"- 数据集文件树前后均为 {len(before)} 个文件；全部文件 SHA256 保持一致：{before == after}。",
             "- 未执行会重写 dataset 的原始 validator；本检查只输出至 reports/。", "",
             "完整逐项断言、729组诊断、11库诊断及前后 SHA256 见同名 JSON。", "",
             "注意：此结果证明数据结构与来源文件完整性，不证明真实部署准确率或 GT 定理的可分性假设。",
             "数据完全合成，99,000 条 H-indexed bank 记录对应 9,000 个配对来源 ID。", ""]
    if failures:
        lines.extend(["失败项：", "", *[f"- {failure}" for failure in failures], ""])
    (report_root / "01_detection_dataset_checks.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--report-root", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()
    payload = validate(args.dataset_root, args.report_root)
    print(f"Dataset {payload['result']}: {payload['banks_checked']} banks, "
          f"{payload['lazy_groups_checked']} groups, {payload['assertions_passed']} assertions; "
          f"all hashes unchanged={payload['all_hashes_unchanged']}", flush=True)
    if payload["result"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
