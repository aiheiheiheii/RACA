"""Replot J1/J2 from authenticated saved CSV statistics only.

This entry can be right-clicked in PyCharm independently of the experiment
runner.  It never imports the dataset loader, GT construction, scheduling
methods, or experiment driver.  Missing paired statistics remain line gaps.
The CSV retains confidence intervals; the paper figures draw means only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from experiments.common.plotting_style import (
    apply_style, method_style, PNG_DPI, DETECTION_LEGEND_GAP_PT,
    DETECTION_FIGURE_WIDTH, DETECTION_FIGURE_HEIGHT,
    DETECTION_PANEL_WIDTH_PT, DETECTION_PANEL_GAP_PT,
    DETECTION_UPPER_HEIGHT_PT, DETECTION_LOWER_HEIGHT_PT,
    DETECTION_BREAK_GAP_PT, DETECTION_AXES_LEFT_PT,
    DETECTION_AXES_BOTTOM_PT, DETECTION_CAPTION_CENTER_PT,
    DETECTION_YLABEL_CENTER_X_PT, DETECTION_TICK_FONT_PT,
    DETECTION_LABEL_FONT_PT,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

_config = importlib.import_module("experiments.05_joint_decision.config_joint_decision")
RESULT_ROOT = _config.RESULT_ROOT
N_VALUES = tuple(_config.N_VALUES)
BETA_PANEL_VALUES = tuple(_config.BETA_PANEL_VALUES)
SIGMA_PANEL_VALUES = tuple(_config.SIGMA_PANEL_VALUES)
METHODS = tuple(_config.METHODS)
SEEDS = tuple(_config.SEEDS)
REPETITIONS = _config.REPETITIONS
ALGORITHM_VERSION = _config.ALGORITHM_VERSION

# Fixed values belong to the published formal grid, not to a plotting mode.
J1_SIGMA = _config.J1_SIGMA
J2_BETA_B = _config.J2_BETA_B
MIN_PAIRED_TRIALS = _config.MIN_PAIRED_TRIALS
FIGURES = {
    "J1": "Fig_J1_joint_utility_budget",
    "J2": "Fig_J2_joint_utility_heterogeneity",
}
SUMMARY_STATISTICS = (
    "mean_joint_utility", "std_joint_utility", "ci95_lower", "ci95_upper",
    "mean_selected_count", "mean_risk_reward", "mean_makespan_ms",
)
SUMMARY_COUNTS = ("n_total", "n_valid", "n_failed", "n_paired")
IDENTITY_FIELDS = ("run_id", "code_version", "algorithm_version", "dataset_fingerprint")


def _display_name(method: str) -> str:
    return "DVA-P" if method == "DVA-P-Adapted" else method


def _file_identity(path: Path, root: Path) -> dict:
    payload = path.read_bytes()
    return {"path": path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _read_sources(result_root: Path) -> tuple[list[dict], dict, dict]:
    """Pin a single summary byte sequence to the completed run manifest."""
    manifest_path = result_root / "logs" / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No saved run manifest: {manifest_path}. Run run_joint_decision.py first.")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    if manifest.get("status") != "complete":
        raise ValueError("The saved joint-decision run is incomplete; resume the formal run before replotting.")
    if manifest.get("algorithm_version") != ALGORITHM_VERSION:
        raise ValueError("The saved summary belongs to a different joint-decision algorithm version.")
    if (list(manifest.get("methods", ())) != list(METHODS)
            or list(manifest.get("seeds", ())) != list(SEEDS)):
        raise ValueError("The saved run must declare the same five methods and all 30 formal seeds.")
    if manifest.get("summary_policy") != "paired_all_five_methods":
        raise ValueError("J1/J2 require means over trials in which all five methods are valid.")
    if any(not manifest.get(field) for field in IDENTITY_FIELDS):
        raise ValueError("The saved run is missing a source/configuration/data identity.")
    identity = manifest.get("files", {}).get("joint_summary")
    if not isinstance(identity, dict):
        raise ValueError("The run manifest must authenticate files.joint_summary.")
    summary_path = result_root / "summary" / "joint_summary.csv"
    declared_path = Path(str(identity.get("path", "")))
    if (declared_path.is_absolute()
            or (result_root / declared_path).resolve() != summary_path.resolve()):
        raise ValueError("The manifest must identify summary/joint_summary.csv relative to the result root.")
    if not summary_path.is_file():
        raise FileNotFoundError(f"No saved joint summary: {summary_path}")
    payload = summary_path.read_bytes()
    actual = {"path": "summary/joint_summary.csv", "size_bytes": len(payload),
              "sha256": hashlib.sha256(payload).hexdigest()}
    if identity.get("sha256") != actual["sha256"] or identity.get("size_bytes") != actual["size_bytes"]:
        raise ValueError("The joint summary differs from the completed run's recorded bytes; resume the run.")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
    required = {"figure", "N", "beta_B", "sigma", "method",
                *SUMMARY_STATISTICS, *SUMMARY_COUNTS, *IDENTITY_FIELDS}
    if not required.issubset(set(reader.fieldnames or ())):
        raise ValueError(f"Joint summary is missing columns: {sorted(required - set(reader.fieldnames or ()))}")
    rows = list(reader)
    sources = {"joint_summary": actual,
               "run_manifest": {"path": "logs/run_manifest.json", "size_bytes": len(manifest_bytes),
                                "sha256": hashlib.sha256(manifest_bytes).hexdigest()}}
    return rows, manifest, sources


def _number(value: str | float | None) -> float:
    return math.nan if value is None or str(value).strip() == "" else float(value)


def _summary_index(rows: list[dict], manifest: dict) -> dict:
    """Validate the formal grid and paired counts without reconstructing data."""
    if (len(N_VALUES) != 9 or len(BETA_PANEL_VALUES) != 5
            or len(SIGMA_PANEL_VALUES) != 5 or len(METHODS) != 5
            or REPETITIONS != 30 or len(SEEDS) != REPETITIONS):
        raise ValueError("This paper figure requires the full 9-N, 5-panel, 5-method, 30-seed configuration.")
    expected = {
        ("J1", int(n), float(beta), float(J1_SIGMA), method)
        for n in N_VALUES for beta in BETA_PANEL_VALUES for method in METHODS
    } | {
        ("J2", int(n), float(J2_BETA_B), float(sigma), method)
        for n in N_VALUES for sigma in SIGMA_PANEL_VALUES for method in METHODS
    }
    indexed = {}
    for source_row in rows:
        row = dict(source_row)
        key = (row["figure"], int(row["N"]), float(row["beta_B"]),
               float(row["sigma"]), row["method"])
        if key in indexed:
            raise ValueError(f"Duplicate joint summary row: {key}")
        if key not in expected:
            raise ValueError(f"Unexpected joint summary configuration: {key}")
        if any(row[field] != str(manifest[field]) for field in IDENTITY_FIELDS):
            raise ValueError(f"Mixed source/configuration/data identities in summary: {key}")
        counts = {field: int(row[field]) for field in SUMMARY_COUNTS}
        if (counts["n_total"] != REPETITIONS
                or not 0 <= counts["n_paired"] <= counts["n_valid"] <= counts["n_total"]
                or counts["n_failed"] != counts["n_total"] - counts["n_valid"]):
            raise ValueError(f"Inconsistent independent/paired trial counts: {key}")
        values = {field: _number(row[field]) for field in SUMMARY_STATISTICS}
        if any(math.isinf(value) for value in values.values()):
            raise ValueError(f"Infinite summary statistic: {key}")
        if counts["n_paired"] < MIN_PAIRED_TRIALS:
            if any(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Insufficient paired trials must retain missing statistics: {key}")
        else:
            if any(not math.isfinite(value) for value in values.values()):
                raise ValueError(f"A sufficient paired sample must have finite statistics: {key}")
            if (values["std_joint_utility"] < 0.0
                    or min(values[field] for field in
                           ("mean_selected_count", "mean_risk_reward", "mean_makespan_ms")) < 0.0):
                raise ValueError(f"Invalid nonnegative diagnostic statistic: {key}")
            mean = values["mean_joint_utility"]
            if not values["ci95_lower"] - 1e-12 <= mean <= values["ci95_upper"] + 1e-12:
                raise ValueError(f"Mean lies outside its saved confidence interval: {key}")
        row.update(counts)
        row.update(values)
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("Summary must contain every formal J1/J2 configuration, including explicit missing points.")
    for figure, n, beta, sigma, _ in expected:
        paired_counts = {indexed[(figure, n, beta, sigma, method)]["n_paired"] for method in METHODS}
        if len(paired_counts) != 1:
            raise ValueError(f"All five methods must use one common paired-trial set: {figure}/{n}/{beta}/{sigma}")
    # Both figure slices must be copied from one shared configuration, including
    # missing means and independent failures; comparing NaNs requires care.
    for n in N_VALUES:
        for method in METHODS:
            left = indexed[("J1", int(n), float(J2_BETA_B), float(J1_SIGMA), method)]
            right = indexed[("J2", int(n), float(J2_BETA_B), float(J1_SIGMA), method)]
            for field in (*SUMMARY_COUNTS, *SUMMARY_STATISTICS):
                a, b = left[field], right[field]
                if a != b and not (math.isnan(a) and math.isnan(b)):
                    raise ValueError(f"J1/J2 overlap disagrees: N={n}, method={method}, statistic={field}")
    return indexed


def _key(figure: str, n: int, panel_value: float, method: str) -> tuple:
    beta, sigma = ((panel_value, J1_SIGMA) if figure == "J1" else (J2_BETA_B, panel_value))
    return figure, int(n), float(beta), float(sigma), method


def _shared_y_limits(series: list[dict]) -> tuple[float, float]:
    finite = np.asarray([value for panel in series for values in panel.values()
                         for value in values if math.isfinite(value)], dtype=float)
    if not len(finite):
        raise ValueError("No finite paired mean exists for this figure; inspect the saved failure counts.")
    lower, upper = float(finite.min()), float(finite.max())
    span = upper - lower
    padding = 0.08 * span if span > 0.0 else max(0.025, abs(lower) * 0.08)
    # Never clamp J to zero or to [0, 1]. Every actual negative mean remains
    # visible on a complete continuous axis shared by all five panels.
    return lower - padding, upper + padding


def _draw_figure(indexed: dict, figure: str):
    panel_values = BETA_PANEL_VALUES if figure == "J1" else SIGMA_PANEL_VALUES
    series = [{method: np.asarray([
        indexed[_key(figure, n, panel_value, method)]["mean_joint_utility"] for n in N_VALUES
    ], dtype=float) for method in METHODS} for panel_value in panel_values]
    y_limits = _shared_y_limits(series)
    fig = plt.figure(figsize=(DETECTION_FIGURE_WIDTH, DETECTION_FIGURE_HEIGHT))
    width_pt, height_pt = fig.get_size_inches() * 72.0
    # Groups 02/03/04 use one uninterrupted axis within this same physical
    # envelope. Detection segment heights define geometry only, never a break.
    panel_height_pt = (DETECTION_UPPER_HEIGHT_PT + DETECTION_LOWER_HEIGHT_PT
                       + DETECTION_BREAK_GAP_PT)
    body_width_pt = (5 * DETECTION_PANEL_WIDTH_PT + 4 * DETECTION_PANEL_GAP_PT)
    bottom = DETECTION_AXES_BOTTOM_PT / height_pt
    top = (DETECTION_AXES_BOTTOM_PT + panel_height_pt) / height_pt
    grid = fig.add_gridspec(1, 5, left=DETECTION_AXES_LEFT_PT / width_pt,
                           right=(DETECTION_AXES_LEFT_PT + body_width_pt) / width_pt,
                           bottom=bottom, top=top,
                           wspace=DETECTION_PANEL_GAP_PT / DETECTION_PANEL_WIDTH_PT)
    axes = []
    for panel_id, (panel_value, panel_series) in enumerate(zip(panel_values, series)):
        ax = fig.add_subplot(grid[0, panel_id], sharey=axes[0] if axes else None)
        axes.append(ax)
        for method in METHODS:
            style = {**method_style(method), "linestyle": "-"}
            # Passing all nine positions and the unfilled NaNs explicitly keeps
            # markers at real N values and breaks the line at missing points.
            ax.plot(N_VALUES, panel_series[method], **style, markerfacecolor="white",
                    markeredgewidth=0.8, label=_display_name(method), zorder=3)
        ax.set_xlim(N_VALUES[0] - 100, N_VALUES[-1] + 100)
        ax.set_xticks(N_VALUES)
        ax.set_xticklabels([str(n) if n % 1000 == 0 else "" for n in N_VALUES])
        ax.set_ylim(*y_limits)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        formatter = ScalarFormatter(useOffset=False, useMathText=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(axis="both", pad=2.0, labelsize=DETECTION_TICK_FONT_PT)
        ax.tick_params(axis="y", labelleft=panel_id == 0)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D3D3D3", linewidth=0.4)
        ax.set_xlabel(r"Number of Model Instances $N$", labelpad=2.0,
                      fontsize=DETECTION_LABEL_FONT_PT)
        variable = r"$\beta_B$" if figure == "J1" else r"$\sigma$"
        position = ax.get_position()
        fig.text((position.x0 + position.x1) / 2.0, DETECTION_CAPTION_CENTER_PT / height_pt,
                 f"({chr(97 + panel_id)}) {variable} = {panel_value:.2f}",
                 ha="center", va="center", fontsize=DETECTION_LABEL_FONT_PT)
    fig.text(DETECTION_YLABEL_CENTER_X_PT / width_pt, (bottom + top) / 2.0,
             "Joint Verification\nUtility $J$", rotation=90, ha="center", va="center",
             fontsize=DETECTION_LABEL_FONT_PT)
    handles = [Line2D([], [], label=_display_name(method),
                      **{**method_style(method), "linestyle": "-"},
                      markerfacecolor="white", markeredgewidth=0.8) for method in METHODS]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.525, top + DETECTION_LEGEND_GAP_PT / height_pt),
               ncol=5, frameon=False, handlelength=2.0, columnspacing=2.0,
               borderaxespad=0.0, borderpad=0.1, fontsize=DETECTION_TICK_FONT_PT)
    evidence = []
    for panel_value in panel_values:
        for n in N_VALUES:
            point = {"N": int(n), "beta_B": float(panel_value if figure == "J1" else J2_BETA_B),
                     "sigma": float(J1_SIGMA if figure == "J1" else panel_value), "methods": {}}
            for method in METHODS:
                row = indexed[_key(figure, n, panel_value, method)]
                point["methods"][method] = {
                    **{field: row[field] for field in SUMMARY_COUNTS},
                    **{field: row[field] if math.isfinite(row[field]) else None
                       for field in SUMMARY_STATISTICS},
                }
            evidence.append(point)
    return fig, {"figure": figure, "name": FIGURES[figure], "y_limits": list(y_limits),
                 "panel_values": list(panel_values), "x_coordinates": list(N_VALUES),
                 "labeled_x_ticks": [n for n in N_VALUES if n % 1000 == 0],
                 "per_point_evidence": evidence}


def _save_figure(fig, path: Path, **kwargs) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        # Deliberately avoid bbox_inches='tight': it changes the fixed 7.16-inch
        # paper page and the physical panel dimensions shared with Groups 01-04.
        fig.savefig(temporary, format=path.suffix.lstrip("."), **kwargs)
        temporary.replace(path)
    except PermissionError as exc:
        raise PermissionError(f"Cannot replace {path}; close its image/PDF preview and rerun this plot entry.") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def plot_joint_decision(result_root=RESULT_ROOT) -> dict:
    """Read saved evidence and write exactly J1/J2 plus figure provenance."""
    result_root = Path(result_root).resolve()
    rows, manifest, sources = _read_sources(result_root)
    indexed = _summary_index(rows, manifest)
    # Check both figures before making outputs so an entirely undefined figure
    # cannot leave a misleading newly saved companion figure.
    for figure, panel_values in (("J1", BETA_PANEL_VALUES), ("J2", SIGMA_PANEL_VALUES)):
        _shared_y_limits([{method: [indexed[_key(figure, n, panel, method)]["mean_joint_utility"]
                                   for n in N_VALUES] for method in METHODS} for panel in panel_values])
    font = apply_style()
    folder = result_root / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    logs = result_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    figure_records = []
    for figure in FIGURES:
        fig, info = _draw_figure(indexed, figure)
        paths = {extension: folder / f"{FIGURES[figure]}.{extension}" for extension in ("pdf", "png")}
        try:
            _save_figure(fig, paths["pdf"], metadata={
                "Title": ("Joint Verification Utility under Different Verification Budgets" if figure == "J1"
                          else "Joint Verification Utility under Different Resource Heterogeneity Levels"),
                "Subject": ("Paired valid five-method seed-level means; 95% CI retained in CSV only; "
                            f"algorithm_version={manifest['algorithm_version']}; run_id={manifest['run_id']}; "
                            f"summary_sha256={sources['joint_summary']['sha256']}"),
            })
            _save_figure(fig, paths["png"], dpi=PNG_DPI)
        finally:
            plt.close(fig)
        info["files"] = {extension: _file_identity(path, result_root) for extension, path in paths.items()}
        figure_records.append(info)
    metadata = {
        **{field: manifest[field] for field in IDENTITY_FIELDS},
        "generated_at": datetime.now(timezone(timedelta(hours=8), "Asia/Shanghai")).isoformat(),
        "timezone": "Asia/Shanghai", "font": font, "source_files": sources,
        "figure_output_root": str(folder), "figures_generated": list(FIGURES.values()),
        "figure_width_inches": DETECTION_FIGURE_WIDTH,
        "figure_height_inches": DETECTION_FIGURE_HEIGHT,
        "panel_width_pt": DETECTION_PANEL_WIDTH_PT,
        "panel_height_pt": (DETECTION_UPPER_HEIGHT_PT + DETECTION_LOWER_HEIGHT_PT + DETECTION_BREAK_GAP_PT),
        "panel_gap_pt": DETECTION_PANEL_GAP_PT, "axes_left_pt": DETECTION_AXES_LEFT_PT,
        "axes_bottom_pt": DETECTION_AXES_BOTTOM_PT,
        "shared_legend_gap_pt": DETECTION_LEGEND_GAP_PT,
        "panel_caption_center_pt": DETECTION_CAPTION_CENTER_PT,
        "panel_geometry_reference": "02_overhead/03_threshold/04_ground_truth continuous panel envelope",
        "methods": list(METHODS), "legend_labels": [_display_name(method) for method in METHODS],
        "method_styles": {method: {**method_style(method), "linestyle": "-"} for method in METHODS},
        "summary_policy": "paired_all_five_methods", "minimum_paired_trials": MIN_PAIRED_TRIALS,
        "repetitions": REPETITIONS, "png_dpi": PNG_DPI, "pdf_vector": True,
        "broken_axis": False, "shaded_bands": False, "mean_annotation": False,
        "confidence_interval_display": "CSV only", "line_styles": "all solid",
        "smoothing": False, "interpolation": False, "artificial_jitter": False,
        "missing_mean_policy": "NaN remains a line gap; never replaced, interpolated, or connected across",
        "negative_utility_policy": "Retain all finite signed means; shared data-derived limits per figure",
        "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plot_style_source_sha256": hashlib.sha256(
            (PROJECT_ROOT / "experiments/common/plotting_style.py").read_bytes()).hexdigest(),
        "figures": figure_records,
        "visual_inspection": "Not performed by this script; inspect the generated PDF/PNG after the formal run.",
    }
    _atomic_json(logs / "figure_manifest.json", metadata)
    return metadata


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Replot saved Group 05 joint verification utility CSV means only")
    parser.parse_args(argv)
    metadata = plot_joint_decision()
    print(f"Saved J1/J2 vector PDF and {PNG_DPI} dpi PNG figures in {metadata['figure_output_root']}. "
          "95% CI stays in CSV; missing paired means stay gaps.", flush=True)


if __name__ == "__main__":
    main()
