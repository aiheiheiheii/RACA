"""Read saved Group 03 evidence and draw the sole threshold sensitivity figure.

This entry point never imports or calls a detector.  The storage reader verifies
the saved formal grid, source identities, and summaries before plotting. Missing
seed-level metrics stay missing; only the saved finite means enter the figure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from experiments.common.plotting_style import (
    apply_style, METRIC_STYLES, PNG_DPI,
    DETECTION_FIGURE_WIDTH, DETECTION_FIGURE_HEIGHT,
    DETECTION_PANEL_WIDTH_PT, DETECTION_PANEL_GAP_PT,
    DETECTION_UPPER_HEIGHT_PT, DETECTION_LOWER_HEIGHT_PT, DETECTION_BREAK_GAP_PT,
    DETECTION_AXES_LEFT_PT, DETECTION_AXES_BOTTOM_PT,
    DETECTION_CAPTION_CENTER_PT, DETECTION_YLABEL_CENTER_X_PT,
    DETECTION_TICK_FONT_PT, DETECTION_LABEL_FONT_PT, DETECTION_LEGEND_GAP_PT,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

_config = importlib.import_module("experiments.03_threshold.config_threshold")
RESULT_ROOT = _config.RESULT_ROOT
CR_VALUES = _config.CR_VALUES
TAU_VALUES = _config.TAU_VALUES
DEFAULT_TAU = _config.DEFAULT_TAU
PLOT_METRICS = _config.PLOT_METRICS
METRIC_LABELS = {
    "accuracy": "Accuracy", "f1": "F1-score", "fpr": "FPR", "fnr": "FNR",
}
FIGURE_NAME = "Fig_S1_threshold_sensitivity"
# Label a subset of the real thresholds, including the unequal final step.
# Every measured threshold remains a plotted marker and an x-axis tick.
LABELED_TAUS = (0.990, 0.993, 0.996, 0.9995)
REFERENCE_COLOR = "#A0A0A0"
REFERENCE_WIDTH_PT = 0.45


def _file_identity(path: Path, result_root: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": path.relative_to(result_root).as_posix(),
            "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _summary_index(rows: list[dict]) -> dict:
    """Retain the exact saved means and their individual finite sample counts."""
    indexed = {}
    for row in rows:
        key = (float(row["cr"]), float(row["tau_lower"]), row["metric"])
        if key in indexed:
            raise ValueError(f"Duplicate threshold metric summary: {key}")
        mean = float(row["mean"])
        if np.isinf(mean) or (np.isfinite(mean) and not 0.0 <= mean <= 1.0):
            raise ValueError(f"Saved metric mean is outside [0, 1]: {key}")
        n_valid = int(row["n_valid"])
        if (n_valid == 0) != bool(np.isnan(mean)):
            raise ValueError(f"Saved mean and finite sample count disagree: {key}")
        indexed[key] = row
    expected = {(float(cr), float(tau), metric)
                for cr in CR_VALUES for tau in TAU_VALUES for metric in PLOT_METRICS}
    if set(indexed) != expected:
        raise ValueError("Saved metric summaries must cover the declared formal grid")
    return indexed


def _point_evidence(indexed: dict) -> list[dict]:
    points = []
    for cr in CR_VALUES:
        for tau in TAU_VALUES:
            evidence = {"cr_percent": float(cr), "tau_lower": float(tau), "metrics": {}}
            diagnostics = []
            for metric in PLOT_METRICS:
                row = indexed[(float(cr), float(tau), metric)]
                mean = float(row["mean"])
                counts = {field: int(row[field]) for field in
                          ("n_valid", "n_failed", "n_trials", "n_complete", "n_undefined")}
                evidence["metrics"][metric] = {
                    "mean": float(mean) if np.isfinite(mean) else None,
                    **counts, "statistics_reason": row.get("statistics_reason", ""),
                }
                display = f"{mean:.6g}" if np.isfinite(mean) else "missing"
                diagnostics.append(
                    f"{METRIC_LABELS[metric]}={display} "
                    f"(n_valid={counts['n_valid']}, n_failed={counts['n_failed']}, "
                    f"n_undefined={counts['n_undefined']}, n_trials={counts['n_trials']})")
            print(f"cr={cr:g}%, tau={tau:.4f}: " + "; ".join(diagnostics), flush=True)
            points.append(evidence)
    return points


def _draw_figure(indexed: dict):
    """Match Groups 01/02 physical panel dimensions on a complete metric axis."""
    fig = plt.figure(figsize=(DETECTION_FIGURE_WIDTH, DETECTION_FIGURE_HEIGHT))
    width_pt, height_pt = fig.get_size_inches() * 72.0
    # As in Group 02, the two detection segments and their gap define the
    # physical envelope of one continuous axis; no detection break is drawn.
    panel_width_pt = DETECTION_PANEL_WIDTH_PT
    panel_gap_pt = DETECTION_PANEL_GAP_PT
    panel_height_pt = (DETECTION_UPPER_HEIGHT_PT + DETECTION_LOWER_HEIGHT_PT
                       + DETECTION_BREAK_GAP_PT)
    body_width_pt = len(CR_VALUES) * panel_width_pt + (len(CR_VALUES) - 1) * panel_gap_pt
    left = DETECTION_AXES_LEFT_PT / width_pt
    right = (DETECTION_AXES_LEFT_PT + body_width_pt) / width_pt
    bottom = DETECTION_AXES_BOTTOM_PT / height_pt
    top = (DETECTION_AXES_BOTTOM_PT + panel_height_pt) / height_pt
    grid = fig.add_gridspec(1, len(CR_VALUES), left=left, right=right,
                           bottom=bottom, top=top, wspace=panel_gap_pt / panel_width_pt)
    thresholds = np.asarray(TAU_VALUES, dtype=float)
    axes = []
    for panel, cr in enumerate(CR_VALUES):
        ax = fig.add_subplot(grid[0, panel], sharey=axes[0] if axes else None)
        axes.append(ax)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D3D3D3", linewidth=0.4)
        ax.axvline(DEFAULT_TAU, color=REFERENCE_COLOR, linewidth=REFERENCE_WIDTH_PT,
                   linestyle="-", zorder=1)
        for metric in PLOT_METRICS:
            means = np.asarray([float(indexed[(float(cr), float(tau), metric)]["mean"])
                                for tau in TAU_VALUES], dtype=float)
            style = dict(METRIC_STYLES[metric])
            style["linestyle"] = "-"
            ax.plot(thresholds, means, **style, markerfacecolor="white",
                    markeredgewidth=0.8, label=METRIC_LABELS[metric],
                    zorder=3, clip_on=False)
        # The unequal 0.999 -> 0.9995 spacing is represented numerically.
        # Keep endpoint labels separated within the common panel dimensions.
        padding = (thresholds[-1] - thresholds[0]) * 0.06
        ax.set_xlim(thresholds[0] - padding, thresholds[-1] + padding)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_xticks(LABELED_TAUS)
        ax.set_xticklabels([f"{tau:.4f}" if tau == TAU_VALUES[-1] else f"{tau:.3f}"
                            for tau in LABELED_TAUS])
        ax.set_xticks([tau for tau in TAU_VALUES if tau not in LABELED_TAUS], minor=True)
        ax.tick_params(axis="both", which="major", pad=2.0,
                       labelsize=DETECTION_TICK_FONT_PT)
        ax.tick_params(axis="x", which="minor", length=1.6, width=0.5)
        ax.tick_params(axis="y", labelleft=panel == 0)
        ax.set_xlabel(r"Similarity Threshold $\tau^-$", labelpad=2.0,
                      fontsize=DETECTION_LABEL_FONT_PT)
        position = ax.get_position()
        fig.text((position.x0 + position.x1) / 2.0, DETECTION_CAPTION_CENTER_PT / height_pt,
                 f"({chr(97 + panel)}) cr = {cr:g}%", ha="center", va="center",
                 fontsize=DETECTION_LABEL_FONT_PT)
    fig.text(DETECTION_YLABEL_CENTER_X_PT / width_pt, (bottom + top) / 2.0,
             "Metric Value", rotation=90, ha="center", va="center",
             fontsize=DETECTION_LABEL_FONT_PT)
    handles = [Line2D([], [], label=METRIC_LABELS[metric],
                      **{**METRIC_STYLES[metric], "linestyle": "-"},
                      markerfacecolor="white", markeredgewidth=0.8)
               for metric in PLOT_METRICS]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.525, top + DETECTION_LEGEND_GAP_PT / height_pt),
               ncol=len(PLOT_METRICS), frameon=False, handlelength=2.0,
               columnspacing=2.0, borderaxespad=0.0, borderpad=0.1,
               fontsize=DETECTION_TICK_FONT_PT)
    return fig, {"figure_width_inches": float(DETECTION_FIGURE_WIDTH),
                 "figure_height_inches": float(DETECTION_FIGURE_HEIGHT),
                 "panel_width_pt": float(panel_width_pt), "panel_height_pt": float(panel_height_pt),
                 "panel_gap_pt": panel_gap_pt, "axes_left_pt": DETECTION_AXES_LEFT_PT,
                 "axes_bottom_pt": DETECTION_AXES_BOTTOM_PT,
                 "axes_top_pt": DETECTION_AXES_BOTTOM_PT + panel_height_pt,
                 "shared_legend_gap_pt": DETECTION_LEGEND_GAP_PT,
                 "legend_center_x": 0.525,
                 "panel_caption_center_pt": DETECTION_CAPTION_CENTER_PT,
                 "tick_legend_font_pt": DETECTION_TICK_FONT_PT,
                 "label_caption_font_pt": DETECTION_LABEL_FONT_PT,
                 "panel_geometry_reference": "01_detection/02_overhead complete panel envelope",
                 "x_axis_range_padding_fraction": 0.06}


def plot_threshold(result_root=RESULT_ROOT) -> dict:
    """Replot verified saved CSV evidence; never load data or execute RACA-HIM."""
    result_root = Path(result_root).resolve()
    storage = importlib.import_module("experiments.03_threshold.threshold_storage")
    raw_rows, summary_rows, gt_summary_rows, manifest = storage.load_plot_sources(result_root)
    indexed = _summary_index(summary_rows)
    print("Plotting conditional finite means; failed/undefined runs remain in the saved "
          "counts and missing means remain gaps. 95% CI is retained in CSV only.", flush=True)
    points = _point_evidence(indexed)
    font = apply_style()
    folder = result_root / "figures"
    logs = result_root / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    fig, geometry = _draw_figure(indexed)
    paths = {extension: folder / f"{FIGURE_NAME}.{extension}" for extension in ("pdf", "png")}
    try:
        fig.savefig(paths["pdf"], metadata={
            "Title": "RACA-HIM Similarity Threshold Sensitivity",
            "Subject": ("Saved finite seed-level metric means; failed and undefined runs "
                        "retained in provenance; gray reference is default tau=0.997; "
                        f"source run_id={manifest.get('run_id', '')}"),
        })
        fig.savefig(paths["png"], dpi=PNG_DPI)
    finally:
        plt.close(fig)
    metadata = {
        "figure_name": FIGURE_NAME, "method": "RACA-HIM", "font": font,
        "generated_at": datetime.now(timezone(timedelta(hours=8), "Asia/Shanghai")).isoformat(),
        "timezone": "Asia/Shanghai",
        "run_id": manifest.get("run_id"),
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "algorithm_version": manifest.get("algorithm_version"),
        "code_version": manifest.get("code_version"),
        "source_files": manifest.get("files"),
        "source_row_counts": {"raw": len(raw_rows), "summary": len(summary_rows),
                              "gt_summary": len(gt_summary_rows)},
        "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plot_style_source_sha256": hashlib.sha256(
            (PROJECT_ROOT / "experiments/common/plotting_style.py").read_bytes()).hexdigest(),
        "figure_files": {kind: _file_identity(path, result_root) for kind, path in paths.items()},
        "png_dpi": PNG_DPI, "metric_unit": "fraction", "y_limits": [0.0, 1.0],
        "broken_axis": False, "mean_annotation": False, "shaded_bands": False,
        "confidence_interval_display": "CSV only", "line_styles": "all solid",
        "smoothing": False, "interpolation": False,
        "x_coordinates": list(TAU_VALUES), "labeled_x_ticks": list(LABELED_TAUS),
        "metric_styles": {metric: dict(METRIC_STYLES[metric]) for metric in PLOT_METRICS},
        "default_threshold_reference": {"tau_lower": DEFAULT_TAU, "color": REFERENCE_COLOR,
                                         "linewidth_pt": REFERENCE_WIDTH_PT},
        "summary_interpretation": "conditional on finite metric values from complete runs",
        "missing_mean_policy": "NaN plotted as a gap, never replaced or connected across",
        "per_point_evidence": points, **geometry,
    }
    provenance_path = logs / "figure_provenance.json"
    storage.atomic_json(provenance_path, metadata)
    metadata["provenance_file"] = _file_identity(provenance_path, result_root)
    return metadata


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Replot only saved Group 03 threshold metrics")
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args(argv)
    metadata = plot_threshold(args.result_root)
    print(f"Saved {metadata['figure_name']} PDF, {PNG_DPI} dpi PNG, and figure provenance.",
          flush=True)


if __name__ == "__main__":
    main()
