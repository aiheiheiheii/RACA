"""Replot verified saved Group 04 reliability summaries without executing GT.

Run this file directly in PyCharm after the complete formal experiment has
saved its CSV files and manifest.  No dataset or GT implementation is imported
here; source identity and formal-grid checks precede any figure output.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
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
    apply_style, GT_METHOD_COLORS, GT_METHOD_MARKERS, GT_METHOD_LINESTYLES, PNG_DPI,
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

_config = importlib.import_module("experiments.04_ground_truth.config_ground_truth")
RESULT_ROOT = _config.RESULT_ROOT
FIGURE_ROOT = _config.FIGURE_ROOT
Q_VALUES = _config.Q_VALUES
FAILURE_VALUES = _config.FAILURE_VALUES
METHODS = _config.METHODS
REPETITIONS = _config.REPETITIONS
IMPLEMENTATION_VERSION = _config.IMPLEMENTATION_VERSION

FIGURE_NAME = "Fig_G1_ground_truth_reliability"
LABELED_Q_VALUES = (0.00, 0.10, 0.20, 0.30, 0.40, 0.50)


def _method_style(method: str) -> dict:
    """Use Group 04's dedicated method palette without altering old mappings."""
    return {"color": GT_METHOD_COLORS[method], "marker": GT_METHOD_MARKERS[method],
            "linestyle": GT_METHOD_LINESTYLES[method]}


def _summary_index(rows: list[dict]) -> dict:
    """Reject missing, duplicate, partial, or nonfinite formal summary points."""
    indexed = {}
    for row in rows:
        key = (row["method"], float(row["q_C"]), float(row["p_fail"]))
        if key in indexed:
            raise ValueError(f"Duplicate ground-truth summary: {key}")
        if int(row["n_total"]) != REPETITIONS:
            raise ValueError(f"Ground-truth summaries require {REPETITIONS} trials: {key}")
        statistics = {field: float(row[field]) for field in
                      ("reliable_gt_rate", "mean", "std", "ci95_lower", "ci95_upper")}
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0
               for value in statistics.values()):
            raise ValueError(f"Saved reliability statistics must be finite in [0, 1]: {key}")
        rate = statistics["reliable_gt_rate"]
        if not math.isclose(statistics["mean"], rate, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Reliable GT rate and binary-success mean disagree: {key}")
        if not (statistics["ci95_lower"] <= rate + 1e-12
                and rate <= statistics["ci95_upper"] + 1e-12):
            raise ValueError(f"Saved reliability mean lies outside its confidence interval: {key}")
        indexed[key] = row
    expected = {(method, float(q_C), float(p_fail)) for method in METHODS
                for q_C in Q_VALUES for p_fail in FAILURE_VALUES}
    if set(indexed) != expected:
        raise ValueError("Saved reliability summaries must cover exactly the formal method/q_C/p_fail grid")
    return indexed


def _point_evidence(indexed: dict) -> list[dict]:
    """Retain exact plotted means, their denominator, and CSV-only uncertainty."""
    evidence = []
    for p_fail in FAILURE_VALUES:
        for q_C in Q_VALUES:
            point = {"q_C": float(q_C), "p_fail": float(p_fail), "methods": {}}
            for method in METHODS:
                row = indexed[(method, float(q_C), float(p_fail))]
                point["methods"][method] = {
                    "n_total": int(row["n_total"]),
                    **{field: float(row[field]) for field in
                       ("reliable_gt_rate", "mean", "std", "ci95_lower", "ci95_upper")},
                }
            evidence.append(point)
    return evidence


def _draw_figure(indexed: dict):
    """Use Group 03's full 0–1 axis and the same five physical panel envelopes."""
    fig = plt.figure(figsize=(DETECTION_FIGURE_WIDTH, DETECTION_FIGURE_HEIGHT))
    width_pt, height_pt = fig.get_size_inches() * 72.0
    panel_width_pt = DETECTION_PANEL_WIDTH_PT
    panel_gap_pt = DETECTION_PANEL_GAP_PT
    panel_height_pt = (DETECTION_UPPER_HEIGHT_PT + DETECTION_LOWER_HEIGHT_PT
                       + DETECTION_BREAK_GAP_PT)
    body_width_pt = (len(FAILURE_VALUES) * panel_width_pt
                     + (len(FAILURE_VALUES) - 1) * panel_gap_pt)
    left = DETECTION_AXES_LEFT_PT / width_pt
    right = (DETECTION_AXES_LEFT_PT + body_width_pt) / width_pt
    bottom = DETECTION_AXES_BOTTOM_PT / height_pt
    top = (DETECTION_AXES_BOTTOM_PT + panel_height_pt) / height_pt
    grid = fig.add_gridspec(1, len(FAILURE_VALUES), left=left, right=right,
                           bottom=bottom, top=top, wspace=panel_gap_pt / panel_width_pt)
    # Numerical percent coordinates preserve all eleven equally spaced real
    # measurements; tick-label thinning never removes a measured marker.
    q_percent = np.asarray(Q_VALUES, dtype=float) * 100.0
    axes = []
    for panel, p_fail in enumerate(FAILURE_VALUES):
        ax = fig.add_subplot(grid[0, panel], sharey=axes[0] if axes else None)
        axes.append(ax)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D3D3D3", linewidth=0.4)
        for method in METHODS:
            rates = np.asarray([
                float(indexed[(method, float(q_C), float(p_fail))]["reliable_gt_rate"])
                for q_C in Q_VALUES
            ], dtype=float)
            ax.plot(q_percent, rates, **_method_style(method), markerfacecolor="white",
                    markeredgewidth=0.8, label=method, zorder=3, clip_on=False)
        padding = (q_percent[-1] - q_percent[0]) * 0.06
        ax.set_xlim(q_percent[0] - padding, q_percent[-1] + padding)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_xticks([q_C * 100.0 for q_C in LABELED_Q_VALUES])
        ax.set_xticklabels([f"{q_C * 100.0:g}" for q_C in LABELED_Q_VALUES])
        ax.set_xticks([float(q_C) * 100.0 for q_C in Q_VALUES
                      if q_C not in LABELED_Q_VALUES], minor=True)
        ax.tick_params(axis="both", which="major", pad=2.0,
                       labelsize=DETECTION_TICK_FONT_PT)
        ax.tick_params(axis="x", which="minor", length=1.6, width=0.5)
        ax.tick_params(axis="y", labelleft=panel == 0)
        ax.set_xlabel("Candidate Corruption Ratio (%)", labelpad=2.0,
                      fontsize=DETECTION_LABEL_FONT_PT)
        position = ax.get_position()
        fig.text((position.x0 + position.x1) / 2.0, DETECTION_CAPTION_CENTER_PT / height_pt,
                 f"({chr(97 + panel)}) " + rf"$p_{{\mathrm{{fail}}}}$ = {p_fail * 100.0:g}%",
                 ha="center", va="center", fontsize=DETECTION_LABEL_FONT_PT)
    fig.text(DETECTION_YLABEL_CENTER_X_PT / width_pt, (bottom + top) / 2.0,
             "Reliable GT\nConstruction Rate", rotation=90, ha="center", va="center",
             fontsize=DETECTION_LABEL_FONT_PT)
    handles = [Line2D([], [], label=method, **_method_style(method),
                      markerfacecolor="white", markeredgewidth=0.8) for method in METHODS]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.525, top + DETECTION_LEGEND_GAP_PT / height_pt),
               ncol=len(METHODS), frameon=False, handlelength=2.0,
               columnspacing=2.0, borderaxespad=0.0, borderpad=0.1,
               fontsize=DETECTION_TICK_FONT_PT)
    return fig, {
        "figure_width_inches": float(DETECTION_FIGURE_WIDTH),
        "figure_height_inches": float(DETECTION_FIGURE_HEIGHT),
        "panel_width_pt": float(panel_width_pt), "panel_height_pt": float(panel_height_pt),
        "panel_gap_pt": float(panel_gap_pt), "axes_left_pt": DETECTION_AXES_LEFT_PT,
        "axes_bottom_pt": DETECTION_AXES_BOTTOM_PT,
        "axes_top_pt": DETECTION_AXES_BOTTOM_PT + panel_height_pt,
        "shared_legend_gap_pt": DETECTION_LEGEND_GAP_PT, "legend_center_x": 0.525,
        "panel_caption_center_pt": DETECTION_CAPTION_CENTER_PT,
        "tick_legend_font_pt": DETECTION_TICK_FONT_PT,
        "label_caption_font_pt": DETECTION_LABEL_FONT_PT,
        "panel_geometry_reference": "03_threshold complete 0–1 axis envelope",
        "x_axis_range_padding_fraction": 0.06,
        "x_label": "Candidate Corruption Ratio (%)", "x_label_line_break": False,
        "y_label": "Reliable GT\nConstruction Rate", "y_label_line_break": True,
    }


def plot_ground_truth() -> dict:
    """Replot saved evidence from the fixed Group 04 directory only."""
    storage = importlib.import_module("experiments.04_ground_truth.ground_truth_storage")
    result_root = RESULT_ROOT.resolve()
    summary_rows, manifest = storage.load_plot_sources(result_root)
    print(f"Plotting saved algorithm version: {manifest['algorithm_version']}; "
          f"admission: {manifest['protocol']['majority']}", flush=True)
    if manifest["algorithm_version"] != IMPLEMENTATION_VERSION:
        print("These are original v1 results. A full run is required to obtain the revised GT algorithm's results; "
              "redrawing preserves the saved measurements.", flush=True)
    indexed = _summary_index(summary_rows)
    font = apply_style()
    # All formal evidence and paper figures use the fixed Group 04 directory.
    folder = FIGURE_ROOT.resolve()
    logs = result_root / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    fig, geometry = _draw_figure(indexed)
    paths = {extension: folder / f"{FIGURE_NAME}.{extension}" for extension in ("pdf", "png")}
    try:
        # Do not use bbox_inches="tight": the saved page must retain the exact
        # paper-wide dimensions and fixed point-based layout above.
        _save_figure(fig, paths["pdf"], storage, metadata={
            "Title": "RACA-HIM Ground-Truth Construction Correctness and Robustness",
            "Subject": ("Saved reliable GT construction rate over all formal trials; "
                        "95% Wilson CI retained in CSV; "
                        f"algorithm_version={manifest['algorithm_version']}; "
                        f"source run_id={manifest.get('run_id', '')}"),
        })
        _save_figure(fig, paths["png"], storage, dpi=PNG_DPI)
    finally:
        plt.close(fig)
    metadata = {
        "figure_name": FIGURE_NAME, "methods": list(METHODS), "font": font,
        "generated_at": datetime.now(timezone(timedelta(hours=8), "Asia/Shanghai")).isoformat(),
        "timezone": "Asia/Shanghai", "run_id": manifest.get("run_id"),
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "algorithm_version": manifest.get("algorithm_version"),
        "code_version": manifest.get("code_version"),
        "source_result_root": str(result_root), "figure_output_root": str(folder),
        "figure_file_identity_base": str(PROJECT_ROOT),
        "plot_implementation_version": IMPLEMENTATION_VERSION,
        "source_files": manifest.get("files"), "source_protocol": manifest.get("protocol"),
        "source_row_counts": {"summary": len(summary_rows)},
        "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plot_style_source_sha256": hashlib.sha256(
            (PROJECT_ROOT / "experiments/common/plotting_style.py").read_bytes()).hexdigest(),
        "figure_files": {kind: storage.file_identity(path, PROJECT_ROOT)
                         for kind, path in paths.items()},
        "png_dpi": PNG_DPI, "pdf_vector": True,
        "metric": "reliable_gt_rate", "metric_unit": "fraction", "y_limits": [0.0, 1.0],
        "denominator_policy": "all independent trials, including GT failures and contamination",
        "repetitions_per_point": REPETITIONS, "broken_axis": False,
        "mean_annotation": False, "shaded_bands": False,
        "confidence_interval_display": "CSV only", "line_styles": "all solid",
        "smoothing": False, "interpolation": False, "artificial_jitter": False,
        "x_coordinates_q_C": [float(value) for value in Q_VALUES],
        "x_coordinates_percent": [float(value) * 100.0 for value in Q_VALUES],
        "labeled_x_ticks_percent": [value * 100.0 for value in LABELED_Q_VALUES],
        "panel_p_fail": [float(value) for value in FAILURE_VALUES],
        "method_styles": {method: _method_style(method) for method in METHODS},
        "per_point_evidence": _point_evidence(indexed), **geometry,
    }
    provenance_path = logs / "figure_provenance.json"
    storage.atomic_json(provenance_path, metadata)
    canonical_provenance_path = folder / f"{FIGURE_NAME}.provenance.json"
    storage.atomic_json(canonical_provenance_path, metadata)
    metadata["provenance_file"] = storage.file_identity(provenance_path, result_root)
    metadata["canonical_provenance_file"] = storage.file_identity(canonical_provenance_path, PROJECT_ROOT)
    return metadata


def _save_figure(fig, path: Path, storage, **kwargs) -> None:
    """Replace a fixed paper figure only after its complete temporary save."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        fig.savefig(temporary, format=path.suffix.lstrip("."), **kwargs)
        storage.replace_file(temporary, path)
    except PermissionError as exc:
        raise PermissionError(f"Cannot replace figure {path}; close its PDF/image preview and rerun the plot entry.") from exc
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Replot only saved Group 04 ground-truth reliability")
    parser.parse_args(argv)
    storage = importlib.import_module("experiments.04_ground_truth.ground_truth_storage")
    result_root = RESULT_ROOT.resolve()
    manifest_path = result_root / "logs" / "run_manifest.json"
    try:
        metadata = plot_ground_truth()
    except BaseException as exc:
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update(figures_status="failed", figure_error={
                "type": type(exc).__name__, "message": str(exc),
            })
            storage.atomic_json(manifest_path, manifest)
        raise
    # Standalone PyCharm redraws maintain the same manifest lifecycle as the
    # full experiment entry, including fresh PDF/PNG/provenance identities.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(figures_status="complete", figures=metadata)
    manifest.pop("figure_error", None)
    storage.atomic_json(manifest_path, manifest)
    print(f"Saved {metadata['figure_name']} PDF, {PNG_DPI} dpi PNG at {metadata['figure_output_root']}, and figure provenance.",
          flush=True)


if __name__ == "__main__":
    main()
