"""Only the two Group 01 accuracy figures, styled after TMC Fig. 3."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

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
    DETECTION_AXES_LEFT_PT, DETECTION_AXES_BOTTOM_PT,
    DETECTION_CAPTION_CENTER_PT, DETECTION_YLABEL_CENTER_X_PT,
    DETECTION_TICK_FONT_PT, DETECTION_LABEL_FONT_PT,
    DETECTION_LOWER_YLIM, DETECTION_UPPER_YLIM, DETECTION_HEIGHT_RATIOS,
    DETECTION_BREAK_GAP_PT, DETECTION_BREAK_LINEWIDTH,
    DETECTION_BREAK_HALF_WIDTH_PT, DETECTION_BREAK_HALF_HEIGHT_PT,
)
from experiments.common.config import N_VALUES, CR_VALUES, D1_CR_VALUES, D2_N_VALUES, DETECTION_IMPLEMENTATION_VERSION
from experiments.common.method_names import canonical_method_name, normalize_method_rows
import importlib
DECLARED_METHODS = importlib.import_module("experiments.01_detection.detection_baselines").METHODS
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

IMPLEMENTATION_VERSION = DETECTION_IMPLEMENTATION_VERSION
READABLE_HASH_VERSIONS = {"hash-baselines-v3", IMPLEMENTATION_VERSION}


def _save(fig, folder, name, title, results_version):
    fig.savefig(folder / f"{name}.pdf", metadata={"Title": title,
                "Subject": f"Detector version: {results_version}; actual seed-level accuracy; confidence intervals retained in CSV"})
    fig.savefig(folder / f"{name}.png", dpi=PNG_DPI)
    plt.close(fig)


def _row(rows, N, cr, method):
    found = [row for row in rows if int(row["N"]) == N and float(row["cr"]) == cr and row["method"] == method]
    if len(found) != 1:
        raise ValueError(f"Expected one saved summary for {N}/{cr}/{method}; got {len(found)}")
    return found[0]


def _points(rows, pairs, method):
    return np.array([float(_row(rows, N, cr, method)["accuracy_mean"]) for N, cr in pairs])


def _axis_bands(values):
    """Use TMC bands when valid; never hide a measured mean inside the break.

    If means enter the omitted interval, enlarge the lower/upper bands around
    the actual values. If two separated bands cannot represent every mean,
    fall back to a complete axis instead of clipping experimental data.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if np.any((finite < 0.) | (finite > 1.)):
        raise ValueError("Saved detection accuracy is outside [0,1]")
    lower, upper = list(DETECTION_LOWER_YLIM), list(DETECTION_UPPER_YLIM)
    in_gap = finite[(finite > lower[1]) & (finite < upper[0])]
    if len(in_gap):
        low_values, high_values = finite[finite < .5], finite[finite >= .5]
        if not len(low_values) or not len(high_values):
            return None
        lower[1] = max(lower[1], float(np.ceil((low_values.max() + .02) * 100) / 100))
        upper[0] = min(upper[0], float(np.floor((high_values.min() - .02) * 100) / 100))
        if lower[1] >= upper[0]:
            return None
        print(f"Expanded broken-axis bands to retain every actual mean: {lower}, {upper}", flush=True)
    return tuple(lower), tuple(upper)


def _break_marks(ax, edge):
    """Double diagonal cuts at both vertical spines, with TMC point geometry.

    Each segment draws one slash at its inner boundary. The pair of axes makes
    the two parallel slashes on each side; no horizontal dividing spine remains.
    """
    position = ax.get_position()
    width, height = ax.figure.get_size_inches() * 72.
    dx = DETECTION_BREAK_HALF_WIDTH_PT / (width * position.width)
    dy = DETECTION_BREAK_HALF_HEIGHT_PT / (height * position.height)
    y = 0. if edge == "bottom" else 1.
    for x in (0., 1.):
        ax.plot((x - dx, x + dx), (y - dy, y + dy), transform=ax.transAxes,
                color="black", linewidth=DETECTION_BREAK_LINEWIDTH,
                clip_on=False, solid_capstyle="butt", zorder=6)


def _draw_data(ax, positions, series, methods, plot_kind):
    width = .82 / len(methods)
    for j, method in enumerate(methods):
        style = method_style(method)
        label = method.removesuffix("-Adapted")
        if plot_kind == "bar":
            ax.bar(positions + (j - (len(methods) - 1) / 2) * width,
                   series[method], width=width, color=style["color"],
                   edgecolor="black", linewidth=.3, label=label, zorder=3)
        else:
            ax.plot(positions, series[method], **style, markerfacecolor="white",
                    markeredgewidth=.8, label=label, zorder=3)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D3D3D3", linewidth=.4)
    ax.tick_params(axis="both", pad=2, labelsize=DETECTION_TICK_FONT_PT)


def _five_panel(rows, folder, kind, methods, plot_kind="bar", results_version=IMPLEMENTATION_VERSION):
    fixed = D1_CR_VALUES if kind == "D1" else D2_N_VALUES
    x = N_VALUES if kind == "D1" else CR_VALUES
    series_by_panel = []
    for fixed_value in fixed:
        pairs = [(N, fixed_value) for N in x] if kind == "D1" else [(fixed_value, cr) for cr in x]
        series_by_panel.append({method: _points(rows, pairs, method) for method in methods})
    bands = _axis_bands([value for series in series_by_panel for values in series.values() for value in values])
    fig = plt.figure(figsize=(DETECTION_FIGURE_WIDTH, DETECTION_FIGURE_HEIGHT))
    # Point-based geometry keeps every panel the same physical size as TMC
    # Fig. 3, independent of the surrounding legend and outer white margins.
    width_pt, height_pt = fig.get_size_inches() * 72.
    body_width_pt = len(fixed) * DETECTION_PANEL_WIDTH_PT + (len(fixed) - 1) * DETECTION_PANEL_GAP_PT
    body_height_pt = DETECTION_UPPER_HEIGHT_PT + DETECTION_LOWER_HEIGHT_PT + DETECTION_BREAK_GAP_PT
    left = DETECTION_AXES_LEFT_PT / width_pt
    right = (DETECTION_AXES_LEFT_PT + body_width_pt) / width_pt
    bottom = DETECTION_AXES_BOTTOM_PT / height_pt
    top = (DETECTION_AXES_BOTTOM_PT + body_height_pt) / height_pt
    gap_ratio = 2. * DETECTION_BREAK_GAP_PT / (DETECTION_UPPER_HEIGHT_PT + DETECTION_LOWER_HEIGHT_PT)
    grid = fig.add_gridspec(2 if bands else 1, len(fixed), left=left, right=right,
                           bottom=bottom, top=top, wspace=DETECTION_PANEL_GAP_PT / DETECTION_PANEL_WIDTH_PT,
                           hspace=gap_ratio,
                           height_ratios=DETECTION_HEIGHT_RATIOS if bands else [1.])
    legend_axis = None
    for j, (fixed_value, series) in enumerate(zip(fixed, series_by_panel)):
        lower_ax = fig.add_subplot(grid[1 if bands else 0, j])
        axes = [lower_ax]
        if bands:
            upper_ax = fig.add_subplot(grid[0, j], sharex=lower_ax)
            axes.append(upper_ax)
            lower_ax.set_ylim(*bands[0])
            upper_ax.set_ylim(*bands[1])
            lower_ax.spines["top"].set_visible(False)
            upper_ax.spines["bottom"].set_visible(False)
            upper_ax.tick_params(axis="x", bottom=False, top=False, labelbottom=False)
            lower_ax.set_yticks([tick for tick in (0., .05, .10, .15) if tick <= bands[0][1]])
            if bands[0] != DETECTION_LOWER_YLIM:
                lower_ax.set_yticks(np.linspace(0., bands[0][1], 4))
            upper_ax.set_yticks([.95, 1.] if bands[1] == DETECTION_UPPER_YLIM
                               else np.linspace(bands[1][0], 1., 2))
        else:
            lower_ax.set_ylim(0., 1.02)
            lower_ax.set_yticks(np.arange(0., 1.01, .2))
        positions = np.arange(len(x), dtype=float) if plot_kind == "bar" else np.asarray(x)
        for ax in axes:
            _draw_data(ax, positions, series, methods, plot_kind)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.tick_params(axis="y", labelleft=j == 0)
            if plot_kind == "bar":
                ax.set_xlim(-.55, len(x) - .45)
        lower_ax.set_xticks(positions)
        if kind == "D1":
            lower_ax.set_xticklabels([f"{N}" if N % 1000 == 0 else "" for N in x])
            lower_ax.set_xlabel("Model Population (N)", labelpad=2, fontsize=DETECTION_LABEL_FONT_PT)
            caption = f"({chr(97+j)}) cr = {fixed_value:g}%"
        else:
            lower_ax.set_xticklabels([f"{cr:g}" if float(cr).is_integer() else "" for cr in x])
            lower_ax.set_xlabel("Corruption Ratio (%)", labelpad=2, fontsize=DETECTION_LABEL_FONT_PT)
            caption = f"({chr(97+j)}) N = {fixed_value}"
        # Fixed figure coordinates put panel captions close to their x labels.
        fig.text((lower_ax.get_position().x0 + lower_ax.get_position().x1) / 2,
                 DETECTION_CAPTION_CENTER_PT / height_pt, caption,
                 ha="center", va="center", fontsize=DETECTION_LABEL_FONT_PT)
        if bands:
            _break_marks(upper_ax, "bottom")
            _break_marks(lower_ax, "top")
        legend_axis = lower_ax if legend_axis is None else legend_axis
    fig.text(DETECTION_YLABEL_CENTER_X_PT / width_pt, (bottom + top) / 2,
             "Accuracy", rotation=90, ha="center", va="center", fontsize=DETECTION_LABEL_FONT_PT)
    handles, labels = legend_axis.get_legend_handles_labels()
    legend_bottom = top + DETECTION_LEGEND_GAP_PT / height_pt
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.525, legend_bottom),
               ncol=len(methods), frameon=False, handlelength=2.0, columnspacing=2.,
               borderaxespad=0., borderpad=.1, fontsize=DETECTION_TICK_FONT_PT)
    name = "Fig_D1_detection_vs_N" if kind == "D1" else "Fig_D2_detection_vs_cr"
    _save(fig, folder, name, "Corruption Detection Accuracy", results_version)
    return {"broken_axis": bands is not None,
            "lower_ylim": list(bands[0]) if bands else [0., 1.02],
            "upper_ylim": list(bands[1]) if bands else None}


def available_methods(rows):
    return tuple(method for method in DECLARED_METHODS if any(
        row["method"] == method and int(row["success_runs"]) > 0
        and np.isfinite(float(row["accuracy_mean"])) for row in rows))


def plot_results(output_dir, plot_kind="bar"):
    if plot_kind not in ("bar", "line"):
        raise ValueError("plot_kind must be bar or line")
    output = Path(output_dir)
    manifest_path = output / "logs" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    saved_version = manifest.get("implementation_version")
    if saved_version not in READABLE_HASH_VERSIONS:
        raise ValueError("Saved results predate the hash-baseline versions. Run the full experiment first; "
                         "historical similarity-baseline metrics cannot be presented as hash results.")
    with (output / "summary" / "detection_summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    saved_method_names = sorted({row["method"] for row in rows})
    saved_declared_methods = manifest.get("methods")
    if saved_declared_methods is not None:
        declared_aliases = [canonical_method_name(name) for name in saved_declared_methods]
        if (len(declared_aliases) != len(DECLARED_METHODS)
                or set(declared_aliases) != set(DECLARED_METHODS)
                or set(saved_declared_methods) != set(saved_method_names)):
            raise ValueError("Saved summaries must preserve their manifest method identities")
    rows = normalize_method_rows(rows)
    keys = [(int(row["N"]), float(row["cr"]), row["method"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate saved summaries, possibly mixing Proposed and RACA-HIM aliases")
    if (not rows or any(row.get("implementation_version") != saved_version for row in rows)
            or {row["method"] for row in rows} != set(DECLARED_METHODS)):
        raise ValueError("Saved summary belongs to an earlier or mixed detector version. "
                         "Complete the current full experiment before replotting.")
    methods = available_methods(rows)
    if not methods:
        raise ValueError("No method has valid saved accuracy")
    unavailable = [method for method in DECLARED_METHODS if method not in methods]
    print("Plotting accuracy for: " + ", ".join(methods), flush=True)
    if unavailable:
        print("Methods without valid accuracy: " + ", ".join(unavailable), flush=True)
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    font = apply_style()
    bands = {kind: _five_panel(rows, folder, kind, methods, plot_kind, results_version=saved_version)
             for kind in ("D1", "D2")}
    return {"font": font, "png_dpi": PNG_DPI, "figure_width_inches": DETECTION_FIGURE_WIDTH,
            "results_implementation_version": saved_version,
            "plot_source_implementation_version": IMPLEMENTATION_VERSION,
            "figure_height_inches": DETECTION_FIGURE_HEIGHT, "plotted_methods": methods,
            "legend_labels": [method.removesuffix("-Adapted") for method in methods],
            "saved_method_names": saved_method_names,
            "method_name_aliases": {name: canonical_method_name(name) for name in saved_method_names
                                    if canonical_method_name(name) != name},
            "unavailable_methods": unavailable, "main_plot_kind": plot_kind,
            "figures_generated": ["Fig_D1_detection_vs_N", "Fig_D2_detection_vs_cr"],
            "accuracy_unit": "fraction", "mean_annotation": False, "shaded_bands": False,
            "confidence_interval_display": "CSV only", "line_styles": "all solid",
            "broken_axis": all(info["broken_axis"] for info in bands.values()),
            "axis_bands": bands, "lower_ylim": list(DETECTION_LOWER_YLIM),
            "upper_ylim": list(DETECTION_UPPER_YLIM), "height_ratios": list(DETECTION_HEIGHT_RATIOS),
            "break_slashes": "double diagonal, both sides", "break_gap_pt": DETECTION_BREAK_GAP_PT,
            "break_linewidth_pt": DETECTION_BREAK_LINEWIDTH,
            "shared_legend_gap_pt": DETECTION_LEGEND_GAP_PT,
            "panel_width_pt": DETECTION_PANEL_WIDTH_PT,
            "panel_upper_height_pt": DETECTION_UPPER_HEIGHT_PT,
            "panel_lower_height_pt": DETECTION_LOWER_HEIGHT_PT,
            "panel_height_pt": DETECTION_UPPER_HEIGHT_PT + DETECTION_LOWER_HEIGHT_PT + DETECTION_BREAK_GAP_PT,
            "panel_gap_pt": DETECTION_PANEL_GAP_PT,
            "axes_bottom_pt": DETECTION_AXES_BOTTOM_PT,
            "panel_caption_center_pt": DETECTION_CAPTION_CENTER_PT,
            "panel_caption_y": DETECTION_CAPTION_CENTER_PT / (DETECTION_FIGURE_HEIGHT * 72.)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Replot only saved Group 01 detection accuracy")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "01_detection")
    parser.add_argument("--plot-kind", choices=("bar", "line"), default="bar")
    args = parser.parse_args(argv)
    metadata = plot_results(args.output_dir, plot_kind=args.plot_kind)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs" / "plot_style.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metadata)


if __name__ == "__main__":
    main()
