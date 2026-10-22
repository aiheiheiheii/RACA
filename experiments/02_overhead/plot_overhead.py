"""Draw O1/O2 from saved fixed-workload statistics; never run an algorithm."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from .overhead_storage import atomic_write, retry_io
from experiments.common.method_names import canonical_method_name
from experiments.common.plotting_style import (
    apply_style, method_style, PNG_DPI, DETECTION_LEGEND_GAP_PT,
    DETECTION_FIGURE_WIDTH, DETECTION_FIGURE_HEIGHT,
    DETECTION_PANEL_WIDTH_PT, DETECTION_PANEL_GAP_PT,
    DETECTION_UPPER_HEIGHT_PT, DETECTION_LOWER_HEIGHT_PT, DETECTION_BREAK_GAP_PT,
    DETECTION_AXES_LEFT_PT, DETECTION_AXES_BOTTOM_PT,
    DETECTION_CAPTION_CENTER_PT, DETECTION_YLABEL_CENTER_X_PT,
    DETECTION_TICK_FONT_PT, DETECTION_LABEL_FONT_PT,
)
from .config_overhead import (
    N_VALUES, D_PANEL_VALUES, METHODS, SEEDS, MAIN_EVALUATION_MODE, OVERHEAD_IMPLEMENTATION_VERSION,
)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter


def _read_summary(path, fingerprint):
    if not path.is_file():
        raise FileNotFoundError(f"No saved summary: {path}. Run run_overhead.py first.")
    def read():
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return [row for row in csv.DictReader(stream)
                    if row["evaluation_mode"] == MAIN_EVALUATION_MODE]
    records = retry_io(read, description=f'Read saved summary {path}')
    expected = {(n, d, method) for n in N_VALUES for d in D_PANEL_VALUES for method in METHODS}
    # Canonicalize only lookup keys. Stored labels and identity fields remain
    # intact so old Proposed results retain their original provenance.
    actual = [(int(row["N"]), int(row["d"]), canonical_method_name(row["method"]))
              for row in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Saved fixed-workload summaries must contain exactly one row for every N/d/method")
    if any(row.get("implementation_version") != OVERHEAD_IMPLEMENTATION_VERSION
           or row.get("run_fingerprint") != fingerprint for row in records):
        raise ValueError("Saved summary contains mixed or incompatible source/config/data identities")
    if any(int(row["n_valid"]) not in (0, 1)
           or ("n_expected" in row and int(row["n_expected"]) != len(SEEDS))
           for row in records):
        raise ValueError("Core-overhead figures require one paired observation per N/d point")
    return {(int(row["N"]), int(row["d"]), canonical_method_name(row["method"])): row
            for row in records}


def _mean(row):
    value = float(row["value"] if "value" in row else row["mean"])
    if int(row["n_valid"]) == 0:
        if math.isfinite(value):
            raise ValueError("A summary with no valid paired observation cannot supply a plotted value")
        return math.nan
    if not math.isfinite(value) or value < 0:
        raise ValueError("Observed overhead values must be nonnegative and finite")
    return value


def _five_panel(records, folder, kind, fingerprint, divisor=1.0, communication_unit="KB",
                *, subject_suffix=""):
    series = [{method: np.array([_mean(records[(n, d, method)]) / divisor for n in N_VALUES])
               for method in METHODS} for d in D_PANEL_VALUES]
    finite = np.array([value for panel in series for values in panel.values()
                       for value in values if math.isfinite(value)])
    if not len(finite):
        return None
    # A complete zero-based axis represents all measurements, with one common
    # scale across five panels. Detection-specific breaks are inapplicable here.
    upper = max(float(finite.max()) * 1.08, 1.0)
    fig = plt.figure(figsize=(DETECTION_FIGURE_WIDTH, DETECTION_FIGURE_HEIGHT))
    # Match Group 01's physical panel envelope: both detection axis segments
    # plus their break gap become a single complete overhead axis.
    width_pt, height_pt = fig.get_size_inches() * 72.
    body_width_pt = (len(D_PANEL_VALUES) * DETECTION_PANEL_WIDTH_PT
                     + (len(D_PANEL_VALUES) - 1) * DETECTION_PANEL_GAP_PT)
    body_height_pt = (DETECTION_UPPER_HEIGHT_PT + DETECTION_LOWER_HEIGHT_PT
                      + DETECTION_BREAK_GAP_PT)
    left = DETECTION_AXES_LEFT_PT / width_pt
    right = (DETECTION_AXES_LEFT_PT + body_width_pt) / width_pt
    bottom = DETECTION_AXES_BOTTOM_PT / height_pt
    top = (DETECTION_AXES_BOTTOM_PT + body_height_pt) / height_pt
    grid = fig.add_gridspec(1, len(D_PANEL_VALUES), left=left, right=right,
                           bottom=bottom, top=top,
                           wspace=DETECTION_PANEL_GAP_PT / DETECTION_PANEL_WIDTH_PT)
    legend_axis = None
    for panel_id, (d, panel_series) in enumerate(zip(D_PANEL_VALUES, series)):
        ax = fig.add_subplot(grid[0, panel_id])
        for method in METHODS:
            label = method.removesuffix("-Adapted")
            ax.plot(N_VALUES, panel_series[method], **method_style(method),
                    markerfacecolor="white", markeredgewidth=.8, label=label, zorder=3)
        ax.set_ylim(0., upper)
        ax.set_xlim(N_VALUES[0] - 100, N_VALUES[-1] + 100)
        ax.set_xticks(N_VALUES)
        ax.set_xticklabels([str(n) if n % 1000 == 0 else "" for n in N_VALUES])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        formatter = ScalarFormatter(useOffset=False, useMathText=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(axis="both", pad=2, labelsize=DETECTION_TICK_FONT_PT)
        ax.tick_params(axis="y", labelleft=panel_id == 0)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D3D3D3", linewidth=.4)
        ax.set_xlabel("Number of Model Instances", fontsize=DETECTION_LABEL_FONT_PT, labelpad=2)
        position = ax.get_position()
        fig.text((position.x0 + position.x1) / 2, DETECTION_CAPTION_CENTER_PT / height_pt,
                 f"({chr(97 + panel_id)}) d = {d}", ha="center", va="center",
                 fontsize=DETECTION_LABEL_FONT_PT)
        if legend_axis is None:
            legend_axis = ax
    ylabel = ("Time Overhead (ms)" if kind == "time"
              else f"Communication Overhead ({communication_unit})")
    display_ylabel = ("Time Overhead (ms)" if kind == "time"
                      else f"Communication\nOverhead ({communication_unit})")
    fig.text(DETECTION_YLABEL_CENTER_X_PT / width_pt, (bottom + top) / 2,
             display_ylabel, rotation=90, ha="center", va="center",
             fontsize=DETECTION_LABEL_FONT_PT)
    handles, labels = legend_axis.get_legend_handles_labels()
    legend_bottom = top + DETECTION_LEGEND_GAP_PT / height_pt
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.525, legend_bottom),
               ncol=len(METHODS), frameon=False, handlelength=2.0, columnspacing=2.,
               borderaxespad=0., borderpad=.1, fontsize=DETECTION_TICK_FONT_PT)
    basename = "Fig_O1_time_overhead" if kind == "time" else "Fig_O2_communication_overhead"
    subject = (f"Version {OVERHEAD_IMPLEMENTATION_VERSION}; fingerprint {fingerprint}; "
               "one paired seed; same B targets; core online steps; no averaging or confidence intervals; "
               "time is measured core preparation CPU + simulated preparation messages + measured decision CPU + simulated core verification"
               if kind == "time" else
               f"Version {OVERHEAD_IMPLEMENTATION_VERSION}; fingerprint {fingerprint}; "
               f"one paired seed; same B targets; core online steps; {communication_unit} uses binary conversion; no confidence intervals")
    if subject_suffix:
        subject += '; '+subject_suffix
    try:
        atomic_write(folder/f'{basename}.pdf',lambda temporary:fig.savefig(
            temporary,format='pdf',metadata={'Title':ylabel,'Subject':subject}))
        atomic_write(folder/f'{basename}.png',lambda temporary:fig.savefig(
            temporary,format='png',dpi=PNG_DPI))
    finally:
        plt.close(fig)
    return {"name": basename, "y_limits": [0., upper], "unit": "ms" if kind == "time" else communication_unit,
            "divisor": divisor, "broken_axis": False}


def plot_communication_results(output_dir):
    """Draw only encoded O2; never read a time value or write O1."""
    from .overhead_communication import validate_encoded_overlay
    output = Path(output_dir).resolve()
    overlay = validate_encoded_overlay(output, allow_computed=True)
    base_fingerprint = overlay['base_run_fingerprint']
    records = _read_summary(output/'summary'/'communication_encoded_summary.csv', base_fingerprint)
    if any(row.get('communication_fingerprint') != overlay['communication_fingerprint']
           or row.get('communication_schema') != overlay['schema'] for row in records.values()):
        raise ValueError('Encoded communication summary has a mixed codec identity')
    if any(int(row['n_valid']) != 1 for row in records.values()):
        raise ValueError('Encoded communication replay requires the complete paired grid')
    values = np.array([_mean(row) for row in records.values()])
    use_mb = bool(values.max() >= 1024.**2)
    unit, divisor = ('MB',1024.**2) if use_mb else ('KB',1024.)
    folder = output/'figures'
    folder.mkdir(parents=True,exist_ok=True)
    font = apply_style()
    figure = _five_panel(records, folder, 'communication', base_fingerprint,
        divisor=divisor, communication_unit=unit,
        subject_suffix=('encoded application packets; decision-equivalent progressive direction transport; '
                        f'communication fingerprint {overlay["communication_fingerprint"]}; '
                        'time experiment retains original uncompressed v4 transport; '
                        'refinement and raw fallback bytes included'))
    if figure is None:
        raise ValueError('No encoded communication figure was generated')
    return dict(font=font,run_fingerprint=base_fingerprint,
        results_implementation_version=overlay['base_implementation_version'],
        communication_fingerprint=overlay['communication_fingerprint'],
        communication_schema=overlay['schema'],figures_generated=[figure['name']],figures=[figure],
        figure_width_inches=DETECTION_FIGURE_WIDTH,figure_height_inches=DETECTION_FIGURE_HEIGHT,
        panel_width_pt=DETECTION_PANEL_WIDTH_PT,
        panel_height_pt=DETECTION_UPPER_HEIGHT_PT+DETECTION_LOWER_HEIGHT_PT+DETECTION_BREAK_GAP_PT,
        legend_labels=[method.removesuffix('-Adapted') for method in METHODS],
        png_dpi=PNG_DPI,time_outputs='preserved byte-for-byte; not plotted by this function',
        communication_conversion='KB = 1024 bytes; MB = 1024^2 bytes',
        shaded_bands=False,line_styles='all solid')


def plot_results(output_dir):
    """Regenerate exactly the five-panel O1/O2 publication figures."""
    output = Path(output_dir).resolve()
    manifest_path = output / "logs" / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No saved run manifest at {manifest_path}")
    manifest = retry_io(lambda:json.loads(manifest_path.read_text(encoding='utf-8')),
                        description=f'Read plotting manifest {manifest_path}')
    if manifest.get('identity_schema')=='computation-v1':
        derived=manifest.get('derived_outputs',{})
        if (derived.get('state') not in ('summaries_ready','figures_ready','complete')
                or derived.get('run_fingerprint')!=manifest.get('run_fingerprint')):
            raise ValueError('Saved raw/summary exports are pending or incomplete. Resume run_overhead.py normally; do not plot a mixed output generation.')
    elif manifest.get('status') in {'running','interrupted','exporting','execution_failed','output_io_failed'}:
        raise ValueError('Saved raw/summary exports are pending or incomplete. Resume run_overhead.py normally before --plots-only.')
    if manifest.get("implementation_version") != OVERHEAD_IMPLEMENTATION_VERSION:
        raise ValueError(f"Saved results use {manifest.get('implementation_version')}; current simulation uses {OVERHEAD_IMPLEMENTATION_VERSION}. "
                         "Run run_overhead.py normally to recompute corrected measurements. Old failed records cannot be repaired with --plots-only.")
    if "seeds" in manifest and list(manifest["seeds"]) != list(SEEDS):
        raise ValueError("Saved core-overhead manifest uses a different paired seed policy")
    fingerprint = manifest.get("run_fingerprint")
    if not fingerprint:
        raise ValueError("Saved results have no source/config/data identity")
    saved_methods = manifest.get("methods")
    if saved_methods is not None:
        canonical_saved_methods = [canonical_method_name(method) for method in saved_methods]
        if (len(canonical_saved_methods) != len(METHODS)
                or set(canonical_saved_methods) != set(METHODS)):
            raise ValueError("Saved overhead manifest declares incompatible or duplicate method names")
    time_records = _read_summary(output / "summary" / "time_overhead_summary.csv", fingerprint)
    communication_records = _read_summary(output / "summary" / "communication_overhead_summary.csv", fingerprint)
    recorded_time_methods = {row["method"] for row in time_records.values()}
    recorded_communication_methods = {row["method"] for row in communication_records.values()}
    if (recorded_time_methods != recorded_communication_methods
            or (saved_methods is not None and recorded_time_methods != set(saved_methods))):
        raise ValueError("Saved summaries must preserve the method names declared by their run identity")
    saved_methods = list(saved_methods) if saved_methods is not None else sorted(recorded_time_methods)
    if any(time_records[key]["n_valid"] != communication_records[key]["n_valid"] for key in time_records):
        raise ValueError("O1 and O2 must use the same eligible paired seed results")
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    font = apply_style()
    byte_values = np.array([_mean(row) for row in communication_records.values()])
    finite_bytes = byte_values[np.isfinite(byte_values)]
    use_mb = bool(len(finite_bytes) and finite_bytes.max() >= 1024. ** 2)
    unit, divisor = ("MB", 1024. ** 2) if use_mb else ("KB", 1024.)
    figures = [_five_panel(time_records, folder, "time", fingerprint),
               _five_panel(communication_records, folder, "communication", fingerprint,
                           divisor=divisor, communication_unit=unit)]
    figures = [record for record in figures if record is not None]
    if len(figures) != 2:
        raise ValueError(
            f'No complete pair of result figures was generated in {folder}. '
            'The saved summaries contain no valid paired observations; inspect '
            'raw/overhead_runs.csv and logs/statistics_policy.json. '
            'Existing images are not evidence of a new plotting run.')
    complete_points = sum(int(time_records[(n, d, METHODS[0])]["n_valid"]) == len(SEEDS)
                          for n in N_VALUES for d in D_PANEL_VALUES)
    no_data_points = [(n, d) for n in N_VALUES for d in D_PANEL_VALUES
                      if int(time_records[(n, d, METHODS[0])]["n_valid"]) == 0]
    if no_data_points:
        print(f"Fixed-workload points without a complete paired observation: {no_data_points}; missing values remain gaps.", flush=True)
    return {"font": font, "figure_width_inches": DETECTION_FIGURE_WIDTH,
            "figure_height_inches": DETECTION_FIGURE_HEIGHT,
            "png_dpi": PNG_DPI, "main_evaluation_mode": MAIN_EVALUATION_MODE,
            "results_implementation_version": manifest["implementation_version"],
            "run_fingerprint": fingerprint, "declared_methods": list(METHODS),
            "saved_method_names": saved_methods,
            "method_name_aliases": {method: canonical_method_name(method) for method in saved_methods
                                    if method != canonical_method_name(method)},
            "legend_labels": [method.removesuffix("-Adapted") for method in METHODS],
            "figures_generated": [record["name"] for record in figures], "figures": figures,
            "shaded_bands": False, "confidence_interval_display": "none; one paired observation",
            "seed_policy": "one paired seed per N/d point; original observed values", "line_styles": "all solid",
            "shared_legend_gap_pt": DETECTION_LEGEND_GAP_PT,
            "panel_width_pt": DETECTION_PANEL_WIDTH_PT,
            "panel_height_pt": (DETECTION_UPPER_HEIGHT_PT + DETECTION_LOWER_HEIGHT_PT
                                + DETECTION_BREAK_GAP_PT),
            "panel_gap_pt": DETECTION_PANEL_GAP_PT,
            "axes_left_pt": DETECTION_AXES_LEFT_PT,
            "axes_bottom_pt": DETECTION_AXES_BOTTOM_PT,
            "panel_caption_center_pt": DETECTION_CAPTION_CENTER_PT,
            "panel_caption_y": DETECTION_CAPTION_CENTER_PT / (DETECTION_FIGURE_HEIGHT * 72.),
            "panel_geometry_reference": "01_detection complete panel envelope including break gap",
            "communication_conversion": "KB = 1024 bytes; MB = 1024^2 bytes",
            "time_definition": "measured core preparation CPU + simulated preparation messages + measured decision CPU + simulated core verification + measured required history update CPU",
            "points_with_complete_paired_observation": complete_points,
            "points_without_valid_seeds": no_data_points,
            "visual_inspection": "not automated; open generated PDF/PNG after a real experiment"}
