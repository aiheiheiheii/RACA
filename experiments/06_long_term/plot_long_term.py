"""Render Figure L1 from authenticated saved summaries, without simulation.

This file can be run directly in PyCharm.  It imports only declarative Group 06
configuration and the paper-wide style; no dataset, detector, scheduler, or
experiment driver is imported.  It never estimates missing experiment points.
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

from experiments.common.plotting_style import (
    apply_style, method_style, PNG_DPI,
    DETECTION_FIGURE_WIDTH, DETECTION_FIGURE_HEIGHT,
    DETECTION_PANEL_WIDTH_PT, DETECTION_PANEL_GAP_PT,
    DETECTION_UPPER_HEIGHT_PT, DETECTION_LOWER_HEIGHT_PT,
    DETECTION_BREAK_GAP_PT, DETECTION_AXES_LEFT_PT,
    DETECTION_AXES_BOTTOM_PT, DETECTION_CAPTION_CENTER_PT,
    DETECTION_YLABEL_CENTER_X_PT, DETECTION_TICK_FONT_PT,
    DETECTION_LABEL_FONT_PT, DETECTION_LEGEND_GAP_PT,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

_config = importlib.import_module("experiments.06_long_term.config_native_long_term")
_legacy_config = importlib.import_module("experiments.06_long_term.config_long_term")
RESULT_ROOT = _config.RESULT_ROOT
FIGURE_OUTPUT_ROOT = PROJECT_ROOT / "results" / "06_long_term"
DYNAMIC_CR_VALUES = _config.DYNAMIC_CR_VALUES
PLOT_ROUNDS = _config.PLOT_ROUNDS
METHODS = _config.METHODS
DISPLAY_METHODS = METHODS
LEGACY_DISPLAY_METHODS = ("RACA-HIM", "I-EDI-Adapted", "DVA-P-Adapted")
DISPLAY_SELECTION_VERSION = "native-three-method-single-run-seed2026-v1"
SEEDS = _config.SEEDS
REPETITIONS = _config.REPETITIONS
TOTAL_ROUNDS = _config.TOTAL_ROUNDS
ALGORITHM_VERSION = _config.ALGORITHM_VERSION
CODE_VERSION = _config.CODE_VERSION

FIGURE_NAME = "Fig_L1_long_term_detection"
IDENTITY_FIELDS = ("run_id", "code_version", "algorithm_version", "dataset_fingerprint")
STATISTICS = ("mean_cdr", "std_cdr", "ci95_lower", "ci95_upper",
              "mean_completed_count", "mean_cumulative_detected_events")
SINGLE_VALUES = ("cdr", "completed_count", "cumulative_detected_events")
COUNTS = ("n_total", "n_valid", "n_failed")
LABELED_ROUNDS = (0, 40, 80, 120, 160, 200)


def _display_name(method: str) -> str:
    # Legacy results remain visibly marked as adaptations. Native EDI-V is
    # named explicitly so a proof protocol is never confused with old SHA256.
    return method.replace("-Adapted", " (Adapted)")


def _style(method: str) -> dict:
    return {**method_style("DVA-P" if method == "DVA-P+EDI-V" else method), "linestyle": "-"}


def _configuration(manifest: dict):
    if manifest.get("protocol_family") == _config.PROTOCOL_FAMILY:
        return _config
    if (manifest.get("algorithm_version") == _legacy_config.ALGORITHM_VERSION
            and list(manifest.get("methods", ())) == list(_legacy_config.METHODS)
            and not manifest.get("protocol_family")):
        return _legacy_config
    raise ValueError("Saved result identity is neither this native protocol run nor the supported legacy adapted run")


def _default_result_root() -> Path:
    # A new single-run request must never silently fall back to an old mean.
    # Historical adapted means remain available only through --legacy or an
    # explicitly selected historical result root.
    return RESULT_ROOT


def _identity(path: Path, result_root: Path) -> dict:
    payload = path.read_bytes()
    return {"path": path.resolve().relative_to(result_root.resolve()).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _read_sources(result_root: Path) -> tuple[list[dict], dict, dict]:
    manifest_path = result_root / "logs" / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No saved formal run manifest: {manifest_path}. Run run_long_term.py first.")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    cfg = _configuration(manifest)
    if manifest.get("status") != "complete":
        raise ValueError("The formal long-term run is incomplete; resume it before producing Figure L1.")
    if (manifest.get("algorithm_version") != cfg.ALGORITHM_VERSION
            or manifest.get("code_version") != cfg.CODE_VERSION):
        raise ValueError("Saved long-term results belong to a different code/algorithm version.")
    if cfg is _config and (manifest.get("aggregation") != "single_run"
            or int(manifest.get("repetitions", -1)) != 1
            or int(manifest.get("single_seed", -1)) != 2026):
        raise ValueError("Native Figure L1 requires one directly observed run at seed 2026, without averaging.")
    if any(not manifest.get(field) for field in IDENTITY_FIELDS):
        raise ValueError("A completed run must preserve its run, code, algorithm, and dataset identities.")
    if (list(manifest.get("methods", ())) != list(cfg.METHODS)
            or list(manifest.get("seeds", ())) != list(cfg.SEEDS)
            or list(manifest.get("dynamic_cr_values", ())) != list(cfg.DYNAMIC_CR_VALUES)
            or int(manifest.get("total_rounds", -1)) != cfg.TOTAL_ROUNDS
            or list(manifest.get("plot_rounds", ())) != list(cfg.PLOT_ROUNDS)):
        raise ValueError("Figure L1 source must retain its declared methods, seeds, five ratios and 200 rounds.")
    declared = manifest.get("files", {}).get("long_term_summary")
    if not isinstance(declared, dict):
        raise ValueError("The run manifest must authenticate files.long_term_summary.")
    summary_path = result_root / "summary" / "long_term_summary.csv"
    relative = Path(str(declared.get("path", "")))
    if relative.is_absolute() or (result_root / relative).resolve() != summary_path.resolve():
        raise ValueError("The saved summary identity must point to summary/long_term_summary.csv.")
    if not summary_path.is_file():
        raise FileNotFoundError(f"No saved long-term summary: {summary_path}")
    payload = summary_path.read_bytes()
    actual = {"path": "summary/long_term_summary.csv", "size_bytes": len(payload),
              "sha256": hashlib.sha256(payload).hexdigest()}
    if any(actual[field] != declared.get(field) for field in ("sha256", "size_bytes")):
        raise ValueError("The summary bytes differ from the completed run manifest; resume/export that run again.")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
    required = {"cr_dynamic", "round", "method", *COUNTS, *IDENTITY_FIELDS}
    if cfg is _config:
        required.update({"seed", "aggregation", "aggregation_unit", "total_corruption_events", *SINGLE_VALUES})
        if set(STATISTICS).intersection(reader.fieldnames or ()):
            raise ValueError("A single-run summary must contain raw CDR/counts, without mean/std/CI columns.")
    else:
        required.update(STATISTICS)
    missing = required - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"Long-term summary is missing columns: {sorted(missing)}")
    sources = {"long_term_summary": actual,
               "run_manifest": {"path": "logs/run_manifest.json", "size_bytes": len(manifest_bytes),
                                "sha256": hashlib.sha256(manifest_bytes).hexdigest()}}
    return list(reader), manifest, sources


def _number(value) -> float:
    return math.nan if value is None or not str(value).strip() else float(value)


def _summary_index(rows: list[dict], manifest: dict) -> dict:
    if _configuration(manifest) is _config:
        return _single_run_summary_index(rows, manifest)
    return _legacy_summary_index(rows, manifest)


def _single_run_summary_index(rows: list[dict], manifest: dict) -> dict:
    """Validate one observed 200-round trajectory per method/ratio, no mean."""
    if (tuple(SEEDS) != (2026,) or REPETITIONS != 1 or TOTAL_ROUNDS != 200
            or list(DYNAMIC_CR_VALUES) != [1, 2, 3, 4, 5]
            or list(PLOT_ROUNDS) != list(range(0, 201, 20))
            or tuple(METHODS) != ("RACA-HIM", "I-EDI", "DVA-P+EDI-V")):
        raise ValueError("Single-run Figure L1 requires seed 2026, five ratios, three methods and 200 actual rounds.")
    indexed = {}
    for source in rows:
        row = dict(source)
        ratio = float(row["cr_dynamic"])
        if not math.isfinite(ratio) or not ratio.is_integer():
            raise ValueError("The single-run dynamic ratio must be an integral percentage")
        key = (int(ratio), int(row["round"]), row["method"])
        cr, t, method = key
        if (key in indexed or cr not in DYNAMIC_CR_VALUES or not 0 <= t <= TOTAL_ROUNDS
                or method not in METHODS or int(row["seed"]) != 2026
                or row["aggregation"] != "single_run"
                or row["aggregation_unit"] != "one_complete_seed_trajectory"):
            raise ValueError(f"Duplicate, mixed-seed or unexpected single-run summary point: {key}")
        if any(row[field] != str(manifest[field]) for field in IDENTITY_FIELDS):
            raise ValueError(f"Mixed source identity in single-run summary: {key}")
        counts = {field: int(row[field]) for field in COUNTS}
        if (counts["n_total"] != 1 or counts["n_valid"] not in (0, 1)
                or counts["n_failed"] != 1 - counts["n_valid"]):
            raise ValueError(f"Invalid single-trajectory validity counts: {key}")
        values = {field: _number(row[field]) for field in SINGLE_VALUES}
        denominator = _number(row["total_corruption_events"])
        if any(math.isinf(value) for value in (*values.values(), denominator)):
            raise ValueError(f"Infinite single-run observed value: {key}")
        if not counts["n_valid"]:
            if any(math.isfinite(value) for value in (*values.values(), denominator)):
                raise ValueError(f"Failed single trajectory must retain missing values: {key}")
        else:
            cdr, completed, discovered = (values[field] for field in SINGLE_VALUES)
            events = cr * _config.N // 100
            if (not all(math.isfinite(value) for value in values.values())
                    or not 0.0 <= cdr <= 1.0 or not 0 <= completed <= 98
                    or not completed.is_integer() or not discovered.is_integer()
                    or not 0 <= discovered <= events or denominator != events
                    or not math.isclose(cdr, discovered / events, rel_tol=0.0, abs_tol=1e-12)):
                raise ValueError(f"Invalid direct CDR/count values: {key}")
            if t == 0 and any(value != 0.0 for value in values.values()):
                raise ValueError(f"The saved single-run initial state must equal zero: {key}")
        row.update(counts)
        row.update(values)
        row["seed"] = 2026
        row["total_corruption_events"] = denominator
        indexed[key] = row
    expected = {(int(cr), t, method) for cr in DYNAMIC_CR_VALUES
                for t in range(TOTAL_ROUNDS + 1) for method in METHODS}
    if set(indexed) != expected:
        raise ValueError("Single-run summary must retain all 201 initial/actual round values for all fifteen trajectories")
    for cr in DYNAMIC_CR_VALUES:
        for method in METHODS:
            trace = [indexed[(int(cr), t, method)] for t in range(TOTAL_ROUNDS + 1)]
            if len({point["n_valid"] for point in trace}) != 1:
                raise ValueError(f"Single-run whole-trajectory validity changes across time: {cr}/{method}")
            finite = [point["cdr"] for point in trace if math.isfinite(point["cdr"])]
            if any(b + 1e-12 < a for a, b in zip(finite, finite[1:])):
                raise ValueError(f"Observed cumulative detection rate decreased: {cr}/{method}")
    return indexed


def _legacy_summary_index(rows: list[dict], manifest: dict) -> dict:
    """Check saved statistics and the full plotted grid without recomputing it."""
    cfg = _configuration(manifest)
    methods = cfg.METHODS
    displayed = methods if cfg is _config else LEGACY_DISPLAY_METHODS
    if (len(set(displayed)) != len(displayed) or not set(displayed).issubset(methods)):
        raise ValueError("Displayed methods must be a unique subset of the authenticated source methods")
    if (list(DYNAMIC_CR_VALUES) != [1, 2, 3, 4, 5]
            or list(PLOT_ROUNDS) != list(range(0, 201, 20))
            or list(cfg.SEEDS) != list(range(2026, 2056))
            or cfg.REPETITIONS != 30 or TOTAL_ROUNDS != 200 or len(methods) != 5):
        raise ValueError("Figure L1 must retain five ratios, 30 seeds and its authenticated protocol's methods.")
    indexed = {}
    for source in rows:
        row = dict(source)
        ratio = float(row["cr_dynamic"])
        if not math.isfinite(ratio) or not ratio.is_integer():
            raise ValueError(f"The dynamic ratio must be an integral percentage: {ratio}")
        key = (int(ratio), int(row["round"]), row["method"])
        cr, round_id, method = key
        if key in indexed:
            raise ValueError(f"Duplicate long-term summary point: {key}")
        if cr not in DYNAMIC_CR_VALUES or not 0 <= round_id <= TOTAL_ROUNDS or method not in methods:
            raise ValueError(f"Unexpected long-term summary configuration: {key}")
        if any(row[field] != str(manifest[field]) for field in IDENTITY_FIELDS):
            raise ValueError(f"Mixed run, code, algorithm, or dataset identities: {key}")
        counts = {field: int(row[field]) for field in COUNTS}
        if (counts["n_total"] != cfg.REPETITIONS or not 0 <= counts["n_valid"] <= cfg.REPETITIONS
                or counts["n_failed"] != cfg.REPETITIONS - counts["n_valid"]):
            raise ValueError(f"Inconsistent independent seed counts: {key}")
        values = {field: _number(row[field]) for field in STATISTICS}
        if any(math.isinf(value) for value in values.values()):
            raise ValueError(f"Infinite saved statistic: {key}")
        if counts["n_valid"] == 0:
            if any(math.isfinite(value) for value in values.values()):
                raise ValueError(f"No valid trajectories: statistics must remain missing: {key}")
        else:
            if not math.isfinite(values["mean_cdr"]) or not 0.0 <= values["mean_cdr"] <= 1.0:
                raise ValueError(f"A saved CDR mean must be finite in [0, 1]: {key}")
            if counts["n_valid"] >= 2 and any(not math.isfinite(values[field])
                                                   for field in STATISTICS):
                raise ValueError(f"Two or more valid trajectories require finite saved statistics: {key}")
            for field in ("std_cdr", "mean_completed_count", "mean_cumulative_detected_events"):
                if math.isfinite(values[field]) and values[field] < 0.0:
                    raise ValueError(f"A nonnegative diagnostic statistic is negative: {key}/{field}")
            low, mean, high = (values[field] for field in ("ci95_lower", "mean_cdr", "ci95_upper"))
            if math.isfinite(low) and math.isfinite(high) and not 0.0 <= low <= mean + 1e-12 <= high + 1e-12 <= 1.0 + 1e-12:
                raise ValueError(f"Saved CDR confidence interval is inconsistent: {key}")
            if round_id == 0 and any(values[field] != 0.0 for field in
                                     ("mean_cdr", "mean_cumulative_detected_events", "mean_completed_count")):
                raise ValueError(f"Round zero must contain the saved initial zero state: {key}")
        row.update(counts)
        row.update(values)
        indexed[key] = row
    expected = {(int(cr), int(round_id), method) for cr in DYNAMIC_CR_VALUES
                for round_id in PLOT_ROUNDS for method in methods}
    if not expected.issubset(indexed):
        raise ValueError(f"Missing saved Figure L1 points: {sorted(expected - set(indexed))[:10]}")
    # Full-trajectory validity preserves one set of independent seeds across
    # time. Never repair a declining curve by sorting, interpolation or clipping.
    for cr in DYNAMIC_CR_VALUES:
        for method in methods:
            series = [indexed[(int(cr), int(t), method)] for t in PLOT_ROUNDS]
            if len({point["n_valid"] for point in series}) != 1:
                raise ValueError(f"Time points must aggregate the same complete seed trajectories: {cr}/{method}")
            finite = [point["mean_cdr"] for point in series if math.isfinite(point["mean_cdr"])]
            if any(b + 1e-12 < a for a, b in zip(finite, finite[1:])):
                raise ValueError(f"Cumulative mean detection rate decreased: {cr}/{method}")
    return indexed


def _draw_figure(indexed: dict, display_methods, *, single_run: bool):
    value_field = "cdr" if single_run else "mean_cdr"
    evidence_fields = (*COUNTS, *SINGLE_VALUES, "total_corruption_events", "seed", "aggregation", "aggregation_unit") if single_run else (*COUNTS, *STATISTICS)
    fig = plt.figure(figsize=(DETECTION_FIGURE_WIDTH, DETECTION_FIGURE_HEIGHT))
    width_pt, height_pt = fig.get_size_inches() * 72.0
    panel_height_pt = (DETECTION_UPPER_HEIGHT_PT + DETECTION_LOWER_HEIGHT_PT
                       + DETECTION_BREAK_GAP_PT)
    body_width_pt = 5 * DETECTION_PANEL_WIDTH_PT + 4 * DETECTION_PANEL_GAP_PT
    bottom = DETECTION_AXES_BOTTOM_PT / height_pt
    top = (DETECTION_AXES_BOTTOM_PT + panel_height_pt) / height_pt
    grid = fig.add_gridspec(1, 5, left=DETECTION_AXES_LEFT_PT / width_pt,
                           right=(DETECTION_AXES_LEFT_PT + body_width_pt) / width_pt,
                           bottom=bottom, top=top,
                           wspace=DETECTION_PANEL_GAP_PT / DETECTION_PANEL_WIDTH_PT)
    axes = []
    evidence = []
    for panel, cr in enumerate(DYNAMIC_CR_VALUES):
        ax = fig.add_subplot(grid[0, panel], sharey=axes[0] if axes else None)
        axes.append(ax)
        for method in display_methods:
            values = [indexed[(int(cr), int(t), method)][value_field] for t in PLOT_ROUNDS]
            ax.plot(PLOT_ROUNDS, values, **_style(method),
                    markerfacecolor="white", markeredgewidth=0.8,
                    label=_display_name(method), clip_on=False, zorder=3)
        # Every one of the eleven genuine measurements retains its marker.
        # Sparse numeric tick labels avoid collisions in 82.7-point panels.
        ax.set_xlim(0.0, float(TOTAL_ROUNDS))
        ax.set_xticks(LABELED_ROUNDS)
        ax.set_xticks([t for t in PLOT_ROUNDS if t not in LABELED_ROUNDS], minor=True)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(axis="both", which="major", pad=2.0, labelsize=DETECTION_TICK_FONT_PT)
        ax.tick_params(axis="x", which="minor", length=1.6, width=0.5)
        ax.tick_params(axis="y", labelleft=panel == 0)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D3D3D3", linewidth=0.4)
        ax.set_xlabel("Verification Round", labelpad=2.0, fontsize=DETECTION_LABEL_FONT_PT)
        position = ax.get_position()
        fig.text((position.x0 + position.x1) / 2.0, DETECTION_CAPTION_CENTER_PT / height_pt,
                 f"({chr(97 + panel)}) $cr_{{dynamic}}$ = {cr:g}%",
                 ha="center", va="center", fontsize=DETECTION_LABEL_FONT_PT)
        for round_id in PLOT_ROUNDS:
            evidence.append({"cr_dynamic": int(cr), "round": int(round_id), "methods": {
                method: {field: (row[field] if not isinstance(row[field], float)
                                  or math.isfinite(row[field]) else None)
                         for field in evidence_fields}
                for method in display_methods for row in [indexed[(int(cr), int(round_id), method)]]}})
    fig.text(DETECTION_YLABEL_CENTER_X_PT / width_pt, (bottom + top) / 2.0,
             "Cumulative Corruption\nDetection Rate", rotation=90,
             ha="center", va="center", fontsize=DETECTION_LABEL_FONT_PT)
    handles = [Line2D([], [], label=_display_name(method),
                      **_style(method),
                      markerfacecolor="white", markeredgewidth=0.8) for method in display_methods]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.525, top + DETECTION_LEGEND_GAP_PT / height_pt),
               ncol=len(display_methods), frameon=False, handlelength=2.0, columnspacing=2.0,
               borderaxespad=0.0, borderpad=0.1, fontsize=DETECTION_TICK_FONT_PT)
    return fig, {"per_point_evidence": evidence,
                 "panel_height_pt": float(panel_height_pt), "y_limits": [0.0, 1.0]}


def _save_figure(fig, path: Path, **kwargs) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        # Fixed physical size: bbox_inches='tight' would change the paper page.
        fig.savefig(temporary, format=path.suffix.lstrip("."), **kwargs)
        temporary.replace(path)
    except PermissionError as exc:
        raise PermissionError(f"Cannot replace {path}; close its PDF/image preview and rerun this plot entry.") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _preserve_previous_presentation(result_root: Path, display_methods,
                                    source_manifest: dict, summary_sha256: str) -> dict | None:
    """Preserve the previous Figure L1 as files in the existing figure folder.

    Authenticate available old outputs, retaining their original provenance.
    Removed old images are recorded as missing and never block regeneration.
    A committed snapshot also permits retries after only one new image saved.
    """
    manifest_path = result_root / "logs" / "figure_manifest.json"
    if not manifest_path.is_file():
        return None
    original_bytes = manifest_path.read_bytes()
    previous = json.loads(original_bytes.decode("utf-8-sig"))
    previous_methods = previous.get("display_methods", previous.get("methods", []))
    if (previous_methods == list(display_methods)
            and previous.get("run_id") == source_manifest["run_id"]
            and previous.get("display_selection_version") == DISPLAY_SELECTION_VERSION
            and previous.get("source_files", {}).get("long_term_summary", {}).get("sha256") == summary_sha256):
        return previous.get("previous_presentation_archive")
    digest = hashlib.sha256(original_bytes).hexdigest()
    folder = result_root / "figures"
    stem = f"{FIGURE_NAME}_previous_{digest}"
    copied = {}
    missing = []
    snapshot = folder / f"{stem}.figure_manifest.json"
    committed = snapshot.is_file()
    # Commit the old manifest only after examining both old images. On retry,
    # an absent backup stays absent: the active path may already contain one
    # newly rendered image and must not be mistaken for the previous output.
    if committed:
        if snapshot.read_bytes() != original_bytes:
            raise ValueError("Existing figure archive manifest differs from its named checksum")
    for extension in ("pdf", "png"):
        source = result_root / "figures" / f"{FIGURE_NAME}.{extension}"
        destination = folder / f"{stem}.{extension}"
        declared = previous.get("files", {}).get(extension)
        if not isinstance(declared, dict):
            raise ValueError(f"Old figure manifest is missing the {extension} identity; refusing unverified archival")
        # An individually verified archived member can also survive an earlier
        # interrupted copy. Never replace it using a partially new active file.
        if destination.is_file():
            identity = _identity(destination, result_root)
            if any(identity[field] != declared.get(field) for field in ("sha256", "size_bytes")):
                raise ValueError(f"Existing figure archive differs: {destination}")
        else:
            if committed or not source.is_file():
                missing.append({"extension": extension,
                                "path": source.relative_to(result_root).as_posix(),
                                "reason": ("previous_image_not_available_after_archive_commit"
                                           if committed else "previous_image_missing")})
                continue
            payload = source.read_bytes()
            if (hashlib.sha256(payload).hexdigest() != declared.get("sha256")
                    or len(payload) != declared.get("size_bytes")):
                raise ValueError(f"Existing {source.name} differs from its old figure manifest")
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_bytes(payload)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            identity = _identity(destination, result_root)
            if any(identity[field] != declared.get(field) for field in ("sha256", "size_bytes")):
                raise ValueError(f"Figure changed while preserving its previous presentation: {source}")
        copied[extension] = identity
    if snapshot.exists() and snapshot.read_bytes() != original_bytes:
        raise ValueError("Existing figure manifest archive differs from its named checksum")
    if not snapshot.exists():
        temporary = snapshot.with_name(f".{snapshot.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(original_bytes)
            temporary.replace(snapshot)
        finally:
            temporary.unlink(missing_ok=True)
    if snapshot.read_bytes() != original_bytes or manifest_path.read_bytes() != original_bytes:
        raise ValueError("Figure manifest changed while preserving its previous presentation")
    if missing:
        print("Previous figure files unavailable: "
              + ", ".join(item["extension"] for item in missing)
              + ". Continuing regeneration from the authenticated saved summary.", flush=True)
    return {"path": folder.relative_to(result_root).as_posix(), "filename_prefix": stem,
            "status": "complete" if not missing else ("partial" if copied else "metadata_only"),
            "missing_files": missing,
            "display_methods": previous_methods, "files": copied,
            "figure_manifest": _identity(snapshot, result_root)}


def plot_long_term(result_root=None, *, output_root=None) -> dict:
    """Write the sole Figure L1 and provenance from saved CSV/manifest only."""
    explicit_source = result_root is not None
    result_root = Path(_default_result_root() if result_root is None else result_root).resolve()
    rows, manifest, sources = _read_sources(result_root)
    cfg = _configuration(manifest)
    legacy = cfg is _legacy_config
    if legacy and not explicit_source:
        raise ValueError("results/06_long_term currently contains an earlier adapted run. "
                         "Run run_long_term.py to create the new single-run result first; "
                         "use --legacy only when explicitly requesting the historical figure.")
    single_run = not legacy
    display_methods = LEGACY_DISPLAY_METHODS if legacy else cfg.METHODS
    if output_root is None:
        output_root = FIGURE_OUTPUT_ROOT
    output_root = Path(output_root).resolve()
    if output_root != FIGURE_OUTPUT_ROOT.resolve():
        raise ValueError("Figure L1 must be written to the existing results/06_long_term/figures directory")
    folder = output_root / "figures"
    log_folder = output_root / "logs"
    if not folder.is_dir() or not log_folder.is_dir():
        raise FileNotFoundError("The existing results/06_long_term/figures and logs directories are required; "
                                "the plot entry does not create output directories")
    print(f"Reading {'LEGACY ADAPTED 30-SEED MEANS' if legacy else 'SINGLE RUN, SEED 2026'} results: {result_root}\n"
          f"Writing Figure L1: {folder}", flush=True)
    indexed = _summary_index(rows, manifest)
    value_field = "cdr" if single_run else "mean_cdr"
    if not any(math.isfinite(row[value_field]) for key, row in indexed.items()
               if key[2] in display_methods):
        raise ValueError("All trajectories failed; inspect raw records and failures before attempting Figure L1.")
    font = apply_style()
    previous_presentation = _preserve_previous_presentation(
        output_root, display_methods, manifest, sources["long_term_summary"]["sha256"])
    fig, info = _draw_figure(indexed, display_methods, single_run=single_run)
    paths = {extension: folder / f"{FIGURE_NAME}.{extension}" for extension in ("pdf", "png")}
    try:
        _save_figure(fig, paths["pdf"], metadata={
            "Title": "Long-Term Dynamic Verification Performance",
            "Subject": (("Single run at seed 2026; direct observed event CDR; no averaging/std/CI; " if single_run
                         else "Historical 30-seed mean event CDR; CI retained in the historical CSV; ")
                        + "fixed all-event denominator; "
                        + f"run_id={manifest['run_id']}; summary_sha256={sources['long_term_summary']['sha256']}"),
        })
        _save_figure(fig, paths["png"], dpi=PNG_DPI)
    finally:
        plt.close(fig)
    metadata = {
        **{field: manifest[field] for field in IDENTITY_FIELDS},
        "generated_at": datetime.now(timezone(timedelta(hours=8), "Asia/Shanghai")).isoformat(),
        "timezone": "Asia/Shanghai", "font": font, "source_files": sources,
        "source_result_root": str(result_root), "output_root": str(output_root),
        "protocol_family": manifest.get("protocol_family", "legacy-adapted-five-method-run"),
        "aggregation": "single_run" if single_run else "historical_seed_mean",
        "aggregation_unit": "one_complete_seed_trajectory" if single_run else "independent_complete_seed_trajectories",
        "value_field": value_field,
        **({"seed": 2026, "single_seed": 2026} if single_run else {}),
        "source_simulation_files_read_only": True,
        "legacy_source_read_only": legacy and output_root != result_root,
        "figure_output_policy": "fixed existing results/06_long_term/figures; no new output directories",
        "figure": FIGURE_NAME, "figure_output_root": str(folder),
        "figure_width_inches": DETECTION_FIGURE_WIDTH, "figure_height_inches": DETECTION_FIGURE_HEIGHT,
        "panel_width_pt": DETECTION_PANEL_WIDTH_PT, "panel_gap_pt": DETECTION_PANEL_GAP_PT,
        "axes_left_pt": DETECTION_AXES_LEFT_PT, "axes_bottom_pt": DETECTION_AXES_BOTTOM_PT,
        "shared_legend_gap_pt": DETECTION_LEGEND_GAP_PT,
        "panel_caption_center_pt": DETECTION_CAPTION_CENTER_PT,
        "panel_geometry_reference": "Groups 02-05 complete continuous panel envelope",
        "methods": list(display_methods),
        "source_methods": list(cfg.METHODS), "display_methods": list(display_methods),
        "omitted_methods": [method for method in cfg.METHODS if method not in display_methods],
        "display_selection_version": DISPLAY_SELECTION_VERSION,
        "selection_reason": "Three-method native-protocol comparison; Age-Only and Random omitted at the user's request",
        "comparison_scope": ("legacy adapted end-to-end results, preserved under their old identity" if legacy
                             else "native I-EDI Algorithm2 and DVA-P+EDI-V proof protocols on the disclosed TNSE descriptor workload"),
        "baseline_fidelity": ({"I-EDI-Adapted": "legacy fixed-modulus tag with QoS/capacity adapters",
                               "DVA-P-Adapted": "legacy subset scheduling with a SHA256 wrapper; no Merkle proof"} if legacy else {
                               "I-EDI": "per-target per-round random-modulus Algorithm2 tag response; disclosed eta domain and workload QoS",
                               "DVA-P+EDI-V": "DVA-P subset scheduling plus fresh-nonce EDI-V VMHT proofs; disclosed block/probability settings"}),
        "previous_presentation_archive": previous_presentation,
        "legend_labels": [_display_name(method) for method in display_methods],
        "method_styles": {method: _style(method) for method in display_methods},
        "x_coordinates": list(PLOT_ROUNDS), "labeled_x_ticks": list(LABELED_ROUNDS),
        "repetitions": cfg.REPETITIONS, "png_dpi": PNG_DPI, "pdf_vector": True,
        "broken_axis": False, "shaded_bands": False, "line_styles": "all solid",
        "confidence_interval_display": "not computed for a single run" if single_run else "historical CSV only",
        "averaging": not single_run, "standard_deviation_computed": not single_run,
        "confidence_interval_computed": not single_run, "smoothing": False,
        "interpolation": False, "artificial_jitter": False,
        "missing_value_policy": "Missing trajectory values remain NaN line gaps; never filled or interpolated",
        "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plot_style_source_sha256": hashlib.sha256(
            (PROJECT_ROOT / "experiments/common/plotting_style.py").read_bytes()).hexdigest(),
        "files": {extension: _identity(path, output_root) for extension, path in paths.items()},
        **info,
        "visual_inspection": "Not performed by this script; inspect Figure L1 after the user runs it.",
    }
    _atomic_json(output_root / "logs" / "figure_manifest.json", metadata)
    return metadata


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Replot saved Group 06 Figure L1 without running algorithms")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--legacy", action="store_true", help="Read legacy adapted results; save Figure L1 in results/06_long_term/figures")
    source.add_argument("--result-root", type=Path, help="Select saved input data; Figure L1 still goes to results/06_long_term/figures")
    args = parser.parse_args(argv)
    selected = _legacy_config.RESULT_ROOT if args.legacy else args.result_root
    if args.legacy:
        manifest_path = selected / "logs" / "run_manifest.json"
        if manifest_path.is_file():
            saved = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if saved.get("protocol_family") == _config.PROTOCOL_FAMILY:
                parser.error("The active results folder now contains native single-run data. "
                             "Omit --legacy to plot it; use --result-root for separately preserved historical data.")
    metadata = plot_long_term(selected)
    print(f"Saved Figure L1 vector PDF and {PNG_DPI} dpi PNG in {metadata['figure_output_root']}.", flush=True)


if __name__ == "__main__":
    main()
