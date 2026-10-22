"""Paper-wide visual style, checked against the supplied TMC PDF.

The four original fill RGB values map exactly to the tab10 hex colors below.
RACA-HIM inherits CoAPM's blue for visual continuity only, not its algorithm.
TMC pages 8/9 define the four method colors and the detection broken-axis bars.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from experiments.common.method_names import RACA_HIM, canonical_method_name

METHOD_COLORS = {
    RACA_HIM: "#1F77B4", "DVA-P": "#8C564B", "I-EDI": "#FF7F0E",
    "CTCV": "#2CA02C", "DynaEDI+": "#D62728",
    "RF-TS": "#17BECF", "Cost-First": "#7F7F7F", "Random": "#333333",
    "Age-Only": "#BCBD22",
}
METHOD_MARKERS = {
    RACA_HIM: "o", "DVA-P": "s", "I-EDI": "^", "CTCV": "D",
    "DynaEDI+": "v", "RF-TS": "P",
    "Cost-First": "X", "Random": "x", "Age-Only": "*",
}
METHOD_LINESTYLES = {name: "-" for name in METHOD_COLORS}
# Group 04 has distinct GT construction controls.  Keep their stable palette
# separate from the detector/budget methods so existing paper figures retain
# exactly the same colors, markers, and styles.
GT_METHOD_COLORS = {
    RACA_HIM: METHOD_COLORS[RACA_HIM],
    "Fixed-K Majority": "#9467BD",
    "Top-M": "#E377C2",
    "Random-M": "#A6761D",
}
GT_METHOD_MARKERS = {
    RACA_HIM: "o", "Fixed-K Majority": "s", "Top-M": "^", "Random-M": "D",
}
GT_METHOD_LINESTYLES = {name: "-" for name in GT_METHOD_COLORS}
FIGURE_WIDTH = 7.16
FIGURE_HEIGHT = 2.15
PNG_DPI = 600
DETECTION_LEGEND_GAP_PT = 3.0
# Geometry measured from TMC Fig. 3 (PDF page 9). Accuracy is a fraction.
DETECTION_FIGURE_WIDTH = 7.16
DETECTION_FIGURE_HEIGHT = 1.50
DETECTION_PANEL_WIDTH_PT = 82.7057809082
DETECTION_PANEL_GAP_PT = 14.8870400826
DETECTION_UPPER_HEIGHT_PT = 10.0304168914
DETECTION_LOWER_HEIGHT_PT = 52.5121816062
DETECTION_AXES_LEFT_PT = 33.5
DETECTION_AXES_BOTTOM_PT = 31.0
DETECTION_CAPTION_CENTER_PT = 5.5
DETECTION_YLABEL_CENTER_X_PT = 7.8665
DETECTION_TICK_FONT_PT = 5.8288
DETECTION_LABEL_FONT_PT = 6.5574
DETECTION_LOWER_YLIM = (0.0, 0.18)
DETECTION_UPPER_YLIM = (0.94, 1.02)
DETECTION_HEIGHT_RATIOS = (DETECTION_UPPER_HEIGHT_PT, DETECTION_LOWER_HEIGHT_PT)
DETECTION_BREAK_GAP_PT = 1.7199215024
DETECTION_BREAK_LINEWIDTH = 0.583
DETECTION_BREAK_HALF_WIDTH_PT = 2.186
DETECTION_BREAK_HALF_HEIGHT_PT = 1.093
METRIC_STYLES = {
    "accuracy": {"color": METHOD_COLORS[RACA_HIM], "marker": "o", "linestyle": "-"},
    "f1": {"color": METHOD_COLORS["I-EDI"], "marker": "s", "linestyle": "-"},
    "fpr": {"color": METHOD_COLORS["CTCV"], "marker": "^", "linestyle": "-"},
    "fnr": {"color": METHOD_COLORS["DynaEDI+"], "marker": "D", "linestyle": "-"},
}
ANNOTATION_COLOR = "#555555"


def apply_style() -> str:
    """Select an available serif font and consistent IEEE-sized typography."""
    names = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in ("Times New Roman", "Liberation Serif", "DejaVu Serif")
                     if name in names), "DejaVu Serif")
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": [selected], "font.size": 7,
        "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "legend.fontsize": 7, "axes.linewidth": .6,
        "lines.linewidth": 1.2, "lines.markersize": 3.2,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    })
    return selected


def method_style(method: str) -> dict:
    key = canonical_method_name(method).removesuffix("-Adapted")
    return {"color": METHOD_COLORS[key], "marker": METHOD_MARKERS[key],
            "linestyle": METHOD_LINESTYLES[key]}
