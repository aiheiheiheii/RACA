"""Binary corruption metrics; undefined zero-denominator rates remain NaN."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

from .config import SUBTYPE_NAMES


def _binary(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.all(np.isin(array, (0, 1))):
        raise ValueError(f"{name} must be a one-dimensional array encoded as 0/1")
    return array.astype(np.int8, copy=False)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int | float | str]:
    """Return counts and rates with corrupted=1.

    A zero denominator produces NaN, never a fabricated perfect score.
    ``undefined_metrics`` lists those rates in a fixed order. F1 uses
    2*TP/(2*TP+FP+FN), the count form of the harmonic mean; thus F1 is zero
    when errors occur without a true positive, and NaN only if this count
    denominator is zero (e.g. no actual or predicted positive).
    """
    truth = _binary(y_true, "y_true")
    prediction = _binary(y_pred, "y_pred")
    if truth.shape != prediction.shape:
        raise ValueError("Truth and prediction arrays must have the same shape")
    tp = int(np.count_nonzero((truth == 1) & (prediction == 1)))
    tn = int(np.count_nonzero((truth == 0) & (prediction == 0)))
    fp = int(np.count_nonzero((truth == 0) & (prediction == 1)))
    fn = int(np.count_nonzero((truth == 1) & (prediction == 0)))

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else math.nan

    rates = {
        "accuracy": rate(tp + tn, len(truth)),
        "precision": rate(tp, tp + fp),
        "recall": rate(tp, tp + fn),
        "f1": rate(2 * tp, 2 * tp + fp + fn),
        "fpr": rate(fp, fp + tn),
        "fnr": rate(fn, fn + tp),
    }
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        **rates,
        "undefined_metrics": ";".join(key for key, value in rates.items() if math.isnan(value)),
    }


def subtype_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, subtypes: np.ndarray
) -> list[dict[str, int | float | str]]:
    """Four subtype rows, including empty strata with NaN error rates."""
    truth = _binary(y_true, "y_true")
    prediction = _binary(y_pred, "y_pred")
    labels = np.asarray(subtypes)
    if truth.shape != prediction.shape or labels.shape != truth.shape:
        raise ValueError("Truth, predictions and subtype labels must have the same 1D shape")
    if not np.all(np.isin(labels, (0, 1, 2, 3))):
        raise ValueError("Subtype labels must be encoded as 0, 1, 2, 3")
    rows = []
    for subtype, name in enumerate(SUBTYPE_NAMES):
        mask = labels == subtype
        count = int(np.count_nonzero(mask))
        errors = int(np.count_nonzero(truth[mask] != prediction[mask]))
        rows.append({
            "subtype": subtype,
            "name": name,
            "count": count,
            "errors": errors,
            "error_rate": errors / count if count else math.nan,
            **binary_metrics(truth[mask], prediction[mask]),
        })
    return rows

