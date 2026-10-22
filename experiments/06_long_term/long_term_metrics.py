"""Offline event-level evaluation and seed-level long-term summaries.

The evaluator is owned by the experiment runner, never by a scheduling method.
Its complete event denominator and all onset/recovery times remain offline.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math

import numpy as np
from scipy.stats import t as student_t


FAILED_STATUSES = frozenset({"failed", "error", "exception", "aborted", "incomplete", "not_run",
                             "runtime_error"})


def _field(value, name: str, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _bool(value) -> bool:
    if isinstance(value, str):
        if value.strip().lower() in {"true", "1"}:
            return True
        if value.strip().lower() in {"false", "0", "", "none", "nan"}:
            return False
        raise ValueError(f"Cannot interpret boolean field: {value!r}")
    return bool(value)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values, dtype=np.float64)) if len(values) else math.nan


class OfflineEventEvaluator:
    """Count each correctly detected active event once, over fixed full-horizon E."""

    def __init__(self, case: Mapping, *, method: str, seed: int, cr_dynamic: float):
        ids = np.asarray(case["event_model_ids"], dtype=np.int64)
        onsets = np.asarray(case["onset_round"], dtype=np.int64)
        recoveries = np.asarray(case["recovery_round"], dtype=np.int64)
        if (ids.ndim != 1 or len(np.unique(ids)) != len(ids)
                or onsets.shape != ids.shape or recoveries.shape != ids.shape):
            raise ValueError("Evaluation requires one correctly paired event per model identity")
        self._ids, self._onsets, self._recoveries = ids.copy(), onsets.copy(), recoveries.copy()
        self._event_for_model = {int(i): j for j, i in enumerate(ids)}
        self._first_detection = np.full(len(ids), -1, dtype=np.int64)
        self._rounds = len(case["corrupted_mask"])
        self._last_round = 0
        self._round_false_positives = 0
        self._total_false_positives = 0
        self._context = {"method": str(method), "seed": int(seed), "cr_dynamic": float(cr_dynamic)}

    @property
    def total_events(self) -> int:
        return len(self._ids)

    @property
    def last_round(self) -> int:
        """Last committed evaluator round, for driver exception handling."""
        return self._last_round

    @property
    def cumulative_detected_events(self) -> int:
        return int(np.count_nonzero(self._first_detection >= 0))

    def snapshot(self) -> dict:
        count = self.cumulative_detected_events
        return {"round_new_detected_events": 0,
                "cumulative_detected_events": count,
                "total_corruption_events": self.total_events,
                "cumulative_detection_rate": count / self.total_events if self.total_events else math.nan,
                "round_false_positives": 0,
                "round_true_positives": 0}

    def consume_round(self, round_id: int, outcomes: Sequence) -> dict:
        """Consume only selected-target outcomes, never candidate observations."""
        t = int(round_id)
        if t != self._last_round + 1 or t > self._rounds:
            raise ValueError("Offline evaluation requires all consecutive rounds exactly once")
        seen: set[int] = set()
        newly_detected, false_positives, true_positives = 0, 0, 0
        # Stage discoveries so malformed input cannot partially mutate the
        # evaluator before the driver records an unexpected method failure.
        discoveries: list[int] = []
        for outcome in outcomes:
            target = int(_field(outcome, "target_id"))
            if target in seen:
                raise ValueError("A target was verified twice in one method-round")
            seen.add(target)
            # A failed or invalid interaction never constitutes an event detection.
            if (not _bool(_field(outcome, "completed", False))
                    or not _bool(_field(outcome, "attempted", True))
                    or str(_field(outcome, "outcome", "")).lower() != "corrupted"):
                continue
            event = self._event_for_model.get(target)
            active = (event is not None
                      and int(self._onsets[event]) <= t < int(self._recoveries[event]))
            if not active:
                false_positives += 1
                continue
            true_positives += 1
            if self._first_detection[event] < 0:
                discoveries.append(event)
                newly_detected += 1
        self._first_detection[discoveries] = t
        self._last_round = t
        self._round_false_positives = false_positives
        self._total_false_positives += false_positives
        result = self.snapshot()
        result.update(round_new_detected_events=newly_detected,
                      round_false_positives=false_positives, round_true_positives=true_positives)
        return result

    def event_rows(self, *, trajectory_status: str = "ok") -> list[dict]:
        """Unseen events retain missing detection times and explicit censoring."""
        rows = []
        for event, (model, onset, recovery, first) in enumerate(zip(
                self._ids, self._onsets, self._recoveries, self._first_detection)):
            detected = bool(first >= 0)
            if detected:
                status = "detected"
            elif trajectory_status in FAILED_STATUSES:
                status = "undetected_run_failed"
            elif recovery <= self._rounds:
                status = "undetected_recovered"
            else:
                status = "undetected_at_horizon"
            censor_exclusive = min(int(recovery), self._last_round + 1)
            rows.append({
                **self._context, "event_id": int(event), "model_id": int(model),
                "onset_round": int(onset), "recovery_round": int(recovery),
                "detected": detected, "first_detection_round": int(first) if detected else None,
                "detection_delay": int(first - onset) if detected else None,
                "status": status, "trajectory_status": trajectory_status,
                "censored": not detected,
                "censor_round_exclusive": censor_exclusive if not detected else None,
                "observed_active_rounds": max(0, censor_exclusive - int(onset)),
                "evaluation_last_round": self._last_round,
            })
        return rows

    # A descriptive alias for callers that prefer a plural name.
    events_rows = event_rows

    def final_summary(self, round_rows: Sequence[Mapping] = (), *,
                      trajectory_status: str = "ok") -> dict:
        found = self._first_detection >= 0
        delays = self._first_detection[found] - self._onsets[found]
        count = int(np.count_nonzero(found))
        selected = sum(int(row.get("selected_count", 0)) for row in round_rows)
        budget = sum(int(row.get("budget", 0)) for row in round_rows)
        completed = sum(int(row.get("completed_count", 0)) for row in round_rows)
        attempted = sum(int(row.get("attempted_count", 0)) for row in round_rows)
        return {**self._context, "total_corruption_events": self.total_events,
                "detected_events": count, "undetected_events": self.total_events - count,
                "final_detection_rate": count / self.total_events if self.total_events else math.nan,
                "mean_detection_delay": _mean(delays),
                "missing_detection_delays": self.total_events - count,
                "total_selected_count": selected, "total_attempted_count": attempted,
                "total_completed_count": completed,
                "verification_budget_utilization": selected / budget if budget else math.nan,
                "completion_budget_utilization": completed / budget if budget else math.nan,
                "false_positive_count": self._total_false_positives,
                "evaluation_last_round": self._last_round,
                "trajectory_status": trajectory_status}


def _trajectory_groups(round_rows: Sequence[Mapping]) -> dict:
    groups: dict[tuple, dict[int, Mapping]] = defaultdict(dict)
    for row in round_rows:
        key = (float(row["cr_dynamic"]), str(row["method"]), int(row["seed"]))
        round_id = int(row["round"])
        if round_id in groups[key]:
            raise ValueError(f"Duplicate method-seed-round result: {key}, round={round_id}")
        groups[key][round_id] = row
    return dict(groups)


def _trajectory_valid(rows: Mapping[int, Mapping], total_rounds: int) -> bool:
    if set(rows) != set(range(1, total_rounds + 1)):
        return False
    previous, denominator = 0.0, None
    for round_id in range(1, total_rounds + 1):
        row = rows[round_id]
        if str(row.get("status", "ok")).strip().lower() in FAILED_STATUSES:
            return False
        if not _finite(row.get("cumulative_detection_rate")):
            return False
        cdr = float(row["cumulative_detection_rate"])
        events = int(row["total_corruption_events"])
        detected = int(row["cumulative_detected_events"])
        if (events < 1 or not 0 <= detected <= events or cdr < previous - 1e-12
                or not 0 <= cdr <= 1
                or not math.isclose(cdr, detected / events, rel_tol=1e-12, abs_tol=1e-12)):
            return False
        if denominator is not None and events != denominator:
            return False
        denominator, previous = events, cdr
    return True


def summarize_rounds(round_rows: Sequence[Mapping], expected_seeds: Sequence[int],
                     plot_rounds: Sequence[int] | None = None, *,
                     total_rounds: int = 200,
                     methods: Sequence[str] | None = None,
                     cr_values: Sequence[float] | None = None) -> list[dict]:
    """Aggregate independent complete seed trajectories, including defined t=0.

    ``plot_rounds`` validates the requested display points; summary retains every
    true round 0..T. Runtime failures invalidate the whole method/seed trace for
    summary means, keeping the same contributing seeds at every round and thus
    preserving monotonicity. Failed traces remain intact in raw outputs and in
    n_failed; one method's failure never invalidates another method's seed.
    """
    seeds = tuple(map(int, expected_seeds))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Expected seeds must be a nonempty unique sequence")
    if plot_rounds is not None and not set(map(int, plot_rounds)).issubset(range(total_rounds + 1)):
        raise ValueError("Plot rounds must be observed rounds within 0..T")
    groups = _trajectory_groups(round_rows)
    extra_seeds = {key[2] for key in groups}.difference(seeds)
    if extra_seeds:
        raise ValueError(f"Raw results contain unpublished seeds: {sorted(extra_seeds)}")
    method_order = list(methods) if methods is not None else sorted({key[1] for key in groups})
    ratios = list(map(float, cr_values)) if cr_values is not None else sorted({key[0] for key in groups})
    summaries = []
    for ratio in ratios:
        for method in method_order:
            valid_seeds = [seed for seed in seeds
                           if _trajectory_valid(groups.get((ratio, method, seed), {}), total_rounds)]
            n_valid = len(valid_seeds)
            for round_id in range(total_rounds + 1):
                if round_id == 0:
                    values = np.zeros(n_valid, dtype=np.float64)
                    completed, discovered = [0.0] * n_valid, [0.0] * n_valid
                else:
                    records = [groups[(ratio, method, seed)][round_id] for seed in valid_seeds]
                    values = np.asarray([float(row["cumulative_detection_rate"]) for row in records])
                    completed = [float(row["completed_count"]) for row in records]
                    discovered = [float(row["cumulative_detected_events"]) for row in records]
                mean = _mean(values)
                std = float(np.std(values, ddof=1)) if n_valid >= 2 else math.nan
                half = (float(student_t.ppf(0.975, n_valid - 1)) * std / math.sqrt(n_valid)
                        if n_valid >= 2 else math.nan)
                summaries.append({
                    "cr_dynamic": ratio, "round": round_id, "method": method,
                    "mean_cdr": mean, "std_cdr": std,
                    "ci95_lower": max(0.0, mean - half) if math.isfinite(half) else math.nan,
                    "ci95_upper": min(1.0, mean + half) if math.isfinite(half) else math.nan,
                    "n_total": len(seeds), "n_valid": n_valid, "n_failed": len(seeds) - n_valid,
                    "mean_completed_count": _mean(completed),
                    "mean_cumulative_detected_events": _mean(discovered),
                    "valid_seeds": ";".join(map(str, valid_seeds)),
                    "failed_seeds": ";".join(str(seed) for seed in seeds if seed not in valid_seeds),
                    "aggregation_unit": "independent_complete_seed_trajectory",
                })
    return summaries


def summarize_events(event_rows: Sequence[Mapping], round_rows: Sequence[Mapping],
                     expected_seeds: Sequence[int] | None = None, *,
                     total_rounds: int = 200) -> list[dict]:
    """Diagnostic final-event/delay summary, preserving failed and missed counts."""
    trajectories = _trajectory_groups(round_rows)
    events: dict[tuple, list[Mapping]] = defaultdict(list)
    seen_events: set[tuple] = set()
    for row in event_rows:
        key = (float(row["cr_dynamic"]), str(row["method"]), int(row["seed"]))
        event_key = (*key, int(row["event_id"]))
        if event_key in seen_events:
            raise ValueError(f"Duplicate event record: {event_key}")
        seen_events.add(event_key)
        detected = _bool(row["detected"])
        delay = row.get("detection_delay")
        if detected and (not _finite(delay) or float(delay) < 0):
            raise ValueError("Detected events require a finite nonnegative detection delay")
        if detected:
            first = row.get("first_detection_round")
            onset, recovery = int(row["onset_round"]), int(row["recovery_round"])
            if (not _finite(first) or not onset <= float(first) < recovery
                    or float(first) - onset != float(delay)):
                raise ValueError("Detection must occur within the active event window with matching delay")
        if not detected and _finite(delay):
            raise ValueError("Undetected events must retain missing detection delays")
        if not detected and _finite(row.get("first_detection_round")):
            raise ValueError("Undetected events cannot have a finite first-detection round")
        events[key].append(row)
    pairs = sorted({key[:2] for key in trajectories} | {key[:2] for key in events})
    all_seeds = tuple(map(int, expected_seeds)) if expected_seeds is not None else tuple(
        sorted({key[2] for key in trajectories} | {key[2] for key in events}))
    result = []
    for ratio, method in pairs:
        valid = [seed for seed in all_seeds
                 if _trajectory_valid(trajectories.get((ratio, method, seed), {}), total_rounds)]
        valid_events = [row for seed in valid for row in events.get((ratio, method, seed), [])]
        raw_events = [row for seed in all_seeds for row in events.get((ratio, method, seed), [])]
        detected_rows = [row for row in valid_events if _bool(row["detected"])]
        seed_rates, seed_delay_means, budget_utilization = [], [], []
        selected = completed = attempted = budget = 0
        for seed in valid:
            records = trajectories[(ratio, method, seed)]
            final = records[total_rounds]
            seed_events = events.get((ratio, method, seed), [])
            if len(seed_events) != int(final["total_corruption_events"]):
                raise ValueError("A complete trajectory is missing its full event table")
            if sum(_bool(row["detected"]) for row in seed_events) != int(final["cumulative_detected_events"]):
                raise ValueError("Final cumulative detection count disagrees with event records")
            seed_rates.append(float(final["cumulative_detection_rate"]))
            delays = [float(row["detection_delay"]) for row in seed_events if _bool(row["detected"])]
            if delays:
                seed_delay_means.append(_mean(delays))
            seed_budget = sum(int(row["budget"]) for row in records.values())
            seed_selected = sum(int(row["selected_count"]) for row in records.values())
            selected += seed_selected
            completed += sum(int(row["completed_count"]) for row in records.values())
            attempted += sum(int(row["attempted_count"]) for row in records.values())
            budget += seed_budget
            budget_utilization.append(seed_selected / seed_budget if seed_budget else math.nan)
        result.append({
            "cr_dynamic": ratio, "method": method,
            "n_total": len(all_seeds), "n_valid": len(valid), "n_failed": len(all_seeds) - len(valid),
            "total_corruption_events": len(valid_events),
            "detected_events": len(detected_rows),
            "undetected_events": len(valid_events) - len(detected_rows),
            "missing_detection_delays": len(valid_events) - len(detected_rows),
            "raw_total_corruption_events": len(raw_events),
            "raw_detected_event_lower_bound": sum(_bool(row["detected"]) for row in raw_events),
            "final_detection_rate": _mean(seed_rates),
            "mean_detection_delay": _mean([float(row["detection_delay"]) for row in detected_rows]),
            "mean_seed_detection_delay": _mean(seed_delay_means),
            "delay_conditioning": "detected_events_only; undetected delay remains missing",
            "seeds_with_any_detected_event": len(seed_delay_means),
            "total_selected_count": selected, "total_attempted_count": attempted,
            "total_completed_count": completed,
            "verification_budget_utilization": selected / budget if budget else math.nan,
            "mean_seed_budget_utilization": _mean(budget_utilization),
            "completion_budget_utilization": completed / budget if budget else math.nan,
            "valid_seeds": ";".join(map(str, valid)),
            "failed_seeds": ";".join(str(seed) for seed in all_seeds if seed not in valid),
        })
    return result


def _single_run_groups(round_rows: Sequence[Mapping], seed: int,
                       total_rounds: int) -> tuple[int, dict]:
    """Validate the requested single-seed context without pooling any seeds."""
    if (isinstance(seed, (bool, np.bool_)) or not _finite(seed)
            or int(seed) != float(seed)):
        raise ValueError("A single run requires one explicit integer seed")
    if (isinstance(total_rounds, (bool, np.bool_)) or not _finite(total_rounds)
            or int(total_rounds) != float(total_rounds) or int(total_rounds) < 1):
        raise ValueError("A single run requires a positive integer total_rounds")
    selected_seed = int(seed)
    groups = _trajectory_groups(round_rows)
    unexpected = {key[2] for key in groups if key[2] != selected_seed}
    if unexpected:
        raise ValueError(f"Single-run raw rows contain other seeds: {sorted(unexpected)}")
    return selected_seed, groups


def _single_run_valid(rows: Mapping[int, Mapping], total_rounds: int) -> bool:
    """Use one validity decision for the whole trace, including count fields."""
    if not _trajectory_valid(rows, total_rounds):
        return False
    for row in rows.values():
        completed = row.get("completed_count")
        if (not _finite(completed) or float(completed) < 0.0
                or int(float(completed)) != float(completed)):
            return False
    return True


def summarize_single_run_rounds(
    round_rows: Sequence[Mapping],
    *,
    seed: int,
    methods: Sequence[str],
    cr_values: Sequence[float],
    total_rounds: int = 200,
) -> list[dict]:
    """Copy every actual point of one complete trajectory, with no seed average.

    Input is the saved raw trace at rounds 1..T. The known initial state is
    explicitly zero at t=0 for a valid complete trace. Unexpected failure or
    missing rounds invalidates the whole method/ratio trace, whose display
    values remain NaN at every point; earlier partial observations stay in raw
    records. An absent method/ratio is reported as one failed trajectory.

    No mean, standard deviation, confidence interval, smoothing or interpolation
    is computed. The returned CDR is the saved cumulative_detection_rate value.
    """
    selected_seed, groups = _single_run_groups(round_rows, seed, total_rounds)
    horizon = int(total_rounds)
    method_order = list(map(str, methods))
    ratios = list(map(float, cr_values))
    if (not method_order or len(set(method_order)) != len(method_order)
            or any(not name.strip() for name in method_order)):
        raise ValueError("Single-run methods must be a nonempty unique sequence")
    if (not ratios or len(set(ratios)) != len(ratios)
            or any(not math.isfinite(ratio) or ratio <= 0.0 for ratio in ratios)):
        raise ValueError("Single-run corruption ratios must be finite, positive and unique")
    expected = {(ratio, method, selected_seed) for ratio in ratios for method in method_order}
    unexpected = set(groups).difference(expected)
    if unexpected:
        raise ValueError(f"Single-run raw rows contain undeclared method/ratio pairs: {sorted(unexpected)}")
    result = []
    for ratio in ratios:
        for method in method_order:
            records = groups.get((ratio, method, selected_seed), {})
            valid = _single_run_valid(records, horizon)
            for round_id in range(horizon + 1):
                if not valid:
                    cdr = completed = discovered = denominator = math.nan
                elif round_id == 0:
                    cdr, completed, discovered = 0.0, 0, 0
                    denominator = int(records[1]["total_corruption_events"])
                else:
                    source = records[round_id]
                    cdr = float(source["cumulative_detection_rate"])
                    completed = int(float(source["completed_count"]))
                    discovered = int(source["cumulative_detected_events"])
                    denominator = int(source["total_corruption_events"])
                result.append({
                    "seed": selected_seed, "cr_dynamic": ratio,
                    "method": method, "round": round_id,
                    "cdr": cdr, "completed_count": completed,
                    "cumulative_detected_events": discovered,
                    "total_corruption_events": denominator,
                    "n_total": 1, "n_valid": int(valid), "n_failed": int(not valid),
                    "aggregation": "single_run",
                    "aggregation_unit": "one_complete_seed_trajectory",
                })
    return result


def summarize_single_run_events(
    event_rows: Sequence[Mapping],
    round_rows: Sequence[Mapping],
    *,
    seed: int,
    total_rounds: int = 200,
) -> list[dict]:
    """Describe one run's actual final rate and its detected-event delays.

    final_detection_rate is copied directly from the saved final round. The
    only mean here is mean_detection_delay over detected events within this
    one trajectory; it is not an average over seeds or repeated experiments.
    Missed events retain missing delays and are counted explicitly. Failed
    traces retain NaN summary values and separate raw-event lower bounds.
    """
    selected_seed, trajectories = _single_run_groups(round_rows, seed, total_rounds)
    horizon = int(total_rounds)
    events: dict[tuple, list[Mapping]] = defaultdict(list)
    seen_events: set[tuple] = set()
    for row in event_rows:
        key = (float(row["cr_dynamic"]), str(row["method"]), int(row["seed"]))
        if key[2] != selected_seed:
            raise ValueError("Single-run event rows contain another seed")
        event_key = (*key, int(row["event_id"]))
        if event_key in seen_events:
            raise ValueError(f"Duplicate event record: {event_key}")
        seen_events.add(event_key)
        detected = _bool(row["detected"])
        delay, first = row.get("detection_delay"), row.get("first_detection_round")
        if detected:
            onset, recovery = int(row["onset_round"]), int(row["recovery_round"])
            if (not _finite(delay) or float(delay) < 0.0 or not _finite(first)
                    or not 1 <= float(first) <= horizon
                    or not onset <= float(first) < recovery
                    or float(first) - onset != float(delay)):
                raise ValueError("Detected events require an observed active-window detection and matching delay")
        elif _finite(delay) or _finite(first):
            raise ValueError("Undetected events must retain missing detection times and delays")
        events[key].append(row)
    result = []
    for key in sorted(set(trajectories) | set(events)):
        ratio, method, _ = key
        records, raw_events = trajectories.get(key, {}), events.get(key, [])
        valid = _single_run_valid(records, horizon)
        if valid:
            final = records[horizon]
            total = int(final["total_corruption_events"])
            detected_events = [row for row in raw_events if _bool(row["detected"])]
            if (len(raw_events) != total
                    or len(detected_events) != int(final["cumulative_detected_events"])):
                raise ValueError("The complete single trajectory and its full event table disagree")
            detected_count = len(detected_events)
            undetected = total - detected_count
            final_rate = float(final["cumulative_detection_rate"])
            delays = [float(row["detection_delay"]) for row in detected_events]
            mean_delay = _mean(delays)
            selected = sum(int(row["selected_count"]) for row in records.values())
            attempted = sum(int(row["attempted_count"]) for row in records.values())
            completed = sum(int(row["completed_count"]) for row in records.values())
            budget = sum(int(row["budget"]) for row in records.values())
            selected_utilization = selected / budget if budget else math.nan
            completion_utilization = completed / budget if budget else math.nan
            delay_sample_size = len(delays)
        else:
            total = detected_count = undetected = final_rate = mean_delay = math.nan
            selected = attempted = completed = math.nan
            selected_utilization = completion_utilization = math.nan
            delay_sample_size = math.nan
        result.append({
            "seed": selected_seed, "cr_dynamic": ratio, "method": method,
            "n_total": 1, "n_valid": int(valid), "n_failed": int(not valid),
            "aggregation": "single_run", "aggregation_unit": "one_complete_seed_trajectory",
            "total_corruption_events": total, "detected_events": detected_count,
            "undetected_events": undetected, "missing_detection_delays": undetected,
            "raw_total_corruption_events": len(raw_events),
            "raw_detected_event_lower_bound": sum(_bool(row["detected"]) for row in raw_events),
            "final_detection_rate": final_rate, "mean_detection_delay": mean_delay,
            "detection_delay_sample_size": delay_sample_size,
            "delay_conditioning": "detected_events_within_this_single_run_only; undetected delay remains missing",
            "total_selected_count": selected, "total_attempted_count": attempted,
            "total_completed_count": completed,
            "verification_budget_utilization": selected_utilization,
            "completion_budget_utilization": completion_utilization,
        })
    return result
