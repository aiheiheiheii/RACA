"""User-invoked runtime checks, supplied but not run during implementation.

Default: audit authenticated complete saved formal results without scheduling
new work.  --check-mechanisms: use only hand-built state/event fixtures, then
return without reading/writing formal results or drawing a figure.  This file
is never imported or called by the formal run or the plot-only entry.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _close(actual, expected, message: str) -> None:
    _require(math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10),
             f"{message}: actual={actual!r}, expected={expected!r}")


def _bool(value) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0", "", "none", "nan"}:
        return False
    raise AssertionError(f"Invalid saved boolean: {value!r}")


def _missing(value) -> bool:
    return value is None or str(value).strip().lower() in {"", "none", "nan"}


def _rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def _trajectory(row: dict) -> tuple:
    ratio = float(row["cr_dynamic"])
    _require(math.isfinite(ratio) and ratio.is_integer(), "Dynamic ratios must be integral percentages")
    return int(ratio), int(row["seed"]), row["method"]


def _native_evidence(row: dict, cfg) -> dict:
    """Audit protocol evidence without accepting a cosine/hash substitution."""
    extras = json.loads(row.get("extra_json") or "{}")
    protocol, evidence = extras.get("protocol"), extras.get("protocol_evidence")
    method, verdict = row["method"], row["outcome"].lower()
    expected = {"RACA-HIM": "cosine", "I-EDI": "iedi_native", "DVA-P+EDI-V": "edi_v"}[method]
    _require(protocol == expected and isinstance(evidence, dict), "Wrong native method verification protocol")
    if method == "RACA-HIM":
        return evidence
    _require(_missing(row.get("similarity")) and _missing(row.get("digest")),
             "A native proof baseline recorded a substituted cosine or exact-digest result")
    target, t = int(row["target_id"]), int(row["round"])
    if method == "I-EDI":
        eta = int(evidence["eta"])
        _require(cfg.IEDI_ETA_MIN <= eta <= cfg.IEDI_ETA_MAX
                 and evidence["mode"] == cfg.IEDI_CHALLENGE_MODE
                 and int(evidence["round_id"]) == t and int(evidence["replica_id"]) == target,
                 "I-EDI challenge is outside its declared domain or bound to a different task")
        if verdict in {"timeout", "invalid"}:
            _require(evidence.get("response_tag") is None and _missing(row.get("tag"))
                     and evidence.get("source_status") == "timeout"
                     and evidence.get("source_flagged_malicious") is True,
                     "I-EDI timeout must preserve missing response and the source malicious flag")
        else:
            response, expected_tag = int(evidence["response_tag"]), int(evidence["expected_tag"])
            _require(0 <= response < eta and 0 <= expected_tag < eta, "I-EDI tag is outside its modulus")
            _require(int(row["tag"]) == response, "I-EDI recorded tag differs from its actual response")
            matches = response == expected_tag
            _require(matches == (verdict == "normal")
                     and evidence["source_status"] == ("match" if matches else "mismatch")
                     and bool(evidence["source_flagged_malicious"]) == (not matches),
                     "I-EDI source status/flag disagrees with actual tag equality")
    else:
        _require(_missing(row.get("tag")), "EDI-V must record a Merkle proof rather than a scalar tag wrapper")
        request = evidence["request"]
        _require(int(request["round_id"]) == t and int(request["replica_id"]) == target
                 and int(request["server_id"]) == target, "EDI-V request is attached to another replica/round")
        _require(int(request["block_bytes"]) == cfg.DVAP_EDIV_BLOCK_BYTES
                 and int(request["total_blocks"]) == cfg.D * 4 // cfg.DVAP_EDIV_BLOCK_BYTES
                 and 1 <= int(request["leaf_count"]) <= int(request["total_blocks"])
                 and 0 <= int(request["shuffle_key"]) < int(request["total_blocks"])
                 and int(request["level"]) >= 1 and int(request["position"]) >= 1,
                 "EDI-V request has an invalid block/subtree/shuffle layout")
        code = bytes.fromhex(request["security_code_hex"])
        _require(len(code) == 16, "EDI-V security code must contain 128 bits")
        _require(int(evidence["sampled_bytes"]) == int(request["leaf_count"]) * cfg.DVAP_EDIV_BLOCK_BYTES
                 and int(evidence["vendor_tree_hash_operations"]) > 0,
                 "EDI-V evidence is missing actual tree work")
        if verdict in {"timeout", "invalid"}:
            _require(evidence.get("proof") is None and int(evidence["response_hash_operations_performed"]) == 0,
                     "EDI-V timeout fabricated a returned proof/hash work")
        else:
            proof = evidence["proof"]
            _require(isinstance(proof, dict) and int(proof["replica_id"]) == target
                     and int(proof["server_id"]) == target and len(bytes.fromhex(proof["tag_hex"])) == 32,
                     "EDI-V response must contain an actual SHA256 VMHT node proof")
            _require(bool(evidence["integrity_match"]) == (verdict == "normal")
                     and int(evidence["response_hash_operations_performed"]) == int(evidence["response_subtree_hash_operations"])
                     and int(evidence["response_hash_operations_performed"]) > 0,
                     "EDI-V verdict/hash-operation evidence disagrees with proof execution")
    return evidence


def _saved_sources(root: Path) -> tuple[dict, dict]:
    manifest = json.loads((root / "logs/run_manifest.json").read_text(encoding="utf-8-sig"))
    _require(manifest.get("status") == "complete", "Resume the formal run before auditing saved results")
    paths = {}
    for key in ("long_term_rounds", "corruption_events", "verification_diagnostics",
                "long_term_summary", "event_detection_summary"):
        identity = manifest.get("files", {}).get(key)
        _require(isinstance(identity, dict), f"Missing manifest identity: {key}")
        relative = Path(str(identity.get("path", "")))
        path = (root / relative).resolve()
        _require(not relative.is_absolute() and path.is_relative_to(root.resolve()),
                 f"Saved file path escapes result directory: {key}")
        _require(path.is_file(), f"Missing saved file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        _require(digest.hexdigest() == identity.get("sha256")
                 and path.stat().st_size == identity.get("size_bytes"),
                 f"Saved file differs from manifest: {key}")
        paths[key] = path
    return manifest, paths


def validate_saved_results(result_root: Path) -> dict:
    """Independently reconstruct event CDR and history/assignment invariants."""
    root = result_root.resolve()
    manifest, paths = _saved_sources(root)
    native_cfg = importlib.import_module("experiments.06_long_term.config_native_long_term")
    legacy_cfg = importlib.import_module("experiments.06_long_term.config_long_term")
    cfg = native_cfg if manifest.get("protocol_family") == native_cfg.PROTOCOL_FAMILY else legacy_cfg
    _require(manifest.get("algorithm_version") == cfg.ALGORITHM_VERSION
             and manifest.get("code_version") == cfg.CODE_VERSION,
             "Saved result protocol/code identity is unsupported")
    _require(list(manifest.get("methods", ())) == list(cfg.METHODS)
             and list(manifest.get("seeds", ())) == list(cfg.SEEDS), "Unexpected formal methods/seeds")
    single_run = cfg is native_cfg
    if single_run:
        _require(tuple(cfg.SEEDS) == (2026,) and cfg.REPETITIONS == 1
                 and manifest.get("aggregation") == "single_run"
                 and int(manifest.get("repetitions", -1)) == 1
                 and int(manifest.get("single_seed", -1)) == 2026,
                 "Native results must be one directly observed simulation at seed 2026")
    expected_trajectories = {(cr, seed, method) for cr in cfg.DYNAMIC_CR_VALUES
                             for seed in cfg.SEEDS for method in cfg.METHODS}
    rounds = defaultdict(dict)
    for row in _rows(paths["long_term_rounds"]):
        key, t = _trajectory(row), int(row["round"])
        _require(key in expected_trajectories and 1 <= t <= cfg.TOTAL_ROUNDS,
                 f"Unexpected method-round key: {key}/{t}")
        _require(t not in rounds[key], f"Duplicate method-round: {key}/{t}")
        rounds[key][t] = row
        total, count = int(row["total_corruption_events"]), int(row["cumulative_detected_events"])
        _require(total == key[0] * cfg.N // 100 and 0 <= count <= total,
                 f"Incorrect event denominator/count: {key}/{t}")
        _close(row["cumulative_detection_rate"], count / total, f"Incorrect CDR: {key}/{t}")
        selected, attempted, completed, budget = (int(row[field]) for field in
            ("selected_count", "attempted_count", "completed_count", "budget"))
        _require(0 <= completed <= attempted <= selected <= budget <= 98,
                 f"Invalid budget/attempt/completion counts: {key}/{t}")
    _require(set(rounds) == expected_trajectories, "Missing formal method/seed trajectory")
    if single_run:
        _require(len(rounds) == 15 and sum(len(trace) for trace in rounds.values()) == 3000
                 and len({key[:2] for key in rounds}) == 5,
                 "The single-run grid must contain five scenarios, fifteen trajectories and 3000 method-rounds")
    failed = {"failed", "error", "exception", "aborted", "incomplete", "not_run", "runtime_error"}
    valid = {}
    for key, trace in rounds.items():
        _require(set(trace) == set(range(1, cfg.TOTAL_ROUNDS + 1)), f"Missing actual round: {key}")
        previous = 0
        for t in range(1, cfg.TOTAL_ROUNDS + 1):
            row = trace[t]
            count = int(row["cumulative_detected_events"])
            _require(count >= previous, f"CDR decreased after recovery: {key}/{t}")
            _require(int(row["round_new_detected_events"]) == count - previous,
                     f"Incorrect first-discovery increment: {key}/{t}")
            previous = count
        valid[key] = not any(str(row["status"]).lower() in failed for row in trace.values())

    events = defaultdict(dict)
    exogenous = {}
    for row in _rows(paths["corruption_events"]):
        key, model = _trajectory(row), int(row["model_id"])
        _require(key in expected_trajectories and 0 <= model < cfg.N,
                 f"Unexpected event identity: {key}/{model}")
        _require(model not in events[key], f"Repeated persistent state counted as a new event: {key}/{model}")
        onset, recovery = int(row["onset_round"]), int(row["recovery_round"])
        _require(1 <= onset < recovery <= cfg.TOTAL_ROUNDS + 1, f"Invalid active window: {key}/{model}")
        detected = _bool(row["detected"])
        if detected:
            first = int(row["first_detection_round"])
            _require(onset <= first < recovery, f"Detection outside active window: {key}/{model}")
            _close(row["detection_delay"], first - onset, f"Incorrect delay: {key}/{model}")
        else:
            _require(_missing(row["first_detection_round"]) and _missing(row["detection_delay"]),
                     f"Undetected event must retain missing times: {key}/{model}")
        events[key][model] = row
    _require(set(events) == expected_trajectories, "Missing event table for a formal method trajectory")
    for key, table in events.items():
        _require(len(table) == key[0] * cfg.N // 100, f"Missing event denominator: {key}")
        signature = sorted((model, int(row["onset_round"]), int(row["recovery_round"]))
                           for model, row in table.items())
        scenario = key[:2]
        _require(scenario not in exogenous or exogenous[scenario] == signature,
                 f"Methods did not share onset/recovery: {scenario}")
        exogenous[scenario] = signature

    task_counts = defaultdict(lambda: [0, 0, 0])
    makespans = {}
    active_task_key = None
    active_loads = {}
    active_targets = set()
    first_detections = {}
    native_flag_counts = defaultdict(lambda: [0, 0])
    edi_round_codes = defaultdict(dict)
    history = {}
    latest_round = {}
    diagnostic_count = 0
    for row in _rows(paths["verification_diagnostics"]):
        diagnostic_count += 1
        key, t = _trajectory(row), int(row["round"])
        _require(key in expected_trajectories and t in rounds[key], "Unattached task diagnostic")
        model = int(row["target_id"])
        _require(0 <= model < cfg.N, "Invalid diagnostic target ID")
        state_key = (*key, model)
        previous_h, previous_last = history.get(state_key, (float(cfg.H0), -1))
        _require(t >= latest_round.get(state_key, 0), "Diagnostics must preserve each model's temporal order")
        age = 1.0 if previous_last < 0 else min((t - previous_last) / cfg.A_MAX, 1.0)
        _close(row["h_before"], previous_h, "History mixed across methods or reset between rounds")
        _close(row["age_before"], age, "Age inconsistent with last effective verdict")
        _close(row["risk_before"], cfg.ALPHA * age + (1 - cfg.ALPHA) * previous_h,
               "Risk does not match visible Age/h")
        verdict = row["outcome"].lower()
        if verdict == "normal":
            updated, last = (1 - cfg.RHO) * previous_h, t
        elif verdict == "corrupted":
            updated, last = previous_h + cfg.RHO * (1 - previous_h), t
        elif verdict in {"timeout", "invalid"}:
            updated, last = previous_h + cfg.RHO * cfg.ETA * (1 - previous_h), previous_last
        else:
            raise AssertionError(f"Unexpected diagnostic outcome: {verdict}")
        _close(row["h_after"], updated, "Incorrect visible feedback update")
        _require(int(row["last_valid_round_after"]) == last, "Invalid interaction reset Verification Age")
        history[state_key], latest_round[state_key] = (updated, last), t
        if row["phase"] != "target":
            _require(row["phase"] in {"gt_admission", "gt_interaction"}, "Unexpected diagnostic phase")
            continue
        task_key = (*key, t)
        if task_key != active_task_key:
            if active_task_key is not None:
                makespans[active_task_key] = max(active_loads.values(), default=0.0)
            _require(task_key not in makespans, "Target diagnostics for a method-round must be contiguous")
            active_task_key, active_loads, active_targets = task_key, {}, set()
        _require(model not in active_targets, "A selected target was repeated in a round")
        active_targets.add(model)
        verifier = int(row["verifier_id"])
        _require(0 <= verifier < cfg.N and verifier != model, "Invalid/self verifier")
        attempted, completed = _bool(row["attempted"]), _bool(row["completed"])
        _require(attempted and (completed == (verdict in {"normal", "corrupted"})),
                 "Missing/failed task was represented as completed")
        if cfg is native_cfg:
            evidence = _native_evidence(row, cfg)
            if row["method"] == "I-EDI":
                native_flag_counts[task_key][0] += bool(evidence["source_flagged_malicious"])
                native_flag_counts[task_key][1] += evidence["source_status"] == "timeout"
            elif row["method"] == "DVA-P+EDI-V":
                code = evidence["request"]["security_code_hex"]
                codes = edi_round_codes[key]
                _require((t in codes and codes[t] == code) or (t not in codes and code not in codes.values()),
                         "EDI-V reused a security code across different audit rounds")
                codes[t] = code
        task_counts[task_key][0] += 1
        task_counts[task_key][1] += attempted
        task_counts[task_key][2] += completed
        cost = float(row["total_ms"])
        _require(math.isfinite(cost) and cost > 0.0, "Failed attempts cannot be assigned zero simulated cost")
        _close(cost, sum(float(row[field]) for field in ("compute_ms", "transfer_ms", "network_ms")),
               "Resource component sum differs from total cost")
        prior_load = active_loads.get(verifier, 0.0)
        _close(row["start_ms"], prior_load, "Verifier tasks were not serialized")
        active_loads[verifier] = prior_load + cost
        _close(row["finish_ms"], prior_load + cost, "Incorrect task completion load")
        _require(prior_load + cost <= cfg.T_ROUND_MAX_MS + 1e-9, "Verifier load violated 800 ms deadline")
        event = events[key].get(model)
        if (completed and verdict == "corrupted" and event is not None
                and int(event["onset_round"]) <= t < int(event["recovery_round"])):
            first_detections.setdefault((*key, model), t)
    if active_task_key is not None:
        makespans[active_task_key] = max(active_loads.values(), default=0.0)
    detection_times = defaultdict(list)
    for ekey, first in first_detections.items():
        detection_times[ekey[:3]].append(first)
    for times in detection_times.values():
        times.sort()
    for key, trace in rounds.items():
        for t, row in trace.items():
            task_key = (*key, t)
            _require(task_counts[task_key] == [int(row[field]) for field in
                ("selected_count", "attempted_count", "completed_count")], "Task/raw count disagreement")
            makespan = makespans.get(task_key, 0.0)
            _close(row.get("verification_makespan_ms", makespan), makespan,
                   "Makespan must be maximum serialized verifier load")
            found = bisect_right(detection_times[key], t)
            _require(found == int(row["cumulative_detected_events"]),
                     f"Stored CDR does not follow actual first valid active-window verdicts: {key}/{t}")
            if cfg is native_cfg and key[2] == "I-EDI" and str(row["status"]).lower() not in failed:
                details = json.loads(row.get("diagnostics_json") or "{}")
                _require([int(details.get(field, -1)) for field in
                         ("source_malicious_flag_count", "source_timeout_flag_count")] == native_flag_counts[task_key],
                         "I-EDI source timeout/malicious flag diagnostics disagree with task evidence")
        for model, event in events[key].items():
            first = first_detections.get((*key, model))
            _require(_bool(event["detected"]) == (first is not None), "Saved event discovery disagrees with tasks")
            if first is not None:
                _require(int(event["first_detection_round"]) == first, "Event first detection is not the first verdict")

    summary_keys = set()
    for row in _rows(paths["long_term_summary"]):
        cr, t, method = int(float(row["cr_dynamic"])), int(row["round"]), row["method"]
        skey = cr, t, method
        _require(skey not in summary_keys, "Duplicate saved summary point")
        summary_keys.add(skey)
        contributors = [seed for seed in cfg.SEEDS if valid[(cr, seed, method)]]
        _require(int(row["n_total"]) == cfg.REPETITIONS and int(row["n_valid"]) == len(contributors)
                 and int(row["n_failed"]) == cfg.REPETITIONS - len(contributors), "Summary failed-seed accounting is incorrect")
        if single_run:
            _require(int(row["seed"]) == 2026 and row["aggregation"] == "single_run"
                     and row["aggregation_unit"] == "one_complete_seed_trajectory",
                     "Summary is not the declared single-seed observation")
            _require(not {"mean_cdr", "std_cdr", "ci95_lower", "ci95_upper",
                          "mean_completed_count", "mean_cumulative_detected_events"}.intersection(row),
                     "A single-run summary must not contain repeated-seed mean/std/CI fields")
            fields = ("cdr", "completed_count", "cumulative_detected_events", "total_corruption_events")
            if not contributors:
                _require(all(_missing(row[field]) for field in fields),
                         "A failed single trajectory must keep undefined summary values missing")
                continue
            observed = rounds[(cr, 2026, method)][t if t else 1]
            _close(row["cdr"], observed["cumulative_detection_rate"] if t else 0,
                   "Single-run CDR must be copied directly from its actual round")
            _close(row["completed_count"], observed["completed_count"] if t else 0,
                   "Single-run completion count differs from its actual round")
            _close(row["cumulative_detected_events"], observed["cumulative_detected_events"] if t else 0,
                   "Single-run discovered count differs from its actual round")
            _close(row["total_corruption_events"], observed["total_corruption_events"],
                   "Single-run event denominator differs from its actual scenario")
            continue
        if not contributors:
            _require(_missing(row["mean_cdr"]), "Undefined mean was replaced by a finite value")
            continue
        values = [0.0 if t == 0 else float(rounds[(cr, seed, method)][t]["cumulative_detection_rate"])
                  for seed in contributors]
        _close(row["mean_cdr"], math.fsum(values) / len(values), "Mean CDR used an incorrect repeated unit")
        if len(values) >= 2:
            _close(row["std_cdr"], statistics.stdev(values), "Summary std must use seed-level ddof=1")
    expected_summary = {(cr, t, method) for cr in cfg.DYNAMIC_CR_VALUES
                        for t in range(cfg.TOTAL_ROUNDS + 1) for method in cfg.METHODS}
    _require(summary_keys == expected_summary, "Summary must preserve all 201 actual/initial rounds")
    if single_run:
        event_summary_keys = set()
        for row in _rows(paths["event_detection_summary"]):
            key = _trajectory(row)
            _require(key in expected_trajectories and key not in event_summary_keys
                     and row["aggregation"] == "single_run"
                     and row["aggregation_unit"] == "one_complete_seed_trajectory",
                     "Unexpected or repeated single-run final-event summary")
            event_summary_keys.add(key)
            _require(int(row["n_total"]) == 1 and int(row["n_valid"]) == int(valid[key])
                     and int(row["n_failed"]) == int(not valid[key]),
                     "Single-run event-summary failure accounting is inconsistent")
            if valid[key]:
                final = rounds[key][cfg.TOTAL_ROUNDS]
                _close(row["final_detection_rate"], final["cumulative_detection_rate"],
                       "Final event CDR was averaged instead of copied from its single run")
                _close(row["total_corruption_events"], len(events[key]), "Final event denominator changed")
                _close(row["detected_events"], final["cumulative_detected_events"], "Final discovered event count changed")
                _close(row["undetected_events"], len(events[key]) - int(final["cumulative_detected_events"]),
                       "Undiscovered events disappeared from the denominator")
            else:
                _require(_missing(row["final_detection_rate"]), "Failed event-summary CDR must remain missing")
        _require(event_summary_keys == expected_trajectories, "Missing one of the fifteen final-event summaries")
    return {"method_rounds": sum(len(trace) for trace in rounds.values()),
            "trajectories": len(rounds), "valid_trajectories": sum(valid.values()),
            "diagnostic_records": diagnostic_count, "summary_points": len(summary_keys),
            "aggregation": "single_run" if single_run else "historical_seed_mean",
            "seed": 2026 if single_run else None}


def check_mechanisms() -> None:
    """Explicit future unit fixtures, with no formal data or result writes."""
    import numpy as np
    cfg = importlib.import_module("experiments.06_long_term.config_long_term")
    methods = importlib.import_module("experiments.06_long_term.long_term_methods")
    dynamic = importlib.import_module("experiments.06_long_term.dynamic_simulator")
    metrics = importlib.import_module("experiments.06_long_term.long_term_metrics")
    loader = importlib.import_module("dataset.load_dataset")
    cfg.validate_protocol()
    values = {name: value for name, value in vars(cfg).items() if name.isupper()}
    fixture_cfg = SimpleNamespace(**{**values, "N": 4, "D": 2, "K": 2, "M": 1})
    first = methods.LongTermMethod("Age-Only", fixture_cfg, 1, np.arange(4))
    second = methods.LongTermMethod("Random", fixture_cfg, 2, np.arange(4))
    age, h, risk = first.visible_history(1)
    _require(np.all(age == 1.0) and np.all(h == 0.10), "Initial Age/h changed")
    _require(np.allclose(risk, 0.55), "Initial risk changed")
    first._update(0, "normal", 1)
    first._update(1, "corrupted", 1)
    first._update(2, "invalid", 1)
    first.state.last_round = 1
    age, h, risk = first.visible_history(2)
    _require(np.allclose(h, [0.05, 0.55, 0.325, 0.10]), "Four history update branches differ")
    _require(np.allclose(age, [0.05, 0.05, 1.0, 1.0]), "Invalid/untouched targets reset Age")
    _require(np.all((risk >= 0) & (risk <= 1)), "Risk escaped [0,1]")
    _require(np.all(second.state.anomaly == .10) and np.all(second.state.last_valid_round == -1),
             "Method histories share mutable arrays")
    first._update(2, "timeout", 2)
    _require(first.state.last_valid_round[2] == -1, "Timeout reset valid-verdict time")

    normal = np.asarray([[1, 0], [0, 1], [1, 1], [1, -1]], dtype=np.float32)
    event_ids = np.asarray([3, 1], dtype=np.int64)
    event_x = np.asarray([[-1, -1], [1, 0]], dtype=np.float32)
    onset, recovery = np.asarray([2, 3]), np.asarray([4, 5])
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 3], mask[2:4, 1] = True, True
    case = {"normal_X": normal, "event_model_ids": event_ids,
            "corrupted_event_X": event_x, "onset_round": onset,
            "recovery_round": recovery, "corrupted_mask": mask,
            "fail_uniform": np.full((4, 4), .99)}
    dynamic.validate_dynamic_case(case, n=4, d=2, rounds=4, cr_percent=50)
    profiles = loader.resource_profiles(seed=89, n=4, rounds=4)
    scenario = dynamic.DynamicScenario(case, profiles)
    port = scenario.for_method("fixture")
    other = scenario.for_method("resource_comparison")
    enrollment = port.trusted_enrollment()
    _require(enrollment[3] == dynamic.descriptor_digest(normal[3]), "Enrollment changed byte identity")
    try:
        port.begin_round(2)
    except RuntimeError:
        pass
    except ValueError:
        pass
    else:
        raise AssertionError("The causal port allowed a future round")
    evaluator = metrics.OfflineEventEvaluator(case, method="fixture", seed=1, cr_dynamic=50)
    for t in range(1, 5):
        port.begin_round(t)
        other.begin_round(t)
        responses = port.observe_candidates([3, 1])
        expected = np.array(normal[[3, 1]], copy=True)
        for j, model in enumerate((3, 1)):
            if mask[t - 1, model]:
                expected[j] = event_x[j]
        _require(np.array_equal(responses.descriptors, expected), "Event position/model ID or recovery switch is wrong")
        costs = port.resource_matrix([3], [0])
        other_costs = other.resource_matrix([3], [0])
        _require(np.array_equal(costs["total_ms"], other_costs["total_ms"]), "Methods received different pair costs")
        outcomes = port.execute([(3, 0)], protocol="sha256", expected_tags=enrollment,
                                allowed_verifier_ids=[0], budget=1)
        result = evaluator.consume_round(t, outcomes)
        _require(result["cumulative_detected_events"] == (0 if t == 1 else 1),
                 "A persistent event was counted repeatedly or removed on recovery")
    events = evaluator.event_rows()
    _require(events[0]["first_detection_round"] == 2 and events[0]["detection_delay"] == 0,
             "Onset-inclusive first detection is incorrect")
    _require(not events[1]["detected"] and events[1]["detection_delay"] is None,
             "Undiscovered event acquired a fabricated delay")
    failed_case = {**case, "fail_uniform": np.array(case["fail_uniform"], copy=True)}
    failed_case["fail_uniform"][0, 3] = 0.0
    failed_port = dynamic.DynamicScenario(failed_case, profiles).for_method("failure")
    tags = failed_port.trusted_enrollment()
    failed_port.begin_round(1)
    failed_outcome = failed_port.execute([(3, 0)], protocol="sha256", expected_tags=tags,
                                        allowed_verifier_ids=[0], budget=1)[0]
    _require(failed_outcome.attempted and not failed_outcome.completed and failed_outcome.total_ms > 0,
             "Failed task was completed or charged zero resource cost")
    check_native_protocols()
    print("Hand-built history/event/resource/failure mechanism checks completed. "
          "No formal scenario was loaded, no result file was written and no figure was drawn.")


def check_native_protocols() -> None:
    """Future hand-built challenge/proof checks; no descriptor dataset is read."""
    import numpy as np
    cfg = importlib.import_module("experiments.06_long_term.config_native_long_term")
    iedi = importlib.import_module("experiments.06_long_term.native_iedi")
    ediv = importlib.import_module("experiments.06_long_term.native_edi_v")
    cfg.validate_protocol()
    payload = np.arange(30, dtype="<f4").tobytes()
    rng = np.random.default_rng(1901)
    challenge = iedi.make_challenge(1, 0, rng=rng, mode=cfg.IEDI_CHALLENGE_MODE,
                                    eta_min=cfg.IEDI_ETA_MIN, eta_max=cfg.IEDI_ETA_MAX)
    _require(cfg.IEDI_ETA_MIN <= challenge.eta <= cfg.IEDI_ETA_MAX, "I-EDI challenge domain changed")
    tag = iedi.compute_tag(payload, challenge.eta)
    verified = iedi.verify_response(challenge, payload, tag, arrival_ms=10, deadline_ms=800)
    _require(verified.status == "match" and verified.integrity_match and not verified.flagged_malicious,
             "I-EDI genuine original tag was rejected")
    malformed = (tag + 1) % challenge.eta
    rejected = iedi.verify_response(challenge, payload, malformed, arrival_ms=10, deadline_ms=800)
    _require(rejected.status == "mismatch" and rejected.flagged_malicious, "I-EDI malformed tag was accepted")
    timeout = iedi.verify_response(challenge, payload, None, arrival_ms=None, deadline_ms=800)
    _require(timeout.status == "timeout" and timeout.flagged_malicious and timeout.integrity_match is None,
             "I-EDI source timeout flag was lost or converted to content proof")
    vendor = ediv.NativeEDIV({0: payload}, seed=1902,
                            block_bytes=cfg.DVAP_EDIV_BLOCK_BYTES,
                            detection_probability=cfg.DVAP_EDIV_DETECTION_PROBABILITY,
                            assumed_corrupted_blocks=cfg.DVAP_EDIV_MIN_CORRUPTED_BLOCKS)
    request = vendor.prepare_round(1, [0])[0]
    proof = vendor.respond(0, payload, request)
    _require(vendor.verify(0, request, proof), "EDI-V rejected its actual original VMHT proof")
    request2 = vendor.prepare_round(2, [0])[0]
    _require(request.security_code != request2.security_code, "EDI-V reused its one-audit security code")
    _require(request2.level == 1 and request2.leaf_count == 30,
             "This conservative 30-block fixture requires a full-root VMHT audit")
    changed = bytes([payload[0] ^ 1]) + payload[1:]
    bad_proof = vendor.respond(0, changed, request2)
    _require(not vendor.verify(0, request2, bad_proof), "EDI-V accepted a changed full-root payload")
    request3 = vendor.prepare_round(3, [0])[0]
    try:
        accepted = vendor.verify(0, request3, proof)
    except ValueError:
        accepted = False
    _require(not accepted, "EDI-V accepted a stale proof from another audit round")
    print("Native I-EDI challenge/tag and EDI-V VMHT proof fixtures completed; no formal data/output was used.")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="User-invoked Group 06 runtime invariants; never automatically run")
    parser.add_argument("--check-mechanisms", action="store_true",
                        help="Use isolated hand-built mechanism fixtures and return without formal data/output")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--result-root", type=Path)
    source.add_argument("--legacy", action="store_true", help="Read-only audit of old adapted results")
    args = parser.parse_args(argv)
    if args.check_mechanisms:
        check_mechanisms()
    else:
        result_root = args.result_root or PROJECT_ROOT / "results" / "06_long_term"
        manifest_path = result_root / "logs" / "run_manifest.json"
        if manifest_path.is_file() and args.result_root is None:
            active = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if args.legacy and active.get("protocol_family"):
                parser.error("The active folder contains native-protocol data; omit --legacy.")
            if not args.legacy and active.get("aggregation") != "single_run":
                parser.error("The folder still contains historical results. Run the current single-run entry first, "
                             "or explicitly select --legacy to audit the historical run.")
        result = validate_saved_results(result_root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("Saved-result checks completed; figure appearance still requires the user's visual inspection.")


if __name__ == "__main__":
    main()
