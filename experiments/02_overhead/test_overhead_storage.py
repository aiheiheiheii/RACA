"""Storage fault-injection tests; no paper dataset or experiment execution."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import errno
import gzip
import importlib
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
storage = importlib.import_module("experiments.02_overhead.overhead_storage")


def _sharing_error(code=5):
    error = PermissionError(errno.EACCES, "Injected Windows sharing denial")
    error.winerror = code
    return error


@pytest.fixture
def no_retry_sleep(monkeypatch):
    waits = []
    monkeypatch.setattr(storage.time, "sleep", waits.append)
    return waits


@pytest.mark.parametrize("compressed", [False, True])
def test_atomic_json_round_trip_and_custom_default(tmp_path, compressed):
    target = tmp_path / "nested" / ("checkpoint.json.gz" if compressed else "manifest.json")
    payload = {"text": "完整结果", "path": Path("example"), "records": [1, 2, 3]}
    storage.atomic_write_json(target, payload, compressed=compressed, default=str)
    opener = gzip.open if compressed else Path.open
    with opener(target, "rt" if compressed else "r", encoding="utf-8") as stream:
        assert json.load(stream) == {**payload, "path": "example"}
    assert list(target.parent.glob("*.tmp")) == []


@pytest.mark.parametrize("error_code", [5, 32, 33])
def test_transient_replace_denial_retries_and_preserves_old_target(
        tmp_path, monkeypatch, no_retry_sleep, error_code):
    target = tmp_path / "run_manifest.json"
    target.write_text('{"old": true}', encoding="utf-8")
    real_replace = storage.os.replace
    attempts = []

    def replace(source, destination):
        attempts.append(Path(source))
        assert json.loads(Path(source).read_text(encoding="utf-8")) == {"complete": True}
        assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
        if len(attempts) <= 2:
            raise _sharing_error(error_code)
        return real_replace(source, destination)

    monkeypatch.setattr(storage.os, "replace", replace)
    storage.atomic_write_json(target, {"complete": True})
    assert len(attempts) == 3
    assert len(set(attempts)) == 1
    assert no_retry_sleep == list(storage.RETRY_DELAYS[:2])
    assert json.loads(target.read_text(encoding="utf-8")) == {"complete": True}
    assert list(tmp_path.glob("*.tmp")) == []


def test_permanent_replace_denial_retains_complete_recovery_file(
        tmp_path, monkeypatch, no_retry_sleep):
    target = tmp_path / "run_manifest.json"
    target.write_text("old complete target", encoding="utf-8")
    attempts = []

    def replace(source, destination):
        attempts.append(Path(source))
        raise _sharing_error()

    monkeypatch.setattr(storage.os, "replace", replace)
    with pytest.raises(storage.AtomicWriteError) as caught:
        storage.atomic_write_json(target, {"complete": True})
    error = caught.value
    assert target.read_text(encoding="utf-8") == "old complete target"
    assert len(attempts) == len(storage.RETRY_DELAYS) + 1
    assert no_retry_sleep == list(storage.RETRY_DELAYS)
    assert sum(no_retry_sleep) <= 2.0
    assert error.target_path == target
    assert error.phase == "replace"
    assert error.recovery_path.parent == target.parent
    assert json.loads(error.recovery_path.read_text(encoding="utf-8")) == {"complete": True}
    assert str(error.recovery_path) in str(error)
    assert error.recovery_path.name != target.name + ".tmp"


def test_nonsharing_replace_error_is_not_retried(tmp_path, monkeypatch, no_retry_sleep):
    target = tmp_path / "manifest.json"
    attempts = []

    def replace(source, destination):
        attempts.append(Path(source))
        raise OSError(errno.ENOSPC, "Injected full device")

    monkeypatch.setattr(storage.os, "replace", replace)
    with pytest.raises(storage.AtomicWriteError) as caught:
        storage.atomic_write_json(target, {"complete": True})
    assert len(attempts) == 1
    assert no_retry_sleep == []
    assert caught.value.recovery_path.is_file()
    assert not target.exists()


def test_serialization_failure_keeps_target_and_removes_incomplete_temp(tmp_path):
    target = tmp_path / "manifest.json"
    target.write_text("old", encoding="utf-8")
    with pytest.raises(TypeError):
        storage.atomic_write_json(target, {"bad": object()})
    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob("*.tmp")) == []


def test_sync_failure_retains_complete_temp_and_preserves_target(
        tmp_path, monkeypatch, no_retry_sleep):
    target = tmp_path / "manifest.json"
    target.write_text("old", encoding="utf-8")

    def fail_sync(descriptor):
        raise _sharing_error()

    monkeypatch.setattr(storage.os, "fsync", fail_sync)
    with pytest.raises(storage.AtomicWriteError) as caught:
        storage.atomic_write_json(target, {"complete": True})
    assert caught.value.phase == "sync"
    assert json.loads(caught.value.recovery_path.read_text(encoding="utf-8")) == {"complete": True}
    assert target.read_text(encoding="utf-8") == "old"


def test_closed_writer_and_fsync_precede_replace(tmp_path, monkeypatch):
    target = tmp_path / "payload.json.gz"
    real_replace = storage.os.replace
    real_fsync = storage.os.fsync
    events = []

    def sync(descriptor):
        events.append("sync")
        return real_fsync(descriptor)

    def replace(source, destination):
        events.append("replace")
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            assert json.load(stream) == {"complete": True}
        return real_replace(source, destination)

    monkeypatch.setattr(storage.os, "fsync", sync)
    monkeypatch.setattr(storage.os, "replace", replace)
    storage.atomic_write_json(target, {"complete": True}, compressed=True)
    assert events == ["sync", "replace"]


def test_independent_atomic_writes_use_unique_same_directory_temps(tmp_path, monkeypatch):
    target = tmp_path / "manifest.json"
    real_replace = storage.os.replace
    barrier = threading.Barrier(2)
    temporary_paths = []
    mutex = threading.Lock()

    def replace(source, destination):
        with mutex:
            first_attempt = Path(source) not in temporary_paths
            if first_attempt:
                temporary_paths.append(Path(source))
        if first_attempt:
            barrier.wait(timeout=10)
        return real_replace(source, destination)

    monkeypatch.setattr(storage.os, "replace", replace)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(storage.atomic_write_json, target, {"writer": value})
                   for value in (1, 2)]
        for future in futures:
            future.result(timeout=15)
    assert len(set(temporary_paths)) == 2
    assert all(path.parent == target.parent for path in temporary_paths)
    assert json.loads(target.read_text(encoding="utf-8")) in ({"writer": 1}, {"writer": 2})
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_open_denial_retries_without_reexecuting_rows(
        tmp_path, monkeypatch, no_retry_sleep):
    target = tmp_path / "summary.csv"
    real_open = Path.open
    attempts = []
    consumed = []

    def rows():
        for value in (1, 2):
            consumed.append(value)
            yield {"seed": value}

    def path_open(path, mode="r", *args, **kwargs):
        if path.suffix == ".tmp" and mode == "w":
            attempts.append(path)
            if len(attempts) <= 2:
                raise _sharing_error()
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", path_open)
    storage.atomic_write_csv(target, rows(), ["seed"])
    assert consumed == [1, 2]
    assert len(attempts) == 3
    with target.open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == [{"seed": "1"}, {"seed": "2"}]


def test_text_and_inferred_csv_fields(tmp_path):
    text_target = tmp_path / "report.md"
    storage.atomic_write_text(text_target, "完整结果\n")
    assert text_target.read_text(encoding="utf-8") == "完整结果\n"
    csv_target = tmp_path / "summary.csv"
    storage.atomic_write_csv(csv_target, [{"z": 2, "a": 1}, {"a": 3}])
    with csv_target.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["a", "z"]
        assert list(reader) == [{"a": "1", "z": "2"}, {"a": "3", "z": ""}]


def test_read_retries_sharing_denial(no_retry_sleep):
    attempts = []

    def read():
        attempts.append(True)
        if len(attempts) <= 2:
            raise _sharing_error(32)
        return {"complete": True}

    assert storage.retry_io(read, description="Read checkpoint") == {"complete": True}
    assert len(attempts) == 3
    assert no_retry_sleep == list(storage.RETRY_DELAYS[:2])


def test_output_lock_rejects_nested_writer_then_releases_on_exception(tmp_path):
    with pytest.raises(ValueError, match="Injected interruption"):
        with storage.output_directory_lock(tmp_path):
            with pytest.raises(storage.OutputDirectoryBusyError) as caught:
                with storage.output_directory_lock(tmp_path):
                    pytest.fail("Second writer entered the locked directory")
            assert str(tmp_path) in str(caught.value)
            assert "--output-dir" in str(caught.value)
            raise ValueError("Injected interruption")
    with storage.output_directory_lock(tmp_path) as locked_root:
        assert locked_root == tmp_path.resolve()
    assert (tmp_path / ".overhead.lock").is_file()


def test_output_lock_excludes_another_process_and_releases(tmp_path):
    code = (
        "import importlib,sys; sys.path.insert(0,sys.argv[1]); "
        "storage=importlib.import_module('experiments.02_overhead.overhead_storage'); "
        "\ntry:\n with storage.output_directory_lock(sys.argv[2]): pass"
        "\nexcept storage.OutputDirectoryBusyError: sys.exit(7)"
    )
    args = [sys.executable, "-B", "-c", code, str(PROJECT_ROOT), str(tmp_path)]
    with storage.output_directory_lock(tmp_path):
        locked = subprocess.run(args, capture_output=True, text=True, timeout=15)
        assert locked.returncode == 7, locked.stderr
    released = subprocess.run(args, capture_output=True, text=True, timeout=15)
    assert released.returncode == 0, released.stderr
