"""Durable, atomic Group 02 output and a cooperative writer lock.

Windows readers, virus scanners, and indexing services can briefly prevent a
rename. Retry those sharing errors without exposing a partially written target.
Each write owns a unique temporary file in the target directory. A completed
temporary file is retained, with its path in the error, if replacement fails.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
import csv
import gzip
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, TypeVar


RETRY_DELAYS = (0.05, 0.10, 0.20, 0.40, 0.80)
_WINDOWS_SHARING_ERRORS = {5, 32, 33}
_T = TypeVar("_T")


def retry_io(operation: Callable[[], _T], *, description: str = "file I/O",
             delays: Sequence[float] | None = None) -> _T:
    """Retry only access/sharing errors, for at most 1.55 seconds by default.

Use this around opening a file, a rename, or an entire read. Do not use it around
simulation work or a write callback that consumes an iterator: retrying such a
callback could change the artifact being saved.
"""
    waits = tuple(RETRY_DELAYS if delays is None else delays)
    if any(delay < 0 for delay in waits):
        raise ValueError("I/O retry delays must be nonnegative")
    for attempt in range(len(waits) + 1):
        try:
            return operation()
        except OSError as exc:
            sharing_error = (isinstance(exc, PermissionError)
                             or getattr(exc, "winerror", None) in _WINDOWS_SHARING_ERRORS)
            if not sharing_error or attempt == len(waits):
                if sharing_error:
                    exc.add_note(f"{description}: access/sharing retry limit reached.")
                raise
            time.sleep(waits[attempt])
    raise AssertionError("Unreachable I/O retry state")


class AtomicWriteError(OSError):
    """A complete artifact exists at recovery_path but could not be committed."""

    def __init__(self, target_path: Path, recovery_path: Path, phase: str,
                 original_error: OSError):
        self.target_path = target_path
        self.recovery_path = recovery_path
        self.phase = phase
        self.original_error = original_error
        message = (
            f"Could not {phase} output {target_path}. The existing target was not "
            f"removed or overwritten in place. Completed recovery file retained at "
            f"{recovery_path}. Cause: {original_error}"
        )
        super().__init__(original_error.errno, message, str(target_path))
        if getattr(original_error, "winerror", None) is not None:
            self.winerror = original_error.winerror


def atomic_write(path: str | Path, writer: Callable[[Path], None]) -> None:
    """Write with ``writer(temporary_path)``, sync, then atomically replace.

The writer must fully overwrite its supplied file and close all handles before
returning. It is invoked once. The old target stays intact on writer, sync, or
replacement failure. This function never deletes the target as a workaround.
"""
    target = Path(path)
    retry_io(lambda: target.parent.mkdir(parents=True, exist_ok=True),
             description=f"Create output directory {target.parent}")
    descriptor, name = retry_io(
        lambda: tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent),
        description=f"Create temporary output for {target}",
    )
    temporary = Path(name)
    os.close(descriptor)  # No mkstemp handle may remain open during replacement.
    completed = False
    try:
        writer(temporary)
        completed = True
        try:
            # Windows _commit/fsync requires a writable descriptor; r+b does
            # not truncate the already completed plain/gzip artifact.
            with retry_io(lambda: temporary.open("r+b"),
                          description=f"Open completed output {temporary}") as stream:
                retry_io(lambda: os.fsync(stream.fileno()),
                         description=f"Sync completed output {temporary}")
        except OSError as exc:
            raise AtomicWriteError(target, temporary, "sync", exc) from exc
        try:
            retry_io(lambda: os.replace(temporary, target),
                     description=f"Replace output {target}")
        except OSError as exc:
            raise AtomicWriteError(target, temporary, "replace", exc) from exc
    finally:
        # Incomplete serialization is not a usable checkpoint. Keep a completed
        # file on commit failure; success has already moved it to the target.
        if not completed:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def atomic_write_json(path: str | Path, payload: Any, compressed: bool = False,
                      default: Callable[[Any], Any] | None = None) -> None:
    """Save plain or gzip JSON, closing the gzip trailer before syncing."""
    def write(temporary: Path) -> None:
        opener = gzip.open if compressed else Path.open
        stream = retry_io(lambda: opener(temporary, "wt" if compressed else "w", encoding="utf-8"),
                          description=f"Open JSON temporary output {temporary}")
        with stream:
            options = {"separators": (",", ":")} if compressed else {"indent": 2}
            json.dump(payload, stream, ensure_ascii=False, default=default, **options)
            stream.flush()
    atomic_write(path, write)


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    """Save complete text without exposing a partial target."""
    def write(temporary: Path) -> None:
        with retry_io(lambda: temporary.open("w", encoding=encoding),
                      description=f"Open text temporary output {temporary}") as stream:
            stream.write(text)
            stream.flush()
    atomic_write(path, write)


def atomic_write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]],
                     fieldnames: Sequence[str] | None = None) -> None:
    """Save rows in caller order; callers supply any value normalization."""
    if fieldnames is None:
        records: Iterable[Mapping[str, Any]] = list(rows)
        columns = sorted({key for row in records for key in row})
    else:
        records = rows
        columns = list(fieldnames)

    def write(temporary: Path) -> None:
        with retry_io(lambda: temporary.open("w", encoding="utf-8", newline=""),
                      description=f"Open CSV temporary output {temporary}") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
            stream.flush()
    atomic_write(path, write)


class OutputDirectoryBusyError(RuntimeError):
    """Another cooperative Group 02 writer already owns this output root."""


@contextmanager
def output_directory_lock(output_dir: str | Path):
    """Exclude simultaneous runner/plotter writes to the same result directory.

The persistent lock file must never be replaced or deleted: locks belong to an
open file, and removing that path could let another process lock a new file.
The operating system releases the lock when a process exits, including crashes.
"""
    root = Path(output_dir).resolve()
    retry_io(lambda: root.mkdir(parents=True, exist_ok=True),
             description=f"Create result directory {root}")
    lock_path = root / ".overhead.lock"
    stream = retry_io(lambda: lock_path.open("a+b"),
                      description=f"Open result writer lock {lock_path}")
    locked = False
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise OutputDirectoryBusyError(
                f"Cannot acquire Group 02 writer lock for {root}. Another run or "
                f"plotter may be writing here. Close that process or choose a "
                f"different --output-dir, then resume. Lock file: {lock_path}. "
                f"Cause: {exc}"
            ) from exc
        yield root
    finally:
        if locked:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()
