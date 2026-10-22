"""Deterministic random streams, serialized results, and file integrity checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
import hashlib
import importlib.metadata
import math
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np


def seed_rng(seed: int, stream: int, N: int, cr: float) -> np.random.Generator:
    """Separate stream keys do not depend on labels or source-bank identities."""
    parts = (seed, stream, N)
    if any(int(value) != value or value < 0 for value in parts):
        raise ValueError("Seed, stream and N must be nonnegative integers")
    if not math.isfinite(cr) or cr < 0:
        raise ValueError("Corruption percentage must be nonnegative and finite")
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(stream), int(N), int(round(cr * 10))])
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_snapshot(root: str | Path) -> dict[str, str]:
    """Hash every existing file, including manifest and validation documents.

    Comparing complete dictionaries also detects newly added or deleted files.
    """
    base = Path(root).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {base}")
    files = sorted((path for path in base.rglob("*") if path.is_file()), key=lambda p: p.as_posix())
    return {path.relative_to(base).as_posix(): _sha256_file(path) for path in files}


def verify_manifest(root: str | Path) -> dict[str, Any]:
    """Verify listed hashes; ``verified`` is a count, failures are filename lists.

    This does not mutate the dataset or regenerate its manifest. Unlisted files
    remain covered by ``dataset_snapshot`` before/after checks.
    """
    base = Path(root).resolve()
    manifest = base / "MANIFEST.sha256"
    if not manifest.is_file():
        return {"verified": 0, "missing": ["MANIFEST.sha256"], "mismatch": []}
    verified = 0
    missing: list[str] = []
    mismatch: list[str] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64 or any(c not in "0123456789abcdefABCDEF" for c in fields[0]):
            raise ValueError(f"Malformed checksum at {manifest}:{line_number}")
        expected, relative_name = fields
        relative_name = relative_name.lstrip("*")
        path = (base / relative_name).resolve()
        if not path.is_relative_to(base):
            raise ValueError(f"Manifest path escapes the dataset: {relative_name}")
        if not path.is_file():
            missing.append(relative_name)
        elif _sha256_file(path) != expected.lower():
            mismatch.append(relative_name)
        else:
            verified += 1
    return {"verified": verified, "missing": missing, "mismatch": mismatch}


def sha256_array(array: np.ndarray) -> str:
    """Include dtype/shape so differently interpreted bytes cannot collide."""
    values = np.asarray(array)
    if values.dtype.hasobject:
        raise ValueError("Object arrays do not have a reproducible byte representation")
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(repr(values.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    """Write supplied row order deterministically; undefined numbers are 'nan'.

    The caller controls the scientific row order. Without explicit field names,
    columns use sorted keys across the supplied rows, avoiding hash-order drift.
    """
    records = list(rows)
    columns = list(fieldnames) if fieldnames is not None else sorted({key for row in records for key in row})
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in records:
            serialized = {}
            for key, value in row.items():
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, float) and math.isnan(value):
                    value = "nan"
                serialized[key] = value
            writer.writerow(serialized)


def environment_metadata() -> dict[str, Any]:
    """Record dependency versions without introducing a volatile timestamp."""
    packages = {}
    for name in ("numpy", "scipy", "matplotlib", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "rng": "numpy.random.PCG64; SeedSequence([seed, stream, N, round(cr*10)])",
        "undefined_metric_policy": "zero denominator => NaN, CSV literal nan",
    }

