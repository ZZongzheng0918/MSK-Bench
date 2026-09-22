"""Read back benchmark artifacts before reporting a successful run."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def _check_finite(value: Any, *, numeric_strings: bool = False) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _check_finite(item, numeric_strings=numeric_strings)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_finite(item, numeric_strings=numeric_strings)
    elif isinstance(value, (float, int)):
        if not math.isfinite(value):
            raise ValueError("Table contains a non-finite numeric value")
    elif numeric_strings and isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return
        _check_finite(number)


def validate_table(path: str | Path) -> int:
    """Require a nonempty JSON/CSV table with finite numbers; return row count."""
    path = Path(path)
    suffix = path.suffix.lower()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if suffix == ".json":
            rows = json.load(stream)
            if isinstance(rows, dict):
                rows = [rows] if rows else []
        elif suffix == ".csv":
            reader = csv.DictReader(stream)
            rows = list(reader)
            if any(None in row or any(value is None for value in row.values()) for row in rows):
                raise ValueError(f"Malformed CSV row in {path}")
        else:
            raise ValueError(f"Unsupported table format: {suffix}")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) or not row for row in rows):
        raise ValueError(f"Table has no rows or invalid rows: {path}")
    _check_finite(rows, numeric_strings=suffix == ".csv")
    return len(rows)


def validate_video(path: str | Path) -> tuple[int, int, int]:
    """Decode an actual frame, requiring a nonempty uint8 RGB image."""
    import imageio.v3 as iio
    import numpy as np

    frames = iio.imiter(Path(path))
    try:
        frame = next(frames)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
            raise ValueError("Expected a nonempty uint8 RGB frame")
        return tuple(frame.shape)
    except Exception as error:
        raise ValueError(f"Could not decode a valid video frame from {path}: {error}") from error
    finally:
        frames.close()
