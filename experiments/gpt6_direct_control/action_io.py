"""Strict, durable JSON I/O for full muscle-action vectors."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import numpy as np


class ActionValidationError(ValueError):
    """Raised when a proposed action violates the environment contract."""


def _space_arrays(action_space: Any) -> tuple[int, np.ndarray, np.ndarray]:
    shape = tuple(int(value) for value in action_space.shape)
    if len(shape) != 1:
        raise ActionValidationError(f"Expected a one-dimensional action space, got {shape}.")
    expected = shape[0]
    low = np.broadcast_to(np.asarray(action_space.low, dtype=np.float64), shape).reshape(-1)
    high = np.broadcast_to(np.asarray(action_space.high, dtype=np.float64), shape).reshape(-1)
    return expected, low, high


def validate_action(action: Sequence[float] | np.ndarray, action_space: Any) -> np.ndarray:
    """Return a float32 vector after exact length, finiteness and bound checks."""

    if isinstance(action, (str, bytes, dict)):
        raise ActionValidationError("Action must be a JSON array of numeric values.")
    try:
        raw = list(action)
    except TypeError as exc:
        raise ActionValidationError("Action must be an iterable numeric vector.") from exc
    if any(isinstance(value, (bool, np.bool_)) for value in raw):
        raise ActionValidationError("Boolean values are not valid muscle actions.")
    expected, low, high = _space_arrays(action_space)
    if len(raw) != expected:
        raise ActionValidationError(f"Expected exactly {expected} values, got {len(raw)}.")
    try:
        value = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ActionValidationError("Every action entry must be numeric.") from exc
    finite = np.isfinite(value)
    if not bool(np.all(finite)):
        bad = int(np.flatnonzero(~finite)[0])
        raise ActionValidationError(f"Action[{bad}] is NaN or infinite.")
    outside = (value < low) | (value > high)
    if bool(np.any(outside)):
        bad = int(np.flatnonzero(outside)[0])
        raise ActionValidationError(
            f"Action[{bad}]={value[bad]:.9g} is outside [{low[bad]:.9g}, {high[bad]:.9g}]."
        )
    result = value.astype(np.float32)
    if not bool(np.all(np.isfinite(result))):
        raise ActionValidationError("Action cannot be represented as finite float32 values.")
    return result


def atomic_write_json(path: str | Path, payload: Any, *, indent: int | None = 2) -> Path:
    """Atomically replace a UTF-8 JSON file without allowing NaN/Infinity."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=indent)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def write_action_file(path: str | Path, action: Sequence[float] | np.ndarray) -> Path:
    """Write an already validated action as a bare JSON array."""

    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if not bool(np.all(np.isfinite(values))):
        raise ActionValidationError("Cannot write NaN or infinite action values.")
    return atomic_write_json(path, [float(value) for value in values], indent=None)


def load_action_file(path: str | Path, action_space: Any) -> np.ndarray:
    """Read and validate the Phase 2 bare-array action format."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActionValidationError(f"Could not read valid JSON action from {source}: {exc}") from exc
    if not isinstance(payload, list):
        raise ActionValidationError("Action file must contain only a JSON array, not an object.")
    return validate_action(payload, action_space)


def parse_utc_timestamp(value: str) -> float:
    """Parse the experiment's UTC ISO timestamp into POSIX seconds."""

    from datetime import datetime

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    timestamp = parsed.timestamp()
    if not math.isfinite(timestamp):
        raise ValueError(f"Invalid timestamp: {value!r}")
    return timestamp
