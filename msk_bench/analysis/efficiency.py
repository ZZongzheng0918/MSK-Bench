"""Training-log efficiency helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

STEP_ALIASES = ("step", "steps", "environment_step", "env_step", "timestep", "timesteps")


def _as_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), -float("inf")):
        return None
    return number


def _step_from_row(row: Mapping[str, Any]) -> float | None:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for alias in STEP_ALIASES:
        if alias in lowered:
            return _as_number(lowered[alias])
    return None


def peak_efficiency_steps(rows: Iterable[Mapping[str, Any]]) -> int | float:
    """Return the largest logged environment step for peak-efficiency reporting."""

    max_step: float | None = None
    for row in rows:
        step = _step_from_row(row)
        if step is None:
            continue
        max_step = step if max_step is None else max(max_step, step)
    if max_step is None:
        return 0
    return int(max_step) if float(max_step).is_integer() else max_step


__all__ = ["STEP_ALIASES", "peak_efficiency_steps"]
