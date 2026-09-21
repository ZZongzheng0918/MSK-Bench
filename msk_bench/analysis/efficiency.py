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
    """Earliest evaluated step attaining the best mean return (one task only).

    Repeated rows at a step are averaged with equal weight (e.g. seed means).
    Invalid observations are omitted; missing return data is not efficiency.
    """
    returns: dict[float, list[float]] = {}
    for row in rows:
        step = _step_from_row(row)
        lowered = {str(key).strip().lower(): value for key, value in row.items()}
        reward = next((_as_number(lowered[key]) for key in
                       ("mean_return", "avg_return", "eval/mean_reward", "avg_reward")
                       if key in lowered), None)
        if step is None or step < 0 or reward is None:
            continue
        returns.setdefault(step, []).append(reward)
    if not returns:
        raise ValueError("Peak efficiency requires finite evaluation steps and mean returns.")
    best = max(sorted(returns), key=lambda step: sum(returns[step]) / len(returns[step]))
    return int(best) if best.is_integer() else best


__all__ = ["STEP_ALIASES", "peak_efficiency_steps"]
