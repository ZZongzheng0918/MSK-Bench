"""Joint smoothness metric helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def finite_difference_jerk(joint_velocity: Any, dt: float = 1.0) -> float:
    """Mean squared finite-difference jerk from joint angular velocity samples."""
    values = np.asarray(joint_velocity, dtype=np.float64)
    if values.size == 0:
        return 0.0
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[0] < 3:
        return 0.0
    safe_dt = max(float(dt), 1e-12)
    jerk_like = np.diff(values, n=2, axis=0) / (safe_dt**2)
    return float(np.mean(np.square(jerk_like)))


def log_mean_squared_jerk(joint_velocity: Any, dt: float = 1.0, eps: float = 1e-12) -> float:
    value = finite_difference_jerk(joint_velocity, dt=dt)
    return float(np.log10(max(value, eps)))


__all__ = ["finite_difference_jerk", "log_mean_squared_jerk"]
