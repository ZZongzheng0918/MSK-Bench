"""Muscle activation energy metric helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def _values(data: Any) -> np.ndarray:
    return np.asarray(data, dtype=np.float64)


def activation_energy(activation: Any) -> float:
    """Mean squared muscle activation over all muscles and timesteps."""
    values = _values(activation)
    if values.size == 0:
        return 0.0
    return float(np.mean(np.square(values)))


def mean_squared_activation(activation: Any) -> float:
    return activation_energy(activation)


def activation_abs_sum(activation: Any) -> float:
    values = _values(activation)
    if values.size == 0:
        return 0.0
    return float(np.sum(np.abs(values)))


def timestep_activation_energy(activation: Any) -> np.ndarray:
    values = _values(activation)
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return np.mean(np.square(values), axis=1)


__all__ = [
    "activation_abs_sum",
    "activation_energy",
    "mean_squared_activation",
    "timestep_activation_energy",
]
