"""Muscle recruitment grouping utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .anatomical_groups import MUSCLE_FAMILY_KEYWORDS


def classify_actuator(name: str) -> str:
    normalized = str(name).lower()
    for family, keywords in MUSCLE_FAMILY_KEYWORDS.items():
        if any(str(keyword).lower() in normalized for keyword in keywords):
            return family
    return "unclassified"


def family_activation_mass(actuator_names: Sequence[str], activation: Any, *, absolute: bool = True) -> dict[str, float]:
    values = np.asarray(activation, dtype=np.float64).reshape(-1)
    masses = {family: 0.0 for family in MUSCLE_FAMILY_KEYWORDS}
    masses["unclassified"] = 0.0
    count = min(len(actuator_names), values.size)
    for index in range(count):
        value = float(values[index])
        family = classify_actuator(actuator_names[index])
        masses[family] = masses.get(family, 0.0) + (abs(value) if absolute else value)
    return masses


def family_activation_mean(actuator_names: Sequence[str], activation: Any, *, absolute: bool = True) -> dict[str, float]:
    values = np.asarray(activation, dtype=np.float64)
    if values.size == 0:
        return {"unclassified": 0.0, **{family: 0.0 for family in MUSCLE_FAMILY_KEYWORDS}}
    if values.ndim > 1:
        values = np.mean(np.abs(values) if absolute else values, axis=0)
        return family_activation_mass(actuator_names, values, absolute=False)
    return family_activation_mass(actuator_names, values, absolute=absolute)


__all__ = ["classify_actuator", "family_activation_mass", "family_activation_mean"]
