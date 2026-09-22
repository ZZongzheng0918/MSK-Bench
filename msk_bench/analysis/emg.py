"""EMG-envelope analysis helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

TARGET_TASK = "MSKBenchWalk-v0"
NORMALIZED_GAIT_POINTS = 101


def resample_to_gait_cycle(signal: Any, points: int = NORMALIZED_GAIT_POINTS) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if points <= 0:
        raise ValueError("points must be positive")
    if values.size == 0:
        return np.zeros(points, dtype=np.float64)
    if values.size == 1:
        return np.full(points, float(values[0]), dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, values.size)
    x_new = np.linspace(0.0, 1.0, points)
    return np.interp(x_new, x_old, values)


def minmax_normalize(signal: Any) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    if values.size == 0:
        return values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float64)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if abs(high - low) < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - low) / (high - low)


def pearson_similarity(simulated: Any, reference: Any) -> float:
    left = np.asarray(simulated, dtype=np.float64).reshape(-1)
    right = np.asarray(reference, dtype=np.float64).reshape(-1)
    count = min(left.size, right.size)
    if count < 2:
        return float("nan")
    left = left[:count]
    right = right[:count]
    mask = np.isfinite(left) & np.isfinite(right)
    if int(np.sum(mask)) < 2:
        return float("nan")
    left = left[mask]
    right = right[mask]
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def best_shifted_pearson(simulated: Any, reference: Any) -> tuple[float, int]:
    sim = np.asarray(simulated, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    count = min(sim.size, ref.size)
    if count == 0:
        return float("nan"), 0
    sim = sim[:count]
    ref = ref[:count]
    best_score = float("nan")
    best_shift = 0
    for shift in range(count):
        score = pearson_similarity(np.roll(sim, shift), ref)
        if np.isnan(score):
            continue
        if np.isnan(best_score) or score > best_score:
            best_score = score
            best_shift = shift
    return best_score, best_shift


def _cycle_peaks(signal: np.ndarray, min_distance: int, prominence_ratio: float) -> list[int]:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size < 3:
        return []
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    low = float(np.min(finite))
    high = float(np.max(finite))
    prominence = max((high - low) * float(prominence_ratio), 0.0)
    candidates: list[int] = []
    for index in range(1, values.size - 1):
        value = values[index]
        if not np.isfinite(value):
            continue
        if value >= values[index - 1] and value > values[index + 1] and value - low >= prominence:
            candidates.append(index)

    peaks: list[int] = []
    min_distance = max(int(min_distance), 1)
    for index in candidates:
        if not peaks or index - peaks[-1] >= min_distance:
            peaks.append(index)
        elif values[index] > values[peaks[-1]]:
            peaks[-1] = index
    return peaks


def process_and_align_simulated_emg(
    raw_signal: Any,
    reference: Any | None = None,
    *,
    points: int = NORMALIZED_GAIT_POINTS,
    min_peak_distance: int = 60,
    prominence_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalize simulated EMG cycles and phase-align them to a reference envelope.

    This is the import-safe evaluator core from the standalone ``drawl_emg.py``
    workflow: detect gait-cycle peaks, resample each cycle to 101 points,
    min-max normalize the mean simulated envelope, then circularly shift it to
    maximize Pearson correlation with the human EMG reference.
    """

    raw = np.asarray(raw_signal, dtype=np.float64).reshape(-1)
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        aligned = np.zeros(points, dtype=np.float64)
        return aligned, aligned.copy(), float("nan")

    peaks = _cycle_peaks(raw, min_distance=min_peak_distance, prominence_ratio=prominence_ratio)
    cycles = [
        resample_to_gait_cycle(raw[peaks[index] : peaks[index + 1]], points)
        for index in range(len(peaks) - 1)
        if peaks[index + 1] - peaks[index] > 1
    ]
    if not cycles:
        cycles = [resample_to_gait_cycle(raw, points)]

    cycle_array = np.asarray(cycles, dtype=np.float64)
    mean_raw = np.mean(cycle_array, axis=0)
    normalized = minmax_normalize(mean_raw)
    scale = float(np.max(mean_raw) - np.min(mean_raw))
    spread = np.std(cycle_array, axis=0) / (scale if abs(scale) >= 1e-12 else 1.0)

    if reference is None:
        return normalized, spread, 0.0

    ref = minmax_normalize(resample_to_gait_cycle(reference, points))
    score, shift = best_shifted_pearson(normalized, ref)
    if not np.isfinite(score):
        score = 0.0
        shift = 0
    return np.roll(normalized, shift), np.roll(spread, shift), float(score)


def mean_emg_similarity(
    simulated_by_muscle: Mapping[str, Sequence[float]],
    reference_by_muscle: Mapping[str, Sequence[float]],
    *,
    align_phase: bool = True,
) -> float:
    scores: list[float] = []
    for muscle, reference in reference_by_muscle.items():
        if muscle not in simulated_by_muscle:
            continue
        if align_phase:
            score = process_and_align_simulated_emg(simulated_by_muscle[muscle], reference)[2]
        else:
            sim = minmax_normalize(resample_to_gait_cycle(simulated_by_muscle[muscle]))
            ref = minmax_normalize(resample_to_gait_cycle(reference))
            score = pearson_similarity(sim, ref)
        if np.isfinite(score):
            scores.append(float(score))
    return float(np.mean(scores)) if scores else float("nan")


__all__ = [
    "NORMALIZED_GAIT_POINTS",
    "TARGET_TASK",
    "best_shifted_pearson",
    "mean_emg_similarity",
    "minmax_normalize",
    "pearson_similarity",
    "process_and_align_simulated_emg",
    "resample_to_gait_cycle",
]
