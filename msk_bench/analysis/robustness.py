"""Normalized trapezoidal robustness AUC; equal task/type weighting."""

from collections import defaultdict
import numpy as np

from msk_bench.benchmarking.perturbations import (
    ACTION_NOISE_SIGMAS, OBSERVATION_NOISE_SIGMAS, DYNAMICS_RANDOMIZATION_SIGMAS,
)

PAPER_GRIDS = dict(action=ACTION_NOISE_SIGMAS, obs=OBSERVATION_NOISE_SIGMAS,
                   dynamics=DYNAMICS_RANDOMIZATION_SIGMAS)


def normalized_robustness_auc(scales, success, *, success_unit="fraction"):
    """Return 0--100 AUC over the supplied range; units are never guessed."""
    x, y = np.asarray(scales, dtype=float), np.asarray(success, dtype=float)
    if success_unit not in ("fraction", "percent"):
        raise ValueError("success_unit must be fraction or percent")
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
        raise ValueError("Need at least two matching scale/success samples")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite robustness samples")
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x[0] < 0 or np.any(np.diff(x) <= 0):
        raise ValueError("Scales must be distinct and nonnegative")
    if success_unit == "percent":
        y = y / 100.0
    if np.any((y < 0) | (y > 1)):
        raise ValueError("Success outside declared unit range")
    # Explicit trapezoidal sum also supports NumPy 1.26.
    return float(100 * np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2) / (x[-1] - x[0]))


def aggregate_robustness(rows, *, success_unit="fraction", require_paper_grid=False,
                         expected_tasks=None):
    """One algorithm per call; repeated seed rows are averaged at each scale.

    All tasks must have all three types and identical grids for each type.
    Set require_paper_grid for the paper protocol; expected_tasks detects omitted tasks.
    """
    curves = defaultdict(lambda: defaultdict(list))
    algorithms = set()
    if success_unit not in ("fraction", "percent"):
        raise ValueError("success_unit must be fraction or percent")
    ceiling = 1 if success_unit == "fraction" else 100
    for row in rows:
        task, noise = str(row["env_id"]), str(row["noise_type"])
        if noise not in PAPER_GRIDS:
            raise ValueError(f"Unknown noise type: {noise}")
        algorithms.add(str(row.get("algorithm", "unspecified")))
        scale, success = float(row["noise_scale"]), float(row["success_rate"])
        if not np.isfinite(scale) or scale < 0 or not np.isfinite(success) or not 0 <= success <= ceiling:
            raise ValueError("Individual robustness row outside declared range")
        curves[task, noise][scale].append(success)
    if not curves or len(algorithms) != 1:
        raise ValueError("Provide nonempty rows for exactly one algorithm")
    tasks = sorted({task for task, _ in curves})
    if expected_tasks is not None and set(tasks) != set(expected_tasks):
        raise ValueError("Task set differs from the requested evaluation protocol")
    per_task = {task: {} for task in tasks}
    grids = {}
    for task in tasks:
        for noise in PAPER_GRIDS:
            curve = curves.get((task, noise))
            if not curve:
                raise ValueError(f"Missing {noise} curve for {task}")
            scales = sorted(curve)
            expected = PAPER_GRIDS[noise] if require_paper_grid else grids.setdefault(noise, scales)
            if len(scales) != len(expected) or not np.allclose(scales, expected, rtol=0, atol=1e-12):
                raise ValueError(f"Inconsistent {noise} scale grid for {task}")
            per_task[task][noise] = normalized_robustness_auc(
                scales, [np.mean(curve[s]) for s in scales], success_unit=success_unit)
    by_type = {noise: float(np.mean([per_task[t][noise] for t in tasks])) for noise in PAPER_GRIDS}
    return dict(task_count=len(tasks), per_task=per_task, by_type=by_type,
                average=float(np.mean(list(by_type.values()))), success_unit="percent",
                scale_grids={n: sorted(curves[tasks[0], n]) for n in PAPER_GRIDS})
