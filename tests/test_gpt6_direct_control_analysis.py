from __future__ import annotations

import numpy as np

from experiments.gpt6_direct_control.analyze_failure import classify_failure, summarize_arrays
from experiments.gpt6_direct_control.render_rollout import select_frame_indices


def test_select_frame_indices_uses_simulation_time_and_keeps_endpoints():
    times = np.arange(0.0, 0.21, 0.01)
    indices = select_frame_indices(times, fps=10)
    assert indices.tolist() == [0, 10, 20]


def test_failure_classification_priority():
    assert classify_failure(fallen=True, progress_ratio=1.0, tracking_rmse_mean=0.0, max_tilt=0.0) == "fall"
    assert classify_failure(fallen=False, progress_ratio=0.1, tracking_rmse_mean=0.0, max_tilt=0.0) == "insufficient_progress"
    assert classify_failure(fallen=False, progress_ratio=1.0, tracking_rmse_mean=0.8, max_tilt=0.0) == "tracking_loss"
    assert classify_failure(fallen=False, progress_ratio=1.0, tracking_rmse_mean=0.1, max_tilt=55.0) == "balance_instability"
    assert classify_failure(fallen=False, progress_ratio=1.0, tracking_rmse_mean=0.1, max_tilt=5.0) == "none_observed"


def test_summary_uses_reference_direction_and_reports_single_seed_limit():
    arrays = {
        "time": np.asarray([0.0, 0.1, 0.2]),
        "root_xyz": np.asarray([[0, 0, 1], [0.1, 0, 1], [0.2, 0, 1]], dtype=float),
        "reference_root_xyz": np.asarray([[0, 0, 1], [0.2, 0, 1], [0.4, 0, 1]], dtype=float),
        "root_height": np.asarray([1.0, 1.0, 1.0]),
        "root_tilt_degrees": np.asarray([0.0, 1.0, 2.0]),
        "tracking_site_rmse": np.asarray([0.0, 0.1, 0.2]),
        "fallen": np.asarray([False, False, False]),
        "commanded_action": np.zeros((3, 2)),
    }
    row = summarize_arrays(arrays, {"task": "walk", "controller": "mock", "seed": 0, "requested_duration": 0.2})
    assert row["progress_ratio"] == 0.5
    assert row["single_seed_qualitative_only"] is True
    assert row["success"] is True
