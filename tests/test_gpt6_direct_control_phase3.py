from __future__ import annotations

import numpy as np

from experiments.gpt6_direct_control import analyze_failure, render_rollout


def test_phase3_analysis_api_exists():
    for name in (
        "SUMMARY_COLUMNS",
        "first_sustained_divergence",
        "normalized_action_change_series",
        "quaternion_pitch_roll",
        "select_keyframe_times",
    ):
        assert hasattr(analyze_failure, name), f"missing Phase 3 analysis API: {name}"
    assert hasattr(render_rollout, "trim_arrays_at_first_fall")


def test_protocol_render_prefix_stops_at_first_fallen_sample():
    arrays = {
        "time": np.asarray([0.0, 0.1, 0.2, 0.3]),
        "fallen": np.asarray([False, False, True, True]),
        "qpos": np.arange(12).reshape(4, 3),
        "constant": np.asarray([1.0, 2.0]),
    }
    trimmed = render_rollout.trim_arrays_at_first_fall(arrays)
    assert trimmed["time"].tolist() == [0.0, 0.1, 0.2]
    assert trimmed["qpos"].shape == (3, 3)
    assert trimmed["constant"].tolist() == [1.0, 2.0]


def test_action_change_is_normalized_and_only_nonzero_at_updates():
    actions = np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, -1.0], [1.0, -1.0]])
    changes = analyze_failure.normalized_action_change_series(actions)
    np.testing.assert_allclose(changes, [0.0, 0.0, 1.0, 0.0])


def test_quaternion_pitch_roll_uses_mujoco_wxyz_convention():
    identity = np.asarray([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    pitch, roll = analyze_failure.quaternion_pitch_roll(identity)
    np.testing.assert_allclose(pitch, [0.0], atol=1e-8)
    np.testing.assert_allclose(roll, [0.0], atol=1e-8)


def test_first_divergence_requires_sustained_excess_over_residual_distribution():
    times = np.arange(0.0, 0.5, 0.1)
    residual = np.full(5, 0.1)
    assert analyze_failure.first_sustained_divergence(
        times, [0.1, 0.1, 0.5, 0.6, 0.7], times, residual, min_steps=3
    ) == 0.2
    assert analyze_failure.first_sustained_divergence(
        times, [0.1, 0.5, 0.6, 0.1, 0.1], times, residual, min_steps=3
    ) is None


def test_keyframes_are_ordered_unique_and_never_exceed_four():
    selected = analyze_failure.select_keyframe_times(
        duration=0.8,
        first_divergence=0.2,
        decision_times=[0.0, 0.2, 0.4, 0.6],
        fallen=True,
    )
    assert selected == [("initial", 0.0), ("first_divergence", 0.2), ("attempted_correction", 0.4), ("fall", 0.8)]


def test_summary_columns_match_phase3_contract():
    assert analyze_failure.SUMMARY_COLUMNS == [
        "task",
        "controller",
        "seed",
        "success",
        "survived_full_3s_horizon",
        "sim_duration_seconds",
        "survival_steps",
        "tracking_rmse",
        "final_tracking_error",
        "max_tracking_error",
        "mean_action_delta",
        "max_action_delta",
        "mean_action_norm",
        "GPT_decisions",
        "mean_wall_clock_decision_time",
        "median_wall_clock_decision_time",
        "max_wall_clock_decision_time",
        "first_divergence_time",
        "termination_reason",
    ]
