import numpy as np
import pytest


def test_joint_jerk_uses_second_difference_of_velocity():
    from benchmark_eval.common import _smooth_metrics
    # v=t^2: second derivative=2, mean squared jerk=4; third derivative=0.
    t = np.arange(6) * 0.1
    metrics = _smooth_metrics({"qvels": (t ** 2)[:, None], "actions": np.zeros((6, 1)), "dt": 0.1})
    assert metrics["joint_jerk_energy"] == pytest.approx(4.0)
    assert metrics["joint_log10_mean_squared_jerk"] == pytest.approx(np.log10(4.0))


def test_robustness_auc_normalizes_scale_range():
    from msk_bench.analysis.robustness import normalized_robustness_auc
    assert normalized_robustness_auc([0, .1, .2], [1, .5, 0]) == pytest.approx(50)
    with pytest.raises(ValueError):
        normalized_robustness_auc([0, .1, .1], [1, .5, 0])


def test_angular_joint_velocity_excludes_free_and_slide_dofs():
    from types import SimpleNamespace
    from benchmark_eval.common import angular_joint_velocity
    # MuJoCo types: free=0 (6 dofs), ball=1 (3), slide=2 (1), hinge=3 (1).
    model = SimpleNamespace(jnt_type=np.array([0, 1, 2, 3]), jnt_dofadr=np.array([0, 6, 9, 10]))
    env = SimpleNamespace(sim=SimpleNamespace(model=model))
    assert angular_joint_velocity(env, np.arange(11)).tolist() == [6, 7, 8, 10]


def test_short_velocity_trace_is_not_reported_as_zero_jerk():
    from benchmark_eval.common import _smooth_metrics
    with pytest.raises(ValueError, match="three velocity"):
        _smooth_metrics(dict(actions=np.zeros((2, 1)), qvels=np.zeros((2, 1)), dt=.1))


def test_dynamics_noise_scales_force_not_muscle_length_range():
    from types import SimpleNamespace
    from msk_bench.benchmarking.perturbations import randomize_muscle_strength
    gains = np.array([[.75, 1.05, 100, 200], [.75, 1.05, -1, 200]], dtype=float)
    model = SimpleNamespace(nu=2, actuator_gaintype=np.array([2, 2]),
                            actuator_gainprm=gains.copy(), actuator_biasprm=gains.copy())
    factors = randomize_muscle_strength(model, .2, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(model.actuator_gainprm[:, :2], gains[:, :2])
    assert model.actuator_gainprm[0, 2] == pytest.approx(100 * factors[0])
    assert model.actuator_gainprm[1, 2] == -1
    assert model.actuator_gainprm[1, 3] == pytest.approx(200 * factors[1])
    np.testing.assert_array_equal(model.actuator_gainprm, model.actuator_biasprm)


def test_robustness_equal_task_and_type_weighting():
    from msk_bench.analysis.robustness import aggregate_robustness
    rows = []
    for task, success in (("A", 1.0), ("B", 0.0)):
        for noise in ("action", "obs", "dynamics"):
            for scale in (0, .1):
                rows.append(dict(env_id=task, noise_type=noise, noise_scale=scale, success_rate=success))
    result = aggregate_robustness(rows)
    assert result["average"] == pytest.approx(50)
    assert result["task_count"] == 2
    with pytest.raises(ValueError):
        aggregate_robustness(rows[:-2])


def test_robustness_rejects_invalid_seed_rows_before_averaging():
    from msk_bench.analysis.robustness import aggregate_robustness
    rows = [dict(env_id="A", noise_type=noise, noise_scale=scale, success_rate=value)
            for noise in ("action", "obs", "dynamics") for scale in (0, .1) for value in (-1, 2)]
    with pytest.raises(ValueError, match="range"):
        aggregate_robustness(rows)
