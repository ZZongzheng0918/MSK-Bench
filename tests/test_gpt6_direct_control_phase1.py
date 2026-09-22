from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from experiments.gpt6_direct_control.action_io import (
    ActionValidationError,
    load_action_file,
    validate_action,
    write_action_file,
)
from experiments.gpt6_direct_control.observation_serializer import (
    ObservationLayout,
    flatten_serialized_observation,
    serialize_policy_observation,
)
from experiments.gpt6_direct_control.paused_env_controller import (
    PUBLIC_ENV_IDS,
    decision_deadline,
    step_until_deadline,
)


class FakeSpace:
    def __init__(self) -> None:
        self.shape = (3,)
        self.low = np.asarray([-1.0, -0.5, 0.0], dtype=np.float32)
        self.high = np.asarray([1.0, 0.5, 1.0], dtype=np.float32)


@dataclass
class FakeData:
    time: float = 0.0


class FakeEnv:
    def __init__(self, dt: float = 0.03) -> None:
        self._data = FakeData()
        self.dt = dt
        self.steps = 0

    def step(self, action):
        self.steps += 1
        self._data.time += self.dt
        obs = np.asarray([self._data.time], dtype=np.float32)
        return obs, 0.0, False, False, {"source": "fake"}


def test_validate_action_requires_exact_finite_in_bounds_vector(tmp_path):
    space = FakeSpace()
    valid = validate_action([1, 0, 0.25], space)
    np.testing.assert_allclose(valid, [1.0, 0.0, 0.25])
    assert valid.dtype == np.float32

    for invalid in ([0.0, 0.0], [0.0, float("nan"), 0.0], [0.0, 0.6, 0.0]):
        with pytest.raises(ActionValidationError):
            validate_action(invalid, space)

    path = tmp_path / "action.json"
    write_action_file(path, valid)
    assert json.loads(path.read_text(encoding="utf-8")) == [1.0, 0.0, 0.25]
    np.testing.assert_array_equal(load_action_file(path, space), valid)


def test_observation_serializer_is_lossless_and_marks_visibility():
    layout = ObservationLayout(
        joint_position_dim=2,
        joint_velocity_dim=2,
        muscle_count=2,
        relative_site_count=2,
        touch_names=("rf", "rt", "lf", "lt"),
    )
    observation = np.arange(layout.total_dim, dtype=np.float32) / 10.0
    payload = serialize_policy_observation(
        observation,
        layout,
        task="walk",
        decision_index=3,
        simulation_time=0.6,
    )
    assert payload["visibility"] == "controller-visible"
    assert payload["task"] == "walk"
    assert payload["decision_index"] == 3
    assert len(payload["muscles"]["values"]) == 2
    assert set(payload["touch"]) == {"rf", "rt", "lf", "lt"}
    np.testing.assert_array_equal(flatten_serialized_observation(payload, layout), observation)


def test_direct_tasks_publish_exact_gym_environment_ids():
    assert PUBLIC_ENV_IDS == {
        "walk": "MSKBenchResidualWalk-v0",
        "run": "MSKBenchResidualRun-v0",
        "stairs": "MSKBenchResidualStair-v0",
    }


def test_decision_clock_uses_absolute_simulation_deadlines_without_drift():
    assert decision_deadline(0, hold_seconds=0.2) == pytest.approx(0.2)
    assert decision_deadline(1, hold_seconds=0.2) == pytest.approx(0.4)

    env = FakeEnv(dt=0.03)
    seen = []
    result = step_until_deadline(
        env,
        np.zeros(1, dtype=np.float32),
        deadline=0.2,
        on_step=lambda transition: seen.append(transition),
    )
    assert result.native_steps == 7
    assert result.simulation_time == pytest.approx(0.21)
    assert len(seen) == 7

    result = step_until_deadline(
        env,
        np.zeros(1, dtype=np.float32),
        deadline=0.4,
    )
    assert result.native_steps == 7
    assert result.simulation_time == pytest.approx(0.42)


def test_step_until_deadline_stops_on_environment_end():
    class EndingEnv(FakeEnv):
        def step(self, action):
            obs, reward, _, _, info = super().step(action)
            return obs, reward, self.steps == 2, False, info

    result = step_until_deadline(
        EndingEnv(dt=0.03),
        np.zeros(1, dtype=np.float32),
        deadline=0.2,
    )
    assert result.native_steps == 2
    assert result.terminated is True
