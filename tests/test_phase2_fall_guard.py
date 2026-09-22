from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

from experiments.gpt6_direct_control import paused_env_controller as controller
from experiments.gpt6_direct_control.action_io import write_action_file
from experiments.gpt6_direct_control.observation_serializer import ObservationLayout


@dataclass
class FakeData:
    time: float = 0.0


class SteppingEnv:
    def __init__(self) -> None:
        self._data = FakeData()
        self.steps = 0

    def step(self, action):
        self.steps += 1
        self._data.time += 0.01
        observation = np.asarray([self._data.time], dtype=np.float32)
        return observation, 0.0, False, False, {}


class FakeSpace:
    shape = (2,)
    low = np.asarray([-1.0, -1.0], dtype=np.float32)
    high = np.asarray([1.0, 1.0], dtype=np.float32)


class ValidationEnv:
    action_space = FakeSpace()

    def close(self):
        return None


class RolloutEnv:
    action_space = FakeSpace()
    observation_space = SimpleNamespace(shape=(1,))

    def __init__(self) -> None:
        self._data = FakeData()
        self.steps = 0

    def reset(self, seed):
        self._data.time = 0.0
        self.steps = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        self._data.time += 0.01
        return np.asarray([self._data.time], dtype=np.float32), 0.0, False, False, {}

    def close(self):
        return None


class MemoryRecorder:
    def __init__(self, metadata):
        self.metadata = metadata

    def append(self, *args, **kwargs):
        return None

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"trajectory")
        return path


def test_step_until_deadline_stops_on_first_fall_signal():
    env = SteppingEnv()
    result = controller.step_until_deadline(
        env,
        np.zeros(1, dtype=np.float32),
        deadline=0.2,
        stop_if=lambda transition: "fall" if transition.native_step == 3 else None,
    )

    assert result.native_steps == 3
    assert result.simulation_time == 0.03
    assert result.stop_reason == "fall"


def test_run_rollout_propagates_fall_and_stops_current_hold(tmp_path, monkeypatch):
    layout = ObservationLayout(
        joint_position_dim=0,
        joint_velocity_dim=0,
        muscle_count=0,
        relative_site_count=1,
        touch_names=(),
        root_position_dim=0,
        root_velocity_dim=0,
        lookahead_count=0,
    )
    env = RolloutEnv()
    monkeypatch.setattr(controller, "make_direct_env", lambda task: env)
    monkeypatch.setattr(controller.ObservationLayout, "from_env", lambda env: layout)
    monkeypatch.setattr(controller, "environment_info", lambda *args, **kwargs: {})
    monkeypatch.setattr(controller, "TrajectoryRecorder", MemoryRecorder)
    monkeypatch.setattr(
        controller,
        "logging_diagnostics",
        lambda env, observation, layout: {"fallen": env.steps >= 3},
    )

    result = controller.run_rollout(
        "walk",
        [np.zeros(2, dtype=np.float32)],
        tmp_path,
        controller="gpt_direct",
    )

    assert result.fallen is True
    assert result.stop_reason == "fall"
    assert result.native_steps == 3
    assert result.final_simulation_time == 0.03


def test_submit_marks_session_complete_when_rollout_falls(tmp_path, monkeypatch):
    layout = ObservationLayout(
        joint_position_dim=2,
        joint_velocity_dim=2,
        muscle_count=2,
        relative_site_count=2,
    )
    exported = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat().replace(
        "+00:00", "Z"
    )
    manifest = {
        "format_version": 1,
        "phase": 2,
        "task": "walk",
        "controller": "gpt_direct",
        "seed": 0,
        "hold_seconds": 0.2,
        "max_duration": 3.0,
        "max_decisions": 15,
        "status": "awaiting_action",
        "decisions": [],
        "pending_observation": "old.json",
        "pending_observation_exported_at_utc": exported,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    action_path = tmp_path / "action.json"
    write_action_file(action_path, [0.25, -0.25])

    monkeypatch.setattr(controller, "make_direct_env", lambda task: ValidationEnv())
    monkeypatch.setattr(
        controller,
        "run_rollout",
        lambda *args, **kwargs: SimpleNamespace(
            final_observation=np.zeros(layout.total_dim, dtype=np.float32),
            final_simulation_time=0.07,
            terminated=False,
            truncated=False,
            fallen=True,
            stop_reason="fall",
            layout=layout,
        ),
    )

    result = controller.submit_action(tmp_path, action_path)

    assert result["status"] == "complete"
    assert result["stop_reason"] == "fall"
    assert result["pending_observation"] is None
    assert len(result["decisions"]) == 1
    assert result["decisions"][0]["stop_reason_after_action"] == "fall"
