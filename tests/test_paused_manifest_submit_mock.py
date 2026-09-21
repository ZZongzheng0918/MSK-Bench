from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

from experiments.gpt6_direct_control.action_io import write_action_file
from experiments.gpt6_direct_control.observation_serializer import ObservationLayout
from experiments.gpt6_direct_control import paused_env_controller as controller


class FakeSpace:
    shape = (2,)
    low = np.asarray([-1.0, -1.0], dtype=np.float32)
    high = np.asarray([1.0, 1.0], dtype=np.float32)


class ValidationEnv:
    action_space = FakeSpace()

    def close(self):
        return None


def test_submit_commits_one_mocked_action_and_exports_next_observation(tmp_path, monkeypatch):
    layout = ObservationLayout(
        joint_position_dim=2,
        joint_velocity_dim=2,
        muscle_count=2,
        relative_site_count=2,
    )
    exported = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
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
            final_simulation_time=0.2,
            terminated=False,
            truncated=False,
            layout=layout,
        ),
    )
    result = controller.submit_action(tmp_path, action_path)
    assert result["status"] == "awaiting_action"
    assert len(result["decisions"]) == 1
    assert result["decisions"][0]["action"] == [0.25, -0.25]
    assert result["decisions"][0]["wall_clock_latency_seconds"] >= 0.0
    next_path = tmp_path / "observations" / "decision_001_controller_visible.json"
    assert next_path.exists()
    assert json.loads(next_path.read_text(encoding="utf-8"))["visibility"] == "controller-visible"
