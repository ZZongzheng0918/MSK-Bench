from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_default_gl_backend_matches_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    from msk_bench.runtime import configure_mujoco_gl

    monkeypatch.delenv("MUJOCO_GL", raising=False)
    assert configure_mujoco_gl(platform_name="win32") == "glfw"
    assert os.environ["MUJOCO_GL"] == "glfw"


def test_explicit_gl_backend_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    from msk_bench.runtime import configure_mujoco_gl

    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    assert configure_mujoco_gl(platform_name="linux") == "osmesa"


def test_camera_none_normalizes_to_zero() -> None:
    from msk_bench.runtime import normalize_camera_id

    assert normalize_camera_id(None) == 0
    assert normalize_camera_id(3) == 3


def test_squat_rgb_array_and_repeated_close(tmp_path: Path) -> None:
    code = """
import gymnasium as gym
import numpy as np
import msk_bench

env = gym.make('MSKBenchSquat-v0', render_mode='rgb_array')
try:
    env.reset(seed=0)
    frame = env.render()
    assert frame.dtype == np.uint8
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.shape[0] > 0 and frame.shape[1] > 0
finally:
    env.close()
    env.close()
"""
    env = {key: value for key, value in os.environ.items() if key.upper() != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, "-B", "-I", "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "exception ignored" not in completed.stderr.lower()
