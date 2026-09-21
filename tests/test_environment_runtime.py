from pathlib import Path
import os
import subprocess
import sys

import mujoco
import pytest

from msk_bench.registry import CANONICAL_ENV_IDS

ROOT = Path(__file__).resolve().parents[1]
BODY = ROOT / "msk_bench" / "simhive" / "msk_sim" / "body"

EXTENSION_ENV_IDS = (
    "MSKBenchResidualRun-v0",
    "MSKBenchResidualStair-v0",
    "MSKBenchResidualWalk-v0",
    "MSKBenchAgenticWalk-v0",
)


@pytest.mark.parametrize("name", ["full_body.xml", "stairs.xml"])
def test_bundled_model_keyframes_match_compiled_qpos(name: str) -> None:
    model = mujoco.MjModel.from_xml_path(str(BODY / name))
    assert model.nq == 131
    assert model.nkey == 4
    assert model.key_qpos.shape == (4, model.nq)


@pytest.mark.parametrize("env_id", CANONICAL_ENV_IDS)
def test_canonical_environment_create_reset_step(env_id: str) -> None:
    code = f"""
import gymnasium as gym
import numpy as np
import msk_bench
env = gym.make({env_id!r})
try:
    observation, info = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    result = env.step(env.action_space.sample())
    assert len(result) == 5
    next_observation, reward, terminated, truncated, info = result
    assert env.observation_space.contains(next_observation)
    assert np.isfinite(float(reward))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
finally:
    env.close()
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-I", "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("env_id", EXTENSION_ENV_IDS)
def test_extension_environment_uses_installed_repository_only(
    env_id: str,
    tmp_path: Path,
) -> None:
    code = f"""
import gymnasium as gym
import msk_bench
env = gym.make({env_id!r})
try:
    obs, info = env.reset(seed=11)
    step = env.step(env.action_space.sample())
    assert len(step) == 5
finally:
    env.close()
"""
    env = {key: value for key, value in os.environ.items() if key.upper() != "PYTHONPATH"}
    agentic_state_dir = tmp_path / "agentic_state"
    env["MSK_BENCH_AGENTIC_STATE_DIR"] = str(agentic_state_dir)
    completed = subprocess.run(
        [sys.executable, "-B", "-I", "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "humanoid-bench" not in output.lower()
    if env_id == "MSKBenchAgenticWalk-v0":
        assert (agentic_state_dir / "shared_reward_weights.json").is_file()
