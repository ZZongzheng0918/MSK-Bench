"""Opt-in test of the real upstream network checkpoint, never a zero-policy stand-in."""

import os

import pytest

from test_deprl_training_runtime import run_isolated


@pytest.mark.skipif(os.environ.get("MSK_BENCH_TEST_PRETRAINED") != "1",
                    reason="Set MSK_BENCH_TEST_PRETRAINED=1 to download/test upstream weights")
@pytest.mark.parametrize("env_id", ["MSKBenchResidualWalk-v0", "MSKBenchResidualRun-v0",
                                   "MSKBenchResidualStair-v0"])
def test_real_pretrained_environment_step(env_id, tmp_path):
    run_isolated(["-c", f"""
import gymnasium as gym
import numpy as np
import msk_bench
from musclemimic.integrations.msk_bench import resolve_checkpoint_source
from musclemimic.runner.checkpointing import _canonicalize_resume_path
checkpoint = _canonicalize_resume_path(resolve_checkpoint_source(None))
env = gym.make({env_id!r}, base_model_dir=str(checkpoint))
try:
    obs, info = env.reset(seed=17)
    assert env.observation_space.contains(obs), obs.shape
    assert env.get_wrapper_attr('_base_policy_fn') is not None
    obs, reward, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32))
    assert env.observation_space.contains(obs)
    assert np.isfinite(reward)
    print('PRETRAINED_STEP_OK', {env_id!r}, reward)
finally:
    env.close()
"""], tmp_path)
