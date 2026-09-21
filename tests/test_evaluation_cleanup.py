"""Failures must not leak environments; requested frame sizes are respected."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark_eval import common
from test_deprl_training_runtime import run_isolated


@pytest.mark.parametrize('family', ['sb3', 'deprl'])
@pytest.mark.parametrize('metric', ['smooth', 'energy', 'render', 'emg'])
def test_common_entry_closes_on_failure(monkeypatch, family, metric):
    closed = []
    env = SimpleNamespace(close=lambda: closed.append(True))
    parser = argparse.ArgumentParser()
    parser.set_defaults(list_envs=False, benchmark_root='.', env='MSKBenchSquat-v0',
                        max_steps=2, episodes=1, deterministic=True, noisy=False)
    base = SimpleNamespace(ALGORITHM_NAME='ppo' if family == 'sb3' else 'deprl',
                           build_parser=lambda: parser, load_runtime=lambda *a: None,
                           selected_envs=lambda *a: ['MSKBenchSquat-v0'],
                           max_steps_for=lambda *a: 2,
                           load_model_and_env=lambda *a: (None, env, Path('.')),
                           build_agent_env=lambda *a: (None, env, Path('.')))
    def fail(*args, **kwargs):
        raise RuntimeError('intentional rollout failure')
    if metric == 'emg':
        from benchmark_eval import emg_export
        owner, target = emg_export, f'collect_{family}_target_emg_episode'
        entry = getattr(owner, f'main_{family}_emg_export')
    else:
        owner = common
        target = f'render_{family}_episode' if metric == 'render' else f'collect_{family}_episode'
        entry = getattr(owner, f'main_{family}_{metric}')
    monkeypatch.setattr(owner, target, fail)
    with pytest.raises(RuntimeError, match='intentional rollout failure'):
        entry(base, [])
    assert closed == [True]


def test_rgb_environment_supports_requested_sizes(tmp_path):
    run_isolated(['-c', """
import gymnasium as gym
import msk_bench
from benchmark_eval.common import render_rgb_frame
env = gym.make('MSKBenchSquat-v0', render_mode='rgb_array')
try:
    env.reset(seed=0)
    env.render()
    assert render_rgb_frame(env, 320, 240).shape == (240, 320, 3)
    assert render_rgb_frame(env, 160, 128).shape == (128, 160, 3)
finally:
    env.close()
"""], tmp_path)


BASE_MODULES = [
    f"rl_paradigms.{folder}.eval_{algorithm}_{mode}"
    for folder, algorithm in [('ppo', 'ppo'), ('sac', 'sac'), ('depRL', 'deprl'),
                              ('msgym', 'msgym'), ('deprl_middleware_22tasks', 'middleware')]
    for mode in ('success', 'robustness')
]


@pytest.mark.parametrize('module_name', BASE_MODULES)
@pytest.mark.parametrize('mode', ['success', 'robustness'])
def test_base_evaluator_closes_on_failure(monkeypatch, module_name, mode):
    import importlib
    module = importlib.import_module(module_name)
    if not hasattr(module, 'load_runtime'):
        module = module.base
    closed = []
    env = SimpleNamespace(close=lambda: closed.append(True))
    args = SimpleNamespace(benchmark_root='.', env='MSKBenchSquat-v0', episodes=1,
                           max_steps=2, deterministic=True, noisy=False, action_scales='0',
                           obs_scales='0', dynamics_scales='0', noise_type='action')
    monkeypatch.setattr(module, 'load_runtime', lambda *a: None)
    loader = 'load_model_and_env' if hasattr(module, 'load_model_and_env') else 'build_agent_env'
    monkeypatch.setattr(module, loader, lambda *a: (None, env, Path('.')))
    def fail(*a, **kw):
        raise RuntimeError('intentional rollout failure')
    monkeypatch.setattr(module, 'run_episode', fail)
    with pytest.raises(RuntimeError, match='intentional rollout failure'):
        getattr(module, 'evaluate_' + mode)(args)
    assert closed == [True]
