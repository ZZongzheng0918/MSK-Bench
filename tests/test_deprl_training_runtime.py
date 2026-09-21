"""Real depRL/middleware training, without machine-local source checkouts."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "rl_paradigms/depRL/experiments/msk_bench_training_files/msk_bench_squat.yaml"


def run_isolated(args, cwd):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", "-I", *args], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("algorithm,parallel", [("deprl", 1), ("middleware", 1), ("deprl", 2)])
def test_short_train_updates_saves_and_reloads(tmp_path, algorithm, parallel):
    config = CONFIG
    if algorithm == "middleware":
        config = tmp_path / "middleware.yaml"
        run_isolated(["-c", "from pathlib import Path; "
                      "from rl_paradigms.deprl_middleware_22tasks.generate_configs import TASKS, config_text; "
                      f"Path({str(config)!r}).write_text(config_text(TASKS[0], None, None), encoding='utf-8')"], tmp_path)
    original = config.read_bytes()
    output = tmp_path / "runs"
    run_isolated([
        "-m", "deprl.main", str(config), "--steps", "8", "--epoch-steps", "8",
        "--save-steps", "8", "--parallel", str(parallel), "--sequential", "1",
        "--seed", "17", "--test-episodes", "0", "--hidden-size", "32",
        "--batch-size", "2", "--buffer-size", "32", "--steps-before-batches", "1",
        "--steps-between-batches", "1", "--batch-iterations", "1", "--cpu",
        "--output-dir", str(output),
    ], tmp_path)
    assert config.read_bytes() == original
    manifests = list(output.rglob("training_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["algorithm"] == algorithm
    assert manifest["timesteps"] == 8
    model = Path(manifest["model_path"])
    assert model.is_file() and model.stat().st_size > 0
    # Reload in a different process and compare against the seeded initial actor.
    run_isolated(["-c", f"""
import numpy as np
import torch
import yaml
import deprl
from deprl.utils import load
from deprl.custom_distributed import distribute
from pathlib import Path
run = Path({str(manifests[0].parent)!r})
config = yaml.safe_load((run / 'config.yaml').read_text())
tonic = config['tonic']
exec(tonic['header'])
env = distribute(tonic['environment'], tonic, config['env_args'], parallel=1, sequential=1)
try:
    env.initialize(17)
    obs, muscle = env.start()
    initial = eval(tonic['agent'])
    initial.set_params(**config['mpo_args'])
    initial.initialize(env.observation_space, env.action_space, seed=17)
    trained = load(str(run), env.environments[0])
    initial_weights = dict(initial.model.actor.named_parameters())
    assert any(not torch.equal(p, initial_weights[n]) for n, p in trained.model.actor.named_parameters()), 'No actor update'
    action = trained.test_step(obs, 8)
    assert np.isfinite(action).all()
    assert action.shape == (1, *env.action_space.shape)
    next_obs, _, info = env.step(action)
    assert np.isfinite(next_obs).all() and np.isfinite(info['rewards']).all()
finally:
    env.close()
    env.close()
"""], tmp_path)


def test_training_failure_exits_nonzero(tmp_path):
    result = subprocess.run(
        [sys.executable, "-B", "-I", "-m", "deprl.main", str(CONFIG),
         "--steps", "0", "--output-dir", str(tmp_path)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert not list(tmp_path.rglob("training_manifest.json"))
