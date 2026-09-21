"""DynSyn short training with actual updates and independent checkpoint reload."""

import json
from pathlib import Path

from test_deprl_training_runtime import ROOT, run_isolated

SCRIPT = ROOT / "rl_paradigms/msgym/SB3-Scripts/train.py"
CONFIG = ROOT / "rl_paradigms/msgym/configs/msk_bench_squat.json"


def test_msgym_short_training_and_reload(tmp_path):
    original = CONFIG.read_bytes()
    output = tmp_path / "runs"
    run_isolated([
        str(SCRIPT), "-f", str(CONFIG), "--total-timesteps", "8", "--env-nums", "1",
        "--check-freq", "0", "--record-freq", "0", "--dump-freq", "0",
        "--log-root-dir", str(output), "--no-progress-bar", "--seed", "17",
        "--learning-starts", "1", "--batch-size", "2", "--buffer-size", "32",
        "--hidden-size", "32", "--device", "cpu",
    ], tmp_path)
    assert CONFIG.read_bytes() == original
    manifests = list(output.rglob("training_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["algorithm"] == "msgym"
    assert manifest["timesteps"] == 8
    model_path, norm_path = Path(manifest["model_path"]), Path(manifest["normalization_path"])
    assert model_path.is_file() and norm_path.is_file()
    run_isolated(["-c", f"""
import runpy
import torch
import numpy as np
script = runpy.run_path({str(SCRIPT)!r})
script['load_training_modules']()
from DynSyn import SAC_DynSyn
from stable_baselines3.common.vec_env import VecNormalize
from utils import create_vec_env
raw = create_vec_env('MSKBenchSquat-v0', dict(), 1, wrapper_list={{'MuscleNormWrapper': dict()}}, seed=17)
env = VecNormalize.load({str(norm_path)!r}, raw)
env.training = False
try:
    model = SAC_DynSyn.load({str(model_path)!r}, env=env, device='cpu')
    assert model._n_updates > 0
    assert model.num_timesteps == 8
    fresh = SAC_DynSyn('MlpPolicy', env=env, policy_kwargs=model.policy_kwargs, seed=17,
                      buffer_size=32, batch_size=2, device='cpu')
    before = dict(fresh.actor.named_parameters())
    assert any(not torch.equal(p, before[n]) for n, p in model.actor.named_parameters())
    action, _ = model.predict(env.reset(), deterministic=True)
    assert np.isfinite(action).all()
    assert env.action_space.contains(action[0])
    obs, rewards, _, _ = env.step(action)
    assert np.isfinite(obs).all() and np.isfinite(rewards).all()
finally:
    env.close()
"""], tmp_path)
