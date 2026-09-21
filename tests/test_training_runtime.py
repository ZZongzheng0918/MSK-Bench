from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module_name",
    (
        "rl_paradigms.ppo.train_ppo_msk_bench",
        "rl_paradigms.sac.train_sac_msk_bench",
    ),
)
def test_sb3_trainers_accept_smoke_overrides(module_name: str) -> None:
    module = importlib.import_module(module_name)
    args = module.build_parser().parse_args(
        [
            "--env",
            "MSKBenchSquat-v0",
            "--total-timesteps",
            "8",
            "--num-envs",
            "1",
            "--eval-envs",
            "1",
            "--eval-freq",
            "0",
            "--output-root",
            "artifacts",
            "--hidden-size",
            "32",
            "--no-resume",
        ]
    )
    assert args.total_timesteps == 8
    assert args.num_envs == 1
    assert args.eval_freq == 0
    assert args.hidden_size == 32
    assert not args.resume


@pytest.mark.parametrize(
    ("algorithm", "module_name", "extra_args"),
    (
        (
            "ppo",
            "rl_paradigms.ppo.train_ppo_msk_bench",
            ["--n-steps", "4", "--batch-size", "4", "--n-epochs", "1"],
        ),
        (
            "sac",
            "rl_paradigms.sac.train_sac_msk_bench",
            [
                "--learning-starts",
                "1",
                "--buffer-size",
                "32",
                "--batch-size",
                "2",
                "--train-freq",
                "1",
                "--gradient-steps",
                "1",
            ],
        ),
    ),
)
def test_sb3_short_training_saves_reloadable_artifacts(
    algorithm: str,
    module_name: str,
    extra_args: list[str],
    tmp_path: Path,
) -> None:
    output_root = tmp_path / algorithm
    command = [
        sys.executable,
        "-B",
        "-I",
        "-m",
        module_name,
        "--env",
        "MSKBenchSquat-v0",
        "--total-timesteps",
        "8",
        "--num-envs",
        "1",
        "--eval-envs",
        "1",
        "--eval-freq",
        "0",
        "--output-root",
        str(output_root),
        "--hidden-size",
        "32",
        "--verbose",
        "0",
        "--no-resume",
        *extra_args,
    ]
    env = {key: value for key, value in os.environ.items() if key.upper() != "PYTHONPATH"}
    completed = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True)
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "humanoid-bench" not in output.lower()

    run_dir = output_root / "squat"
    model_path = run_dir / "final_model.zip"
    norm_path = run_dir / "vec_normalize.pkl"
    manifest_path = run_dir / "training_manifest.json"
    assert model_path.is_file() and model_path.stat().st_size > 0
    assert norm_path.is_file() and norm_path.stat().st_size > 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["algorithm"] == algorithm
    assert manifest["env_id"] == "MSKBenchSquat-v0"
    assert manifest["timesteps"] >= 8
    assert Path(manifest["model_path"]) == model_path.resolve()
    assert Path(manifest["normalization_path"]) == norm_path.resolve()
