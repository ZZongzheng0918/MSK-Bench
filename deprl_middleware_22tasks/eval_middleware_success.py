"""Middleware success-rate evaluation for all 22 MSK-Bench tasks.

The script evaluates depRL/Tonic checkpoints trained with the middleware
environments. By default it searches for the newest run under:
    D:/MSK-Bench/depRL/baselines_MSKBench_Middleware/<tonic-name>/<timestamp>/checkpoints

You can also pass --run-path or --checkpoint-file explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any


os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "depRL", ROOT / "MSK-Bench", ROOT, Path(__file__).resolve().parent):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import eval_deprl_success as deprl_base


ALGORITHM_NAME = "middleware"
EVAL_MODE = "success"
DEFAULT_MIDDLEWARE_ROOT = Path("D:/MSK-Bench/deprl_middleware_22tasks")
DEFAULT_DEPRL_ROOT = Path("D:/MSK-Bench/depRL")
DEFAULT_BENCHMARK_ROOT = Path("D:/MSK-Bench")
DEFAULT_MODEL_ROOT = Path("D:/MSK-Bench/depRL/baselines_MSKBench_Middleware")
DEFAULT_CONFIG_DIR = Path("D:/MSK-Bench/deprl_middleware_22tasks/configs")

MSK_BENCH_ENVS = deprl_base.MSK_BENCH_ENVS
TASK_CONFIGS = deprl_base.TASK_CONFIGS

np = None
yaml = None
deprl = None
load_checkpoint = None
env_wrappers = None


env_slug = deprl_base.env_slug
selected_envs = deprl_base.selected_envs
parse_scales = deprl_base.parse_scales
max_steps_for = deprl_base.max_steps_for
NoiseWrapper = deprl_base.NoiseWrapper
unwrap_env = deprl_base.unwrap_env
apply_dynamics_noise = deprl_base.apply_dynamics_noise
EpisodeTracker = deprl_base.EpisodeTracker
EpisodeResult = deprl_base.EpisodeResult
run_episode = deprl_base.run_episode
summarize = deprl_base.summarize
print_success_row = deprl_base.print_success_row
print_robust_row = deprl_base.print_robust_row


def ensure_import_paths(args: argparse.Namespace) -> None:
    candidates = (
        Path(args.middleware_root),
        Path(args.deprl_root),
        Path(args.benchmark_root) / "MSK-Bench",
        Path(args.benchmark_root),
        Path(__file__).resolve().parent,
    )
    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_runtime(args: argparse.Namespace) -> None:
    global np, yaml, deprl, load_checkpoint, env_wrappers
    ensure_import_paths(args)
    import msk_bench  # noqa: F401
    import numpy as np_mod
    import yaml as yaml_mod
    import deprl as deprl_mod
    import deprl_middleware_22tasks.registry  # noqa: F401
    from deprl import env_wrappers as wrappers_mod
    from deprl.utils import load_checkpoint as load_checkpoint_func

    np = np_mod
    yaml = yaml_mod
    deprl = deprl_mod
    env_wrappers = wrappers_mod
    load_checkpoint = load_checkpoint_func

    deprl_base.np = np_mod
    deprl_base.yaml = yaml_mod
    deprl_base.deprl = deprl_mod
    deprl_base.env_wrappers = wrappers_mod
    deprl_base.load_checkpoint = load_checkpoint_func


def config_path_for(args: argparse.Namespace, env_id: str) -> Path:
    return Path(args.config_dir) / f"msk_bench_{env_slug(env_id)}_middleware.yaml"


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)


def latest_child(path: Path) -> Path | None:
    children = [child for child in path.iterdir() if child.is_dir()] if path.is_dir() else []
    if not children:
        return None
    return max(children, key=lambda item: item.stat().st_mtime)


def _candidate_model_roots(args: argparse.Namespace) -> tuple[Path, ...]:
    explicit = Path(args.model_root)
    fallback = Path(args.middleware_root) / "baselines_MSKBench_Middleware"
    roots = [explicit]
    if fallback != explicit:
        roots.append(fallback)
    return tuple(roots)


def resolve_run_path(args: argparse.Namespace, env_id: str) -> Path:
    if args.run_path:
        return Path(args.run_path)
    config = read_yaml(config_path_for(args, env_id))
    tonic_name = config["tonic"]["name"]
    for model_root in _candidate_model_roots(args):
        latest = latest_child(model_root / tonic_name)
        if latest is not None and (latest / "config.yaml").exists():
            return latest
    searched = ", ".join(str(root / tonic_name) for root in _candidate_model_roots(args))
    raise FileNotFoundError(f"No middleware run found for {env_id}; searched {searched}. Pass --run-path or --model-root.")


def checkpoint_reference(args: argparse.Namespace, run_path: Path) -> tuple[dict[str, Any], Path]:
    if args.checkpoint_file:
        checkpoint_file = Path(args.checkpoint_file)
        run_path = Path(str(checkpoint_file).split("checkpoints")[0])
        checkpoint = checkpoint_file.stem.replace("step_", "")
    else:
        checkpoint = args.checkpoint
    config, checkpoint_path, _ = load_checkpoint(str(run_path / "checkpoints"), checkpoint)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No middleware checkpoint found in {run_path / 'checkpoints'}")
    return config, Path(checkpoint_path)


def build_agent_env(args: argparse.Namespace, env_id: str, noise_type: str = "none", noise_scale: float = 0.0):
    run_path = resolve_run_path(args, env_id)
    config, checkpoint_path = checkpoint_reference(args, run_path)
    header = args.header or config["tonic"]["header"]
    agent_expr = args.agent or config["tonic"]["agent"]
    env_expr = args.environment or config["tonic"].get("test_environment") or config["tonic"]["environment"]
    if header:
        exec(header, globals())
    agent = eval(agent_expr, globals())
    env = eval(env_expr, globals())
    if hasattr(env, "seed"):
        env.seed(args.seed)
    env = env_wrappers.apply_wrapper(env)
    if "env_args" in config and config["env_args"] is not None:
        env.merge_args(config["env_args"])
        env.apply_args()
    apply_dynamics_noise(env, noise_scale if noise_type == "dynamics" else 0.0)
    env = NoiseWrapper(env, noise_type, noise_scale)
    if "mpo_args" in config:
        agent.set_params(**config["mpo_args"])
    agent.initialize(observation_space=env.observation_space, action_space=env.action_space, seed=args.seed)
    agent.load(str(checkpoint_path), only_checkpoint=True)
    return agent, env, run_path


def evaluate_success(args: argparse.Namespace) -> list[dict[str, Any]]:
    load_runtime(args)
    rows = []
    for env_id in selected_envs(args.env):
        agent, env, run_path = build_agent_env(args, env_id)
        results = [run_episode(agent, env, env_id, args.noisy, max_steps_for(env_id, args.max_steps)) for _ in range(args.episodes)]
        row = {"algorithm": ALGORITHM_NAME, "env_id": env_id, "run_path": str(run_path), **summarize(results)}
        rows.append(row)
        print_success_row(row)
        if hasattr(env, "close"):
            env.close()
    return rows


def evaluate_robustness(args: argparse.Namespace) -> list[dict[str, Any]]:
    load_runtime(args)
    scale_map = {
        "action": parse_scales(args.action_scales),
        "obs": parse_scales(args.obs_scales),
        "dynamics": parse_scales(args.dynamics_scales),
    }
    noise_types = tuple(scale_map) if args.noise_type == "all" else (args.noise_type,)
    rows = []
    for env_id in selected_envs(args.env):
        max_steps = max_steps_for(env_id, args.max_steps)
        for noise_type in noise_types:
            for scale in scale_map[noise_type]:
                agent, env, run_path = build_agent_env(args, env_id, noise_type, scale)
                results = [run_episode(agent, env, env_id, args.noisy, max_steps) for _ in range(args.episodes)]
                row = {"algorithm": ALGORITHM_NAME, "env_id": env_id, "noise_type": noise_type, "noise_scale": scale, "run_path": str(run_path), **summarize(results)}
                rows.append(row)
                print_robust_row(row)
                if hasattr(env, "close"):
                    env.close()
    return rows


def write_rows(rows: list[dict[str, Any]], json_path: str | None, csv_path: str | None) -> None:
    if json_path:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if csv_path and rows:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Middleware {EVAL_MODE} evaluation for MSK-Bench.")
    parser.add_argument("--env", choices=("all", *MSK_BENCH_ENVS), default="all")
    parser.add_argument("--list-envs", action="store_true")
    parser.add_argument("--episodes", "-n", type=int, default=50 if EVAL_MODE == "success" else 30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noisy", action="store_true", help="Use depRL noisy_test_step when available.")
    parser.add_argument("--middleware-root", type=Path, default=DEFAULT_MIDDLEWARE_ROOT)
    parser.add_argument("--deprl-root", type=Path, default=DEFAULT_DEPRL_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--run-path", type=Path, default=None)
    parser.add_argument("--checkpoint", default="last")
    parser.add_argument("--checkpoint-file", type=Path, default=None)
    parser.add_argument("--header", default=None)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--environment", "--env-expr", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    if EVAL_MODE == "robustness":
        parser.add_argument("--noise-type", choices=("all", "action", "obs", "dynamics"), default="all")
        parser.add_argument("--action-scales", default="0,0.02,0.05,0.08,0.12")
        parser.add_argument("--obs-scales", default="0,0.01,0.02,0.04,0.06")
        parser.add_argument("--dynamics-scales", default="0,0.05,0.10,0.15,0.20")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_envs:
        for env_id in MSK_BENCH_ENVS:
            print(env_id)
        return 0
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    rows = evaluate_success(args) if EVAL_MODE == "success" else evaluate_robustness(args)
    write_rows(rows, args.json, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
