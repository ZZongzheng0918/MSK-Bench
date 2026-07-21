"""Stable-Baselines3 PPO trainer for the 22 MSK-Bench benchmark tasks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

def _ensure_registry_import_path() -> None:
    root = Path(__file__).resolve().parents[1]
    package_root = root / "MSK-Bench"
    for candidate in (package_root, root):
        candidate_text = str(candidate)
        if candidate.exists() and candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)


_ensure_registry_import_path()
from msk_bench.registry import CANONICAL_ENV_IDS  # noqa: E402
ALGORITHM_NAME = "PPO"
DEFAULT_ENV_ID = "MSKBenchWalk-v0"
MSK_BENCH_ENVS = CANONICAL_ENV_IDS


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_msk_bench_on_path() -> None:
    root = project_root()
    candidates = (root, root / "MSK-Bench")
    for candidate in candidates:
        if (candidate / "msk_bench" / "__init__.py").exists():
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)


def import_msk_bench():
    ensure_msk_bench_on_path()
    import msk_bench  # noqa: F401

    return msk_bench


def env_slug(env_id: str) -> str:
    name = env_id.removeprefix("MSKBench").removesuffix("-v0")
    return re.sub(r"(?<!^)([A-Z])", r"_\1", name).lower()


def selected_envs(env_id: str) -> tuple[str, ...]:
    if env_id == "all":
        return MSK_BENCH_ENVS
    return (env_id,)


@dataclass(frozen=True)
class SanitizeResult:
    value: object
    had_invalid: bool


def sanitize_value(np, value):
    if isinstance(value, dict):
        sanitized = {key: sanitize_value(np, item) for key, item in value.items()}
        return SanitizeResult(
            {key: item.value for key, item in sanitized.items()},
            any(item.had_invalid for item in sanitized.values()),
        )
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return SanitizeResult(value, False)
    if not np.issubdtype(array.dtype, np.number):
        return SanitizeResult(value, False)
    had_invalid = bool(np.isnan(array).any() or np.isinf(array).any())
    if had_invalid:
        return SanitizeResult(np.nan_to_num(value, nan=0.0, posinf=10.0, neginf=-10.0), True)
    return SanitizeResult(value, False)


def build_save_vec_normalize_callback(BaseCallback):
    class SaveVecNormalizeCallback(BaseCallback):
        def __init__(self, save_path: Path, verbose: int = 0):
            super().__init__(verbose)
            self.save_path = Path(save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)

        def _on_step(self) -> bool:
            vec_normalize = self.model.get_vec_normalize_env()
            if vec_normalize is not None:
                vec_normalize.save(str(self.save_path / "vec_normalize.pkl"))
            return True

    return SaveVecNormalizeCallback


def build_make_env(np, gym, Monitor):
    class NanSanitizerWrapper(gym.Wrapper):
        def reset(self, **kwargs):
            result = self.env.reset(**kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
            else:
                obs, info = result, {}
            clean_result = sanitize_value(np, obs)
            info = dict(info)
            if clean_result.had_invalid:
                info["sanitizer_had_invalid"] = True
            return clean_result.value, info

        def step(self, action):
            result = self.env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
            else:
                obs, reward, terminated, info = result
                truncated = False
            info = dict(info)
            clean_result = sanitize_value(np, obs)
            clean_obs = clean_result.value
            try:
                reward_is_bad = bool(np.isnan(reward) or np.isinf(reward))
            except TypeError:
                reward_is_bad = False
            if clean_result.had_invalid:
                reward = -100.0
                terminated = True
                info["error_flag"] = "obs_nan"
                info["sanitizer_had_invalid"] = True
            if reward_is_bad:
                reward = -100.0
                terminated = True
                info["error_flag"] = "reward_nan"
            return clean_obs, float(reward), bool(terminated), bool(truncated), info

    def make_env(env_id: str, seed: int = 0):
        def _init():
            import_msk_bench()
            env = gym.make(env_id)
            env.action_space.seed(seed)
            env = NanSanitizerWrapper(env)
            return Monitor(env)

        return _init

    return make_env


def load_training_dependencies():
    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    return {
        "BaseCallback": BaseCallback,
        "DummyVecEnv": DummyVecEnv,
        "EvalCallback": EvalCallback,
        "Monitor": Monitor,
        "PPO": PPO,
        "SubprocVecEnv": SubprocVecEnv,
        "VecNormalize": VecNormalize,
        "gym": gym,
        "np": np,
    }


def make_vec_env(deps, make_env, env_id: str, count: int, seed_offset: int):
    env_fns = [make_env(env_id, seed=seed_offset + index) for index in range(count)]
    if count == 1:
        return deps["DummyVecEnv"](env_fns)
    return deps["SubprocVecEnv"](env_fns, start_method="spawn")


def run_paths(output_root: Path, env_id: str) -> dict[str, Path]:
    run_dir = output_root / env_slug(env_id)
    paths = {
        "run": run_dir,
        "logs": run_dir / "logs",
        "checkpoints": run_dir / "checkpoints",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def dry_run_env(env_id: str) -> None:
    import_msk_bench()
    import gymnasium as gym

    env = gym.make(env_id)
    obs, _ = env.reset()
    print(f"{env_id}: loaded")
    print(f"  observation_space={env.observation_space}")
    print(f"  action_space={env.action_space}")
    env.close()


def train_one_env(args, env_id: str) -> None:
    import_msk_bench()
    deps = load_training_dependencies()
    make_env = build_make_env(deps["np"], deps["gym"], deps["Monitor"])
    SaveVecNormalizeCallback = build_save_vec_normalize_callback(deps["BaseCallback"])

    paths = run_paths(args.output_root, env_id)
    model_path = paths["checkpoints"] / "best_model.zip"
    norm_path = paths["checkpoints"] / "vec_normalize.pkl"

    env = make_vec_env(deps, make_env, env_id, args.num_envs, args.seed)
    eval_env = make_vec_env(deps, make_env, env_id, args.eval_envs, args.seed + 10_000)

    if args.resume and norm_path.exists():
        env = deps["VecNormalize"].load(str(norm_path), env)
        env.training = True
        env.norm_reward = True
        eval_env = deps["VecNormalize"].load(str(norm_path), eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        env = deps["VecNormalize"](env, norm_obs=True, norm_reward=True, clip_obs=args.clip_obs)
        eval_env = deps["VecNormalize"](
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=args.clip_obs,
            training=False,
        )

    policy_kwargs = {"net_arch": {"pi": [1024, 1024], "vf": [1024, 1024]}}
    if args.resume and model_path.exists():
        model = deps["PPO"].load(str(model_path), env=env, tensorboard_log=str(paths["logs"]))
        reset_num_timesteps = False
    else:
        model = deps["PPO"](
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=args.verbose,
            tensorboard_log=str(paths["logs"]),
        )
        reset_num_timesteps = True

    eval_callback = deps["EvalCallback"](
        eval_env,
        best_model_save_path=str(paths["checkpoints"]),
        log_path=str(paths["logs"] / "eval_logs"),
        eval_freq=max(args.eval_freq // max(args.num_envs, 1), 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        callback_on_new_best=SaveVecNormalizeCallback(paths["checkpoints"], verbose=args.verbose),
    )

    print(f"Training {ALGORITHM_NAME} on {env_id}")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        reset_num_timesteps=reset_num_timesteps,
    )
    model.save(str(paths["run"] / f"ppo_{env_slug(env_id)}_final"))
    env.save(str(paths["run"] / "vec_normalize_final.pkl"))
    env.close()
    eval_env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Train {ALGORITHM_NAME} on MSK-Bench tasks.")
    parser.add_argument("--env", choices=("all", *MSK_BENCH_ENVS), default=DEFAULT_ENV_ID)
    parser.add_argument("--list-envs", action="store_true", help="Print supported MSK-Bench environment ids and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Create and reset selected environment(s), then exit.")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--eval-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=200_000_000)
    parser.add_argument("--eval-freq", type=int, default=500_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--clip-obs", type=float, default=10.0)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true", default=True)
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_envs:
        for env_id in MSK_BENCH_ENVS:
            print(env_id)
        return 0
    if args.num_envs < 1 or args.eval_envs < 1:
        parser.error("--num-envs and --eval-envs must be positive")
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    for env_id in selected_envs(args.env):
        if args.dry_run:
            dry_run_env(env_id)
        else:
            train_one_env(args, env_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
