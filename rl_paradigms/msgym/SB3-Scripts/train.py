"""
DynSyn-SAC training entry point for the MSK-Bench msgym benchmark.

Examples:
    python SB3-Scripts/train.py --list-configs
    python SB3-Scripts/train.py -f configs/msk_bench_walk.json
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, List

from rl_paradigms.training_contract import TrainingArtifact

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
MSK_BENCH_SOURCE = PROJECT_ROOT.parents[1]

for path in (PROJECT_ROOT, SCRIPT_DIR, MSK_BENCH_SOURCE):
    path_text = str(path)
    if path.exists() and path_text not in sys.path:
        sys.path.insert(0, path_text)

AGENT_REGISTRY: dict[str, Any] = {}
POLICY_REGISTRY: dict[str, Any] = {"MlpPolicy": "MlpPolicy", "CnnPolicy": "CnnPolicy", "MultiInputPolicy": "MultiInputPolicy"}
SCHEDULE_REGISTRY: dict[str, Any] = {}
_CUSTOM_AGENTS: dict[str, Any] = {}


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def list_config_files() -> list[Path]:
    config_dir = PROJECT_ROOT / "configs"
    if not config_dir.is_dir():
        return []
    return sorted(config_dir.glob("*.json"))


def load_training_modules() -> None:
    global CheckpointCallback, EvalCallback, SaveConfigToTensorboardCallback
    global SaveVecNormalizeOnBestCallback, TensorboardCallback, VideoRecorderCallback
    global VecNormalize, create_vec_env, gymnasium, linear_schedule, np, sb3
    global sb3_contrib, set_random_seed, torch, AGENT_REGISTRY, SCHEDULE_REGISTRY, _CUSTOM_AGENTS

    import gymnasium
    import numpy as np
    import sb3_contrib
    import stable_baselines3 as sb3
    import torch
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import VecNormalize

    from DynSyn import SAC_DynSyn
    from callback import (
        SaveConfigToTensorboardCallback,
        SaveVecNormalizeOnBestCallback,
        TensorboardCallback,
        VideoRecorderCallback,
    )
    from schedule import linear_schedule
    from utils import create_vec_env

    _CUSTOM_AGENTS = {
        "SAC_DynSyn": SAC_DynSyn,
    }
    AGENT_REGISTRY = {
        name: getattr(sb3, name)
        for name in ("A2C", "DDPG", "DQN", "PPO", "SAC", "TD3")
        if hasattr(sb3, name)
    }
    AGENT_REGISTRY.update(
        {
            name: getattr(sb3_contrib, name)
            for name in ("ARS", "CrossQ", "MaskablePPO", "QRDQN", "RecurrentPPO", "TQC", "TRPO")
            if hasattr(sb3_contrib, name)
        }
    )
    AGENT_REGISTRY.update(_CUSTOM_AGENTS)
    SCHEDULE_REGISTRY = {"linear_schedule": linear_schedule}


def set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_random_seed(seed, using_cuda=torch.cuda.is_available())
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _ensure_env_registered(env_name: str) -> None:
    if isinstance(env_name, str) and env_name.startswith("msgym/"):
        import msgym  # noqa: F401
    elif isinstance(env_name, str) and env_name.startswith("MSKBench"):
        import msk_bench  # noqa: F401


def load_policy(args: argparse.Namespace) -> Any:
    policy = args.agent_kwargs.pop("policy", None)
    policy = "MlpPolicy" if policy is None else policy
    if policy not in POLICY_REGISTRY:
        raise ValueError(f"Unknown policy '{policy}'. Available policies: {sorted(POLICY_REGISTRY)}")
    return POLICY_REGISTRY[policy]


def resolve_schedule(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    try:
        return float(stripped)
    except ValueError:
        pass
    match = re.fullmatch(r"linear_schedule\(([-+0-9.eE]+)\)", stripped)
    if match:
        return SCHEDULE_REGISTRY["linear_schedule"](float(match.group(1)))
    if stripped in SCHEDULE_REGISTRY:
        return SCHEDULE_REGISTRY[stripped]
    raise ValueError(f"Unknown learning-rate schedule '{value}'. Available schedules: {sorted(SCHEDULE_REGISTRY)}")


def register_callback(
    args: argparse.Namespace,
    video_dir: str,
    log_dir: str,
    config_str: str,
    eval_env: Any,
    checkpoint_dir: str,
) -> List[Any]:
    callback_list = []
    args.check_freq //= args.env_nums
    args.record_freq //= args.env_nums
    args.dump_freq //= args.env_nums
    callback_list.append(SaveConfigToTensorboardCallback(log_dir, config_str))

    if args.check_freq > 0:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=checkpoint_dir,
            log_path=os.path.join(log_dir, "eval"),
            eval_freq=args.check_freq,
            n_eval_episodes=getattr(args, "eval_episodes", 3),
            deterministic=True,
            render=False,
            callback_on_new_best=SaveVecNormalizeOnBestCallback(
                save_path=os.path.join(checkpoint_dir, "best_env.zip"),
                verbose=1,
            ),
            verbose=1,
        )
        callback_list.append(eval_callback)

    if args.dump_freq > 0:
        callback_list.append(
            CheckpointCallback(
                save_freq=args.dump_freq,
                save_path=checkpoint_dir,
                name_prefix="model",
                save_replay_buffer=getattr(args, "save_replay_buffer", False),
                verbose=1,
            )
        )
    if args.record_freq > 0:
        callback_list.append(
            VideoRecorderCallback(args, args.record_freq, video_dir=video_dir, video_ep_num=5, verbose=1)
        )

    callback_list.append(TensorboardCallback(getattr(args, "info_keywords", {}), reward_freq=args.reward_freq))
    return callback_list


def build_env(args: argparse.Namespace, monitor_dir: str) -> Any:
    vec_env = create_vec_env(
        args.env_name,
        args.single_env_kwargs,
        args.env_nums,
        wrapper_list=args.wrapper_list,
        monitor_dir=monitor_dir,
        monitor_kwargs=getattr(args, "monitor_kwargs", {}),
        seed=args.seed,
    )
    if args.vec_normalize["is_norm"] and not args.load_model_dir:
        vec_env = VecNormalize(vec_env, **args.vec_normalize["kwargs"])
    return vec_env


def build_eval_env(args: argparse.Namespace) -> Any:
    eval_env = create_vec_env(
        args.env_name,
        args.single_env_kwargs,
        1,
        wrapper_list=args.wrapper_list,
        monitor_dir=None,
        monitor_kwargs=None,
        seed=args.seed + 1000,
        render_mode=None,
    )
    if args.vec_normalize["is_norm"]:
        eval_env = VecNormalize(eval_env, training=False, **args.vec_normalize["kwargs"])
    return eval_env


def find_env_file(env_name: str) -> Any:
    _ensure_env_registered(env_name)
    env_spec = gymnasium.spec(env_name)
    module_path, class_name = env_spec.entry_point.split(":")
    module = importlib.import_module(module_path)
    env_dir = os.path.dirname(module.__file__)

    init_file = os.path.join(env_dir, "__init__.py")
    if os.path.exists(init_file):
        with open(init_file, "r", encoding="utf-8") as f:
            init_content = f.read()
        import_lines = [line.strip() for line in init_content.split("\n") if class_name in line]
        for line in import_lines:
            if "from" in line and "import" in line:
                from_part = line.split("from")[1].split("import")[0].strip()
                module_name = from_part.split(".")[-1]
                env_file = os.path.join(env_dir, f"{module_name}.py")
                if os.path.exists(env_file):
                    return env_file
    return None


def train(args: argparse.Namespace, config_str: str) -> None:
    load_training_modules()
    set_global_determinism(getattr(args, "seed", 0))
    _ensure_env_registered(args.env_name)

    log_name = args.config_name
    env_name_log = args.env_name.split("/")[-1]
    log_dir = os.path.join(args.log_root_dir, env_name_log, time.strftime("%m%d-%H%M%S") + "_" + str(args.seed))

    monitor_dir = os.path.join(log_dir, "monitor")
    checkpoint_dir = os.path.join(log_dir, "checkpoint")
    video_dir = os.path.join(log_dir, "video")
    for directory in [log_dir, monitor_dir, checkpoint_dir, video_dir]:
        os.makedirs(directory, exist_ok=True)

    with open(os.path.join(log_dir, log_name + ".json"), "w", encoding="utf-8") as f:
        f.write(config_str)

    env_file_path = find_env_file(args.env_name)
    if env_file_path:
        target_path = os.path.join(log_dir, os.path.basename(env_file_path))
        shutil.copy2(env_file_path, target_path)
        print(f"Environment file copied to: {target_path}")
    else:
        print(f"Warning: Could not find environment file for {args.env_name}")

    with ExitStack() as resources:
        eval_env = build_eval_env(args)
        resources.callback(eval_env.close)
        callback_list = register_callback(args, video_dir, log_dir, config_str, eval_env=eval_env, checkpoint_dir=checkpoint_dir)

        Agent = AGENT_REGISTRY.get(args.agent)
        if Agent is None:
            raise ValueError(f"Unknown agent '{args.agent}'. Available agents: {sorted(AGENT_REGISTRY)}")

        if "learning_rate" in args.agent_kwargs:
            args.agent_kwargs["learning_rate"] = resolve_schedule(args.agent_kwargs["learning_rate"])
        args.agent_kwargs["seed"] = args.seed

        vec_env = build_env(args, monitor_dir)
        resources.callback(vec_env.close)
        if args.load_model_dir:
            if os.path.isdir(args.load_model_dir):
                env_path = os.path.join(args.load_model_dir, "best_env.zip")
                model_path = os.path.join(args.load_model_dir, "best_model.zip")
            else:
                env_path = args.load_model_dir.replace("model", "env")
                model_path = args.load_model_dir
            print(f"Loading model from {model_path}")
            vec_env = VecNormalize.load(env_path, vec_env)
            load_kwargs = getattr(args, "load_kwargs", {})
            model = Agent.load(model_path, env=vec_env, verbose=1, tensorboard_log=log_dir, **load_kwargs)
            model.learning_rate = args.agent_kwargs["learning_rate"]
            model._setup_lr_schedule()
            if getattr(args, "load_buffer", False):
                model.load_replay_buffer(os.path.join(args.load_model_dir, "best_replay_buffer.zip"))
        else:
            policy = load_policy(args)
            model = Agent(policy, env=vec_env, verbose=1, tensorboard_log=log_dir, **args.agent_kwargs)

        model.learn(
            total_timesteps=args.total_timesteps,
            progress_bar=not getattr(args, 'no_progress_bar', False),
            callback=callback_list,
            tb_log_name=log_name,
            log_interval=100,
            reset_num_timesteps=False,
        )

        model_path = Path(checkpoint_dir) / "final_model.zip"
        model.save(model_path)
        norm_path = None
        if isinstance(vec_env, VecNormalize):
            norm_path = Path(checkpoint_dir) / "final_env.zip"
            vec_env.save(norm_path)
        if getattr(args, "save_replaybuffer", False) and hasattr(model, "save_replay_buffer"):
            model.save_replay_buffer(os.path.join(checkpoint_dir, "final_replay_buffer.zip"))
        verification_env = create_vec_env(
            args.env_name, args.single_env_kwargs, 1, wrapper_list=args.wrapper_list, seed=args.seed,
        )
        resources.callback(verification_env.close)
        if norm_path is not None:
            verification_env = VecNormalize.load(norm_path, verification_env)
            verification_env.training = False
            verification_env.norm_reward = False
        restored = Agent.load(model_path, env=verification_env, device=args.agent_kwargs.get("device", "auto"))
        observation = verification_env.reset()
        action, _ = restored.predict(observation, deterministic=True)
        expected, _ = model.predict(observation, deterministic=True)
        np.testing.assert_allclose(action, expected, rtol=1e-5, atol=1e-6)
        if not np.isfinite(action).all() or not verification_env.action_space.contains(action[0]):
            raise ValueError("Reloaded msgym policy returned invalid actions")
        TrainingArtifact(
            algorithm="msgym", env_id=args.env_name, seed=args.seed, timesteps=model.num_timesteps,
            model_path=model_path, normalization_path=norm_path,
        ).write(Path(log_dir))


def parse_args(argv=None) -> tuple[argparse.Namespace | None, str | None]:
    parser = argparse.ArgumentParser(description="Train a DynSyn-SAC benchmark agent")
    parser.add_argument("--config_file", "-f", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--list-configs", action="store_true", help="List bundled config files and exit")
    for flag in ("total-timesteps", "env-nums", "check-freq", "record-freq", "dump-freq",
                 "seed", "learning-starts", "batch-size", "buffer-size", "hidden-size"):
        parser.add_argument("--" + flag, type=int)
    parser.add_argument("--log-root-dir", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-progress-bar", action="store_true")
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, int) and not isinstance(value, bool):
            minimum = 1 if key in {"total_timesteps", "env_nums", "batch_size", "buffer_size", "hidden_size"} else 0
            if value < minimum:
                parser.error(f"--{key.replace('_', '-')} must be >= {minimum}")

    if args.list_configs:
        for config_file in list_config_files():
            print(config_file.name)
        return None, None
    if args.config_file is None:
        parser.error("--config_file is required unless --list-configs is used")

    config_path = resolve_project_path(args.config_file)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("total_timesteps", "env_nums", "check_freq", "record_freq", "dump_freq", "seed"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.log_root_dir is not None:
        config["log_root_dir"] = str(args.log_root_dir.resolve())
    for key in ("learning_starts", "batch_size", "buffer_size", "device"):
        if getattr(args, key) is not None:
            config["agent_kwargs"][key] = getattr(args, key)
    if args.hidden_size is not None:
        config["agent_kwargs"].setdefault("policy_kwargs", {})["net_arch"] = {
            "pi": [args.hidden_size, args.hidden_size], "qf": [args.hidden_size, args.hidden_size],
        }
    config["no_progress_bar"] = args.no_progress_bar
    config_str = json.dumps(config, indent=2)

    arg_config = argparse.Namespace(**config)
    arg_config.log_root_dir = str(resolve_project_path(arg_config.log_root_dir))
    arg_config.total_config = config
    arg_config.config_name = config_path.stem
    arg_config.config_file = str(config_path)
    return arg_config, config_str


def main() -> None:
    args, config_str = parse_args()
    if args is None:
        return
    train(args, config_str or "")


if __name__ == "__main__":
    main()
