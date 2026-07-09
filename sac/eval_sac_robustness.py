"""SAC robustness evaluation for MSK-Bench.

The script evaluates all 22 canonical MSK-Bench tasks. It loads Stable-Baselines3
weights from the training layout used in this repository:
    <model-root>/<task-slug>/checkpoints/best_model.zip
    <model-root>/<task-slug>/checkpoints/vec_normalize.pkl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore")

ALGORITHM_NAME = "SAC"
EVAL_MODE = "robustness"
DEFAULT_BENCHMARK_ROOT = Path("D:/MSK-Bench")
DEFAULT_MODEL_ROOT = Path("D:/MSK-Bench/sac/runs")
MSK_BENCH_ENVS = (
    "MSKBenchStand-v0",
    "MSKBenchPowerlift-v0",
    "MSKBenchSingleLegStand-v0",
    "MSKBenchSit-v0",
    "MSKBenchBalance-v0",
    "MSKBenchSquat-v0",
    "MSKBenchWalk-v0",
    "MSKBenchCrawl-v0",
    "MSKBenchRun-v0",
    "MSKBenchJump-v0",
    "MSKBenchWalkTurn-v0",
    "MSKBenchSidestep-v0",
    "MSKBenchStairs-v0",
    "MSKBenchHurdle-v0",
    "MSKBenchStepStones-v0",
    "MSKBenchSlide-v0",
    "MSKBenchDoorOpen-v0",
    "MSKBenchReach-v0",
    "MSKBenchWalkAndSit-v0",
    "MSKBenchChinUp-v0",
    "MSKBenchCatch-v0",
    "MSKBenchPoleWalk-v0",
)
TASK_CONFIGS = {
    "MSKBenchStand-v0": {"kind": "stand", "steps": 1000},
    "MSKBenchPowerlift-v0": {"kind": "powerlift", "height": 1.6, "steps": 900},
    "MSKBenchSingleLegStand-v0": {"kind": "survival", "steps": 1000},
    "MSKBenchSit-v0": {"kind": "sit", "steps": 1000},
    "MSKBenchBalance-v0": {"kind": "balance", "steps": 1000},
    "MSKBenchSquat-v0": {"kind": "squat", "depth": 0.75, "stand": 0.90, "steps": 1000},
    "MSKBenchWalk-v0": {"kind": "walk", "steps": 1000},
    "MSKBenchCrawl-v0": {"kind": "crawl", "distance": 2.0, "steps": 1000},
    "MSKBenchRun-v0": {"kind": "forward_distance", "distance": 15.0, "steps": 1000},
    "MSKBenchJump-v0": {"kind": "jump", "hops": 10, "steps": 1000},
    "MSKBenchWalkTurn-v0": {"kind": "walk_turn", "steps": 2000},
    "MSKBenchSidestep-v0": {"kind": "sidestep", "distance": 9.0, "steps": 1000},
    "MSKBenchStairs-v0": {"kind": "stairs", "height": 1.6, "steps": 2000},
    "MSKBenchHurdle-v0": {"kind": "hurdle", "steps": 1000},
    "MSKBenchStepStones-v0": {"kind": "stones", "steps": 1000},
    "MSKBenchSlide-v0": {"kind": "forward_distance", "distance": 18.0, "steps": 1500},
    "MSKBenchDoorOpen-v0": {"kind": "door", "distance": 2.0, "steps": 1000},
    "MSKBenchReach-v0": {"kind": "reach", "touches": 2, "steps": 500},
    "MSKBenchWalkAndSit-v0": {"kind": "walk_and_sit", "steps": 500},
    "MSKBenchChinUp-v0": {"kind": "chinup", "hold": 25, "steps": 500},
    "MSKBenchCatch-v0": {"kind": "catch", "steps": 200},
    "MSKBenchPoleWalk-v0": {"kind": "forward_distance", "distance": 1.6, "steps": 1000},
}

np = None
gym = None
Monitor = None
DummyVecEnv = None
VecNormalize = None
MODEL_CLASS = None


def project_paths(benchmark_root: Path) -> tuple[Path, ...]:
    return (
        benchmark_root,
        benchmark_root / "MSK-Bench",
        Path(__file__).resolve().parent,
    )


def ensure_import_paths(benchmark_root: Path) -> None:
    for path in project_paths(benchmark_root):
        if path.exists():
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)


def load_runtime(benchmark_root: Path) -> None:
    global np, gym, Monitor, DummyVecEnv, VecNormalize, MODEL_CLASS
    ensure_import_paths(benchmark_root)
    import msk_bench  # noqa: F401
    import gymnasium as gym_mod
    import numpy as np_mod
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor as MonitorCls
    from stable_baselines3.common.vec_env import DummyVecEnv as DummyVecEnvCls
    from stable_baselines3.common.vec_env import VecNormalize as VecNormalizeCls

    np = np_mod
    gym = gym_mod
    Monitor = MonitorCls
    DummyVecEnv = DummyVecEnvCls
    VecNormalize = VecNormalizeCls
    MODEL_CLASS = SAC


def env_slug(env_id: str) -> str:
    name = env_id.removeprefix("MSKBench").removesuffix("-v0")
    return re.sub(r"(?<!^)([A-Z])", r"_\1", name).lower()


def selected_envs(env_arg: str) -> tuple[str, ...]:
    return MSK_BENCH_ENVS if env_arg == "all" else (env_arg,)


def parse_scales(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def max_steps_for(env_id: str, cli_max_steps: int | None) -> int:
    if cli_max_steps is not None:
        return cli_max_steps
    return int(TASK_CONFIGS[env_id].get("steps", 1000))


def sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_value(item) for key, item in value.items()}
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return value
    if not np.issubdtype(array.dtype, np.number):
        return value
    if np.isnan(array).any() or np.isinf(array).any():
        return np.nan_to_num(value, nan=0.0, posinf=10.0, neginf=-10.0)
    return value


class EpisodeTracker:
    def __init__(self, env_id: str):
        self.env_id = env_id
        self.config = TASK_CONFIGS[env_id]
        self.kind = self.config["kind"]
        self.reset()

    def reset(self) -> None:
        self.start_x = None
        self.start_y = None
        self.max_x = -float("inf")
        self.min_x = float("inf")
        self.max_abs_y_delta = 0.0
        self.max_z = -float("inf")
        self.min_z = float("inf")
        self.max_bar_z = -float("inf")
        self.max_board_drift = float("inf")
        self.best_board_drift = float("inf")
        self.hops = 0
        self.airborne = False
        self.squat_state = 0
        self.max_hurdles = 0
        self.total_hurdles = None
        self.max_stone = 0
        self.total_stones = None
        self.max_target_idx = 0
        self.max_touches = 0
        self.has_sat_down = False
        self.max_chinup_hold = 0
        self.current_chinup_hold = 0
        self.caught = False
        self.door_opened = False
        self.max_door_angle = 0.0
        self.solved_seen = False

    def on_reset(self, env: Any) -> None:
        base = unwrap_env(env)
        try:
            qpos = np.asarray(base.sim.data.qpos)
            self.start_x = float(qpos[0])
            self.start_y = float(qpos[1])
        except Exception:
            self.start_x = 0.0
            self.start_y = 0.0
        self.observe(base, {}, 0)

    def observe(self, env: Any, info: dict[str, Any], step: int) -> None:
        base = unwrap_env(env)
        self.solved_seen = self.solved_seen or bool_from_info(info, "solved")
        try:
            qpos = np.asarray(base.sim.data.qpos)
            x = float(qpos[0])
            y = float(qpos[1])
            z = float(qpos[2])
            if self.start_x is None:
                self.start_x = x
            if self.start_y is None:
                self.start_y = y
            self.max_x = max(self.max_x, x)
            self.min_x = min(self.min_x, x)
            self.max_abs_y_delta = max(self.max_abs_y_delta, abs(y - self.start_y))
            self.max_z = max(self.max_z, z)
            self.min_z = min(self.min_z, z)
            self._track_jump(z)
            self._track_squat(z)
        except Exception:
            pass
        self._track_named_attrs(base)
        self._track_bodies(base)
        self._track_info(info)

    def _track_jump(self, pelvis_z: float) -> None:
        if pelvis_z > 1.00 and not self.airborne:
            self.airborne = True
        if pelvis_z < 0.95 and self.airborne:
            self.hops += 1
            self.airborne = False

    def _track_squat(self, pelvis_z: float) -> None:
        if self.squat_state == 0 and pelvis_z < float(self.config.get("depth", 0.75)):
            self.squat_state = 1
        if self.squat_state == 1 and pelvis_z > float(self.config.get("stand", 0.90)):
            self.squat_state = 2

    def _track_named_attrs(self, base: Any) -> None:
        for name in ("hurdles_cleared",):
            value = getattr(base, name, None)
            if value is not None:
                self.max_hurdles = max(self.max_hurdles, int(value))
        hurdles = getattr(base, "hurdles", None)
        if hurdles is not None:
            try:
                self.total_hurdles = len(hurdles)
            except Exception:
                pass
        stone_idx = getattr(base, "current_stone_idx", None)
        if stone_idx is not None:
            self.max_stone = max(self.max_stone, int(stone_idx))
        stones = getattr(base, "stones", None)
        if stones is not None:
            try:
                self.total_stones = len(stones)
            except Exception:
                pass
        target_idx = getattr(base, "current_target_idx", None)
        if target_idx is not None:
            self.max_target_idx = max(self.max_target_idx, int(target_idx))
        touch_count = getattr(base, "touch_count", None)
        if touch_count is not None:
            self.max_touches = max(self.max_touches, int(touch_count))
        self.has_sat_down = self.has_sat_down or bool(getattr(base, "has_sat_down", False))
        self.door_opened = self.door_opened or bool(getattr(base, "door_opened_flag", False))
        self.max_door_angle = max(self.max_door_angle, float(getattr(base, "max_door_angle", 0.0) or 0.0))

    def _track_bodies(self, base: Any) -> None:
        sim = getattr(base, "sim", None)
        if sim is None:
            return
        self.max_bar_z = max(self.max_bar_z, body_z(base, "dumbbell", -float("inf")))
        if self.kind == "balance":
            board = body_xy(base, "board")
            ball = body_xy(base, "ball")
            if board is not None and ball is not None:
                drift = float(np.linalg.norm(board - ball))
                self.best_board_drift = min(self.best_board_drift, drift)
                self.max_board_drift = drift
        if self.kind == "chinup":
            head_z = body_z(base, "head", -float("inf"))
            bar_height = float(getattr(base, "bar_height", 2.3))
            if head_z > bar_height + 0.1:
                self.current_chinup_hold += 1
                self.max_chinup_hold = max(self.max_chinup_hold, self.current_chinup_hold)
            else:
                self.current_chinup_hold = 0
        if self.kind == "catch":
            self.caught = self.caught or bool_from_reward_key(base, "catch_bonus", threshold=0.5)

    def _track_info(self, info: dict[str, Any]) -> None:
        self.max_touches = max(self.max_touches, int(float_value(info.get("rwd_sparse"), 0.0)))
        self.caught = self.caught or bool_from_info(info, "catch_bonus")
        self.door_opened = self.door_opened or bool_from_info(info, "open_door")

    def success(self, steps: int, terminated: bool) -> bool:
        if self.solved_seen:
            return True
        kind = self.kind
        dx = self.max_x - (self.start_x or 0.0)
        back = (self.start_x or 0.0) - self.min_x
        if kind in {"stand", "survival"}:
            return steps >= int(self.config.get("steps", 1000)) and not terminated
        if kind == "powerlift":
            return steps >= int(self.config.get("steps", 900)) and self.max_bar_z >= float(self.config.get("height", 1.6))
        if kind == "balance":
            return steps > 350 and self.best_board_drift < 0.5
        if kind == "squat":
            return self.squat_state >= 2
        if kind == "walk":
            return steps >= int(self.config.get("steps", 1000)) and not terminated
        if kind == "crawl":
            return back >= float(self.config.get("distance", 2.0))
        if kind == "forward_distance":
            return dx >= float(self.config.get("distance", 1.0))
        if kind == "jump":
            return self.hops >= int(self.config.get("hops", 10))
        if kind == "walk_turn":
            return self.max_target_idx >= 2
        if kind == "sidestep":
            return self.max_abs_y_delta >= float(self.config.get("distance", 9.0))
        if kind == "stairs":
            return self.max_z >= float(self.config.get("height", 1.6))
        if kind == "hurdle":
            total = self.total_hurdles if self.total_hurdles is not None else 5
            return self.max_hurdles >= max(total, 1)
        if kind == "stones":
            total = self.total_stones if self.total_stones is not None else 1
            return self.max_stone >= max(total - 1, 0)
        if kind == "door":
            return dx >= float(self.config.get("distance", 2.0)) or self.door_opened or self.max_door_angle > 1.0
        if kind == "reach":
            return self.max_touches >= int(self.config.get("touches", 2))
        if kind == "walk_and_sit":
            return self.has_sat_down and steps >= int(self.config.get("steps", 500))
        if kind == "chinup":
            return self.max_chinup_hold >= int(self.config.get("hold", 25))
        if kind == "catch":
            return self.caught and steps >= int(self.config.get("steps", 200))
        if kind == "sit":
            return self.has_sat_down or self.solved_seen
        return self.solved_seen

    def summary_metric(self) -> float:
        kind = self.kind
        if kind == "crawl":
            return (self.start_x or 0.0) - self.min_x
        if kind == "sidestep":
            return self.max_abs_y_delta
        if kind == "jump":
            return float(self.hops)
        if kind == "powerlift":
            return self.max_bar_z
        if kind == "stairs":
            return self.max_z
        if kind == "hurdle":
            return float(self.max_hurdles)
        if kind == "stones":
            return float(self.max_stone)
        if kind == "reach":
            return float(self.max_touches)
        if kind == "chinup":
            return float(self.max_chinup_hold)
        return self.max_x - (self.start_x or 0.0)


def unwrap_env(env: Any) -> Any:
    current = env
    seen = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    if hasattr(current, "unwrapped"):
        return current.unwrapped
    return current


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        array = np.asarray(value)
        return float(array.reshape(-1)[0])
    except Exception:
        return default


def bool_from_info(info: dict[str, Any], key: str) -> bool:
    candidates = (key, f"rwd_{key}")
    for candidate in candidates:
        if candidate in info and float_value(info[candidate], 0.0) > 0.5:
            return True
    rwd_dict = info.get("rwd_dict")
    if isinstance(rwd_dict, dict) and key in rwd_dict:
        return float_value(rwd_dict[key], 0.0) > 0.5
    return False


def bool_from_reward_key(base: Any, key: str, threshold: float = 0.5) -> bool:
    try:
        rwd_dict = getattr(base, "rwd_dict", {})
        return float_value(rwd_dict.get(key), 0.0) > threshold
    except Exception:
        return False


def body_z(base: Any, name: str, default: float) -> float:
    try:
        body_id = base.sim.model.body_name2id(name)
        return float(base.sim.data.body_xpos[body_id][2])
    except Exception:
        return default


def body_xy(base: Any, name: str):
    try:
        body_id = base.sim.model.body_name2id(name)
        return np.asarray(base.sim.data.body_xpos[body_id][:2], dtype=float)
    except Exception:
        return None


def apply_dynamics_noise(env: Any, scale: float) -> None:
    if scale <= 0:
        return
    base = unwrap_env(env)
    try:
        model = base.sim.model
        count = int(getattr(model, "na", 0) or getattr(model, "nu", 0))
        if count <= 0:
            return
        variance = np.random.uniform(1.0 - scale, 1.0 + scale, size=count)
        model.actuator_gainprm[:count, 0] *= variance
    except Exception:
        pass


def build_make_env(env_id: str, seed: int, noise_type: str = "none", noise_scale: float = 0.0):
    class NanSanitizerWrapper(gym.Wrapper):
        def reset(self, **kwargs):
            result = self.env.reset(**kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
            else:
                obs, info = result, {}
            return sanitize_value(obs), info

        def step(self, action):
            result = self.env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
            else:
                obs, reward, terminated, info = result
                truncated = False
            clean_obs = sanitize_value(obs)
            bad_reward = False
            try:
                bad_reward = bool(np.isnan(reward) or np.isinf(reward))
            except TypeError:
                pass
            if clean_obs is not obs:
                reward = -100.0
                terminated = True
                info = dict(info)
                info["error_flag"] = "obs_nan"
            if bad_reward:
                reward = -100.0
                terminated = True
                info = dict(info)
                info["error_flag"] = "reward_nan"
            return clean_obs, float(reward), bool(terminated), bool(truncated), info

    class NoiseInjectionWrapper(gym.Wrapper):
        def reset(self, **kwargs):
            result = self.env.reset(**kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
            else:
                obs, info = result, {}
            return self._obs_noise(obs), info

        def step(self, action):
            if noise_type == "action" and noise_scale > 0:
                action = np.clip(action + np.random.normal(0.0, noise_scale, size=np.asarray(action).shape), -1.0, 1.0)
            result = self.env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
            else:
                obs, reward, terminated, info = result
                truncated = False
            return self._obs_noise(obs), reward, terminated, truncated, info

        def _obs_noise(self, obs):
            if noise_type == "obs" and noise_scale > 0:
                return np.clip(np.asarray(obs) + np.random.normal(0.0, noise_scale, size=np.asarray(obs).shape), -20.0, 20.0)
            return obs

    def _init():
        env = gym.make(env_id)
        apply_dynamics_noise(env, noise_scale if noise_type == "dynamics" else 0.0)
        env.action_space.seed(seed)
        env = NoiseInjectionWrapper(env)
        env = NanSanitizerWrapper(env)
        return Monitor(env)

    return _init


def resolve_model_paths(args: argparse.Namespace, env_id: str) -> tuple[Path, Path | None, Path]:
    candidates: list[Path] = []
    if args.model_dir:
        model_dir = Path(args.model_dir)
        candidates.extend([model_dir, model_dir / "checkpoints", model_dir / env_slug(env_id) / "checkpoints"])
    root = Path(args.model_root)
    candidates.extend([root / env_slug(env_id) / "checkpoints", root / env_id / "checkpoints", root / env_slug(env_id)])
    for candidate in candidates:
        model_path = Path(args.model_path) if args.model_path else candidate / "best_model.zip"
        norm_path = Path(args.norm_path) if args.norm_path else candidate / "vec_normalize.pkl"
        if model_path.exists() and (norm_path.exists() or args.allow_missing_norm):
            return model_path, norm_path if norm_path.exists() else None, candidate
    raise FileNotFoundError(
        f"No {ALGORITHM_NAME} checkpoint found for {env_id}. Pass --model-dir, --model-root, --model-path and --norm-path."
    )


def load_model_and_env(args: argparse.Namespace, env_id: str, noise_type: str = "none", noise_scale: float = 0.0):
    model_path, norm_path, model_dir = resolve_model_paths(args, env_id)
    venv = DummyVecEnv([build_make_env(env_id, args.seed, noise_type, noise_scale)])
    if norm_path is not None:
        venv = VecNormalize.load(str(norm_path), venv)
        venv.training = False
        venv.norm_reward = False
    model = MODEL_CLASS.load(str(model_path), env=venv)
    return model, venv, model_dir


@dataclass
class EpisodeResult:
    env_id: str
    success: bool
    reward: float
    steps: int
    terminated: bool
    metric: float


def run_episode(model: Any, venv: Any, env_id: str, deterministic: bool, max_steps: int) -> EpisodeResult:
    obs = venv.reset()
    tracker = EpisodeTracker(env_id)
    tracker.on_reset(venv.envs[0])
    total_reward = 0.0
    steps = 0
    terminated = False
    while steps < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = venv.step(action)
        steps += 1
        reward = float(np.asarray(rewards).reshape(-1)[0])
        info = infos[0] if infos else {}
        total_reward += reward
        tracker.observe(venv.envs[0], info, steps)
        if tracker.success(steps, False):
            break
        if bool(dones[0]):
            terminated = True
            break
    return EpisodeResult(env_id, tracker.success(steps, terminated), total_reward, steps, terminated, tracker.summary_metric())


def summarize_results(results: list[EpisodeResult]) -> dict[str, Any]:
    successes = [r.success for r in results]
    rewards = [r.reward for r in results]
    steps = [r.steps for r in results]
    metrics = [r.metric for r in results]
    return {
        "episodes": len(results),
        "successes": int(sum(successes)),
        "success_rate": float(np.mean(successes) * 100.0) if results else 0.0,
        "avg_reward": float(np.mean(rewards)) if results else 0.0,
        "avg_steps": float(np.mean(steps)) if results else 0.0,
        "avg_metric": float(np.mean(metrics)) if results else 0.0,
    }


def evaluate_success(args: argparse.Namespace) -> list[dict[str, Any]]:
    load_runtime(Path(args.benchmark_root))
    rows = []
    for env_id in selected_envs(args.env):
        model, venv, model_dir = load_model_and_env(args, env_id)
        max_steps = max_steps_for(env_id, args.max_steps)
        results = [
            run_episode(model, venv, env_id, args.deterministic, max_steps)
            for _ in range(args.episodes)
        ]
        summary = summarize_results(results)
        row = {"algorithm": ALGORITHM_NAME, "env_id": env_id, "model_dir": str(model_dir), **summary}
        rows.append(row)
        print_success_row(row)
        venv.close()
    return rows


def evaluate_robustness(args: argparse.Namespace) -> list[dict[str, Any]]:
    load_runtime(Path(args.benchmark_root))
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
                model, venv, model_dir = load_model_and_env(args, env_id, noise_type, scale)
                results = [
                    run_episode(model, venv, env_id, args.deterministic, max_steps)
                    for _ in range(args.episodes)
                ]
                summary = summarize_results(results)
                row = {
                    "algorithm": ALGORITHM_NAME,
                    "env_id": env_id,
                    "noise_type": noise_type,
                    "noise_scale": scale,
                    "model_dir": str(model_dir),
                    **summary,
                }
                rows.append(row)
                print_robust_row(row)
                venv.close()
    return rows


def print_success_row(row: dict[str, Any]) -> None:
    print(
        f"{row['env_id']:<28} success={row['success_rate']:6.1f}% "
        f"reward={row['avg_reward']:10.1f} steps={row['avg_steps']:7.1f} "
        f"metric={row['avg_metric']:7.2f}"
    )


def print_robust_row(row: dict[str, Any]) -> None:
    print(
        f"{row['env_id']:<28} {row['noise_type']:<8} scale={row['noise_scale']:<5.3g} "
        f"success={row['success_rate']:6.1f}% reward={row['avg_reward']:10.1f} "
        f"steps={row['avg_steps']:7.1f}"
    )


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
    parser = argparse.ArgumentParser(description=f"{ALGORITHM_NAME} {EVAL_MODE} evaluation for MSK-Bench.")
    parser.add_argument("--env", choices=("all", *MSK_BENCH_ENVS), default="all")
    parser.add_argument("--list-envs", action="store_true")
    parser.add_argument("--episodes", "-n", type=int, default=50 if EVAL_MODE == "success" else 30)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--norm-path", type=Path, default=None)
    parser.add_argument("--allow-missing-norm", action="store_true")
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
