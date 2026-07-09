"""msgym robustness evaluation for MSK-Bench.

The script evaluates DynSyn/msgym checkpoints from:
    <run-log>/checkpoint/best_model.zip
    <run-log>/checkpoint/best_env.zip

It can auto-select the newest run under D:/MSK-Bench/msgym/runs/msgym_logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore")

ALGORITHM_NAME = "msgym"
EVAL_MODE = "robustness"
DEFAULT_BENCHMARK_ROOT = Path("D:/MSK-Bench")
DEFAULT_MODEL_ROOT = Path("D:/MSK-Bench/msgym/runs/msgym_logs")
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
DummyVecEnv = None
VecNormalize = None
AgentRegistry = None
create_msgym_env = None


def ensure_import_paths(benchmark_root: Path) -> None:
    candidates = (
        benchmark_root / "msgym",
        benchmark_root / "msgym" / "SB3-Scripts",
        benchmark_root / "MSK-Bench",
        benchmark_root,
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent / "SB3-Scripts",
    )
    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_runtime(benchmark_root: Path) -> None:
    global np, gym, DummyVecEnv, VecNormalize, AgentRegistry, create_msgym_env
    ensure_import_paths(benchmark_root)
    import msk_bench  # noqa: F401
    import gymnasium as gym_mod
    import numpy as np_mod
    import sb3_contrib
    import stable_baselines3 as sb3
    from stable_baselines3.common.vec_env import DummyVecEnv as DummyVecEnvCls
    from stable_baselines3.common.vec_env import VecNormalize as VecNormalizeCls
    from DynSyn import SAC_DynSyn
    from utils import create_env as create_env_func

    np = np_mod
    gym = gym_mod
    DummyVecEnv = DummyVecEnvCls
    VecNormalize = VecNormalizeCls
    AgentRegistry = {"sb3": sb3, "sb3_contrib": sb3_contrib, "SAC_DynSyn": SAC_DynSyn}
    create_msgym_env = create_env_func


def env_slug(env_id: str) -> str:
    name = env_id.removeprefix("MSKBench").removesuffix("-v0")
    return re.sub(r"(?<!^)([A-Z])", r"_\1", name).lower()


def selected_envs(env_arg: str) -> tuple[str, ...]:
    return MSK_BENCH_ENVS if env_arg == "all" else (env_arg,)


def parse_scales(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def max_steps_for(env_id: str, cli_max_steps: int | None) -> int:
    return cli_max_steps if cli_max_steps is not None else int(TASK_CONFIGS[env_id].get("steps", 1000))


def latest_child(path: Path) -> Path | None:
    children = [child for child in path.iterdir() if child.is_dir()] if path.is_dir() else []
    if not children:
        return None
    return max(children, key=lambda item: item.stat().st_mtime)


def resolve_log_path(args: argparse.Namespace, env_id: str) -> Path:
    if args.log_path:
        return Path(args.log_path)
    root = Path(args.model_root)
    env_dir = root / env_id
    latest = latest_child(env_dir)
    if latest is not None and (latest / "checkpoint" / "best_model.zip").exists():
        return latest
    matches = sorted(root.glob(f"**/*{env_slug(env_id)}*.json"))
    for config_file in reversed(matches):
        log_dir = config_file.parent
        if (log_dir / "checkpoint" / "best_model.zip").exists():
            return log_dir
    raise FileNotFoundError(f"No msgym log/checkpoint found for {env_id}; pass --log-path or --model-root.")


def load_config(log_path: Path) -> dict[str, Any]:
    json_files = sorted(log_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No config json found in {log_path}")
    return json.loads(json_files[0].read_text(encoding="utf-8"))


def agent_class(name: str):
    if hasattr(AgentRegistry["sb3"], name):
        return getattr(AgentRegistry["sb3"], name)
    if hasattr(AgentRegistry["sb3_contrib"], name):
        return getattr(AgentRegistry["sb3_contrib"], name)
    if name == "SAC_DynSyn":
        return AgentRegistry["SAC_DynSyn"]
    raise ValueError(f"Unknown msgym agent: {name}")


class NoiseWrapper:
    def __init__(self, env, noise_type: str, noise_scale: float):
        self.env = env
        self.noise_type = noise_type
        self.noise_scale = noise_scale

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        return self._obs_noise(obs), info

    def step(self, action):
        if self.noise_type == "action" and self.noise_scale > 0:
            action = np.clip(action + np.random.normal(0.0, self.noise_scale, size=np.asarray(action).shape), -1.0, 1.0)
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, terminated, info = result
            truncated = False
        return self._obs_noise(obs), reward, terminated, truncated, info

    def _obs_noise(self, obs):
        if self.noise_type == "obs" and self.noise_scale > 0:
            return np.clip(np.asarray(obs) + np.random.normal(0.0, self.noise_scale, size=np.asarray(obs).shape), -20.0, 20.0)
        return obs


def unwrap_env(env: Any) -> Any:
    current = env
    seen = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    return current.unwrapped if hasattr(current, "unwrapped") else current


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(np.asarray(value).reshape(-1)[0])
    except Exception:
        return default


def bool_from_info(info: dict[str, Any], key: str) -> bool:
    for candidate in (key, f"rwd_{key}"):
        if candidate in info and float_value(info[candidate]) > 0.5:
            return True
    rwd_dict = info.get("rwd_dict")
    return isinstance(rwd_dict, dict) and key in rwd_dict and float_value(rwd_dict[key]) > 0.5


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


class EpisodeTracker:
    def __init__(self, env_id: str):
        self.env_id = env_id
        self.config = TASK_CONFIGS[env_id]
        self.kind = self.config["kind"]
        self.reset()

    def reset(self) -> None:
        self.start_x = 0.0
        self.start_y = 0.0
        self.max_x = -float("inf")
        self.min_x = float("inf")
        self.max_abs_y_delta = 0.0
        self.max_z = -float("inf")
        self.max_bar_z = -float("inf")
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
            pass
        self.observe(base, {}, 0)

    def observe(self, env: Any, info: dict[str, Any], step: int) -> None:
        base = unwrap_env(env)
        self.solved_seen = self.solved_seen or bool_from_info(info, "solved")
        try:
            qpos = np.asarray(base.sim.data.qpos)
            x, y, z = float(qpos[0]), float(qpos[1]), float(qpos[2])
            self.max_x = max(self.max_x, x)
            self.min_x = min(self.min_x, x)
            self.max_abs_y_delta = max(self.max_abs_y_delta, abs(y - self.start_y))
            self.max_z = max(self.max_z, z)
            if z > 1.00 and not self.airborne:
                self.airborne = True
            if z < 0.95 and self.airborne:
                self.hops += 1
                self.airborne = False
            if self.squat_state == 0 and z < float(self.config.get("depth", 0.75)):
                self.squat_state = 1
            if self.squat_state == 1 and z > float(self.config.get("stand", 0.90)):
                self.squat_state = 2
        except Exception:
            pass
        self.max_bar_z = max(self.max_bar_z, body_z(base, "dumbbell", -float("inf")))
        if self.kind == "balance":
            board = body_xy(base, "board")
            ball = body_xy(base, "ball")
            if board is not None and ball is not None:
                self.best_board_drift = min(self.best_board_drift, float(np.linalg.norm(board - ball)))
        self.max_hurdles = max(self.max_hurdles, int(getattr(base, "hurdles_cleared", 0) or 0))
        if getattr(base, "hurdles", None) is not None:
            self.total_hurdles = len(base.hurdles)
        self.max_stone = max(self.max_stone, int(getattr(base, "current_stone_idx", 0) or 0))
        if getattr(base, "stones", None) is not None:
            self.total_stones = len(base.stones)
        self.max_target_idx = max(self.max_target_idx, int(getattr(base, "current_target_idx", 0) or 0))
        self.max_touches = max(self.max_touches, int(getattr(base, "touch_count", 0) or 0), int(float_value(info.get("rwd_sparse"), 0.0)))
        self.has_sat_down = self.has_sat_down or bool(getattr(base, "has_sat_down", False))
        self.door_opened = self.door_opened or bool(getattr(base, "door_opened_flag", False)) or bool_from_info(info, "open_door")
        self.max_door_angle = max(self.max_door_angle, float(getattr(base, "max_door_angle", 0.0) or 0.0))
        if self.kind == "chinup":
            head_z = body_z(base, "head", -float("inf"))
            bar_height = float(getattr(base, "bar_height", 2.3))
            if head_z > bar_height + 0.1:
                self.current_chinup_hold += 1
                self.max_chinup_hold = max(self.max_chinup_hold, self.current_chinup_hold)
            else:
                self.current_chinup_hold = 0
        self.caught = self.caught or bool_from_info(info, "catch_bonus")

    def success(self, steps: int, terminated: bool) -> bool:
        if self.solved_seen:
            return True
        dx = self.max_x - self.start_x
        back = self.start_x - self.min_x
        kind = self.kind
        if kind in {"stand", "survival", "walk"}:
            return steps >= int(self.config.get("steps", 1000)) and not terminated
        if kind == "powerlift":
            return steps >= int(self.config.get("steps", 900)) and self.max_bar_z >= float(self.config.get("height", 1.6))
        if kind == "balance":
            return steps > 350 and self.best_board_drift < 0.5
        if kind == "squat":
            return self.squat_state >= 2
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
            return self.max_hurdles >= max(self.total_hurdles or 5, 1)
        if kind == "stones":
            return self.max_stone >= max((self.total_stones or 1) - 1, 0)
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
            return self.has_sat_down
        return False

    def summary_metric(self) -> float:
        if self.kind == "crawl":
            return self.start_x - self.min_x
        if self.kind == "sidestep":
            return self.max_abs_y_delta
        if self.kind == "jump":
            return float(self.hops)
        if self.kind == "powerlift":
            return self.max_bar_z
        return self.max_x - self.start_x


def apply_dynamics_noise(env: Any, scale: float) -> None:
    if scale <= 0:
        return
    base = unwrap_env(env)
    try:
        model = base.sim.model
        count = int(getattr(model, "na", 0) or getattr(model, "nu", 0))
        variance = np.random.uniform(1.0 - scale, 1.0 + scale, size=count)
        model.actuator_gainprm[:count, 0] *= variance
    except Exception:
        pass


def build_env_from_config(config: dict[str, Any], noise_type: str = "none", noise_scale: float = 0.0):
    def _init():
        env = create_msgym_env(
            config["env_name"],
            config.get("single_env_kwargs", {}),
            config.get("wrapper_list", {}),
            render_mode=None,
        )
        apply_dynamics_noise(env, noise_scale if noise_type == "dynamics" else 0.0)
        return NoiseWrapper(env, noise_type, noise_scale)
    return _init


def load_model_and_env(args: argparse.Namespace, env_id: str, noise_type: str = "none", noise_scale: float = 0.0):
    log_path = resolve_log_path(args, env_id)
    config = load_config(log_path)
    checkpoint = log_path / "checkpoint"
    model_path = Path(args.model_path) if args.model_path else checkpoint / "best_model.zip"
    env_path = Path(args.norm_path) if args.norm_path else checkpoint / "best_env.zip"
    if not model_path.exists() or not env_path.exists():
        raise FileNotFoundError(f"Missing msgym model/env files under {checkpoint}")
    vec_env = DummyVecEnv([build_env_from_config(config, noise_type, noise_scale)])
    vec_norm = VecNormalize.load(str(env_path), vec_env)
    vec_norm.training = False
    vec_norm.norm_reward = False
    Agent = agent_class(config["agent"])
    model = Agent.load(str(model_path), env=vec_norm)
    return model, vec_norm, log_path


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


def summarize(results: list[EpisodeResult]) -> dict[str, Any]:
    return {
        "episodes": len(results),
        "successes": int(sum(item.success for item in results)),
        "success_rate": float(np.mean([item.success for item in results]) * 100.0) if results else 0.0,
        "avg_reward": float(np.mean([item.reward for item in results])) if results else 0.0,
        "avg_steps": float(np.mean([item.steps for item in results])) if results else 0.0,
        "avg_metric": float(np.mean([item.metric for item in results])) if results else 0.0,
    }


def print_success_row(row: dict[str, Any]) -> None:
    print(f"{row['env_id']:<28} success={row['success_rate']:6.1f}% reward={row['avg_reward']:10.1f} steps={row['avg_steps']:7.1f} metric={row['avg_metric']:7.2f}")


def print_robust_row(row: dict[str, Any]) -> None:
    print(f"{row['env_id']:<28} {row['noise_type']:<8} scale={row['noise_scale']:<5.3g} success={row['success_rate']:6.1f}% reward={row['avg_reward']:10.1f} steps={row['avg_steps']:7.1f}")


def evaluate_success(args: argparse.Namespace) -> list[dict[str, Any]]:
    load_runtime(Path(args.benchmark_root))
    rows = []
    for env_id in selected_envs(args.env):
        model, venv, log_path = load_model_and_env(args, env_id)
        results = [run_episode(model, venv, env_id, args.deterministic, max_steps_for(env_id, args.max_steps)) for _ in range(args.episodes)]
        row = {"algorithm": ALGORITHM_NAME, "env_id": env_id, "log_path": str(log_path), **summarize(results)}
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
                model, venv, log_path = load_model_and_env(args, env_id, noise_type, scale)
                results = [run_episode(model, venv, env_id, args.deterministic, max_steps) for _ in range(args.episodes)]
                row = {"algorithm": ALGORITHM_NAME, "env_id": env_id, "noise_type": noise_type, "noise_scale": scale, "log_path": str(log_path), **summarize(results)}
                rows.append(row)
                print_robust_row(row)
                venv.close()
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
    parser = argparse.ArgumentParser(description=f"msgym {EVAL_MODE} evaluation for MSK-Bench.")
    parser.add_argument("--env", choices=("all", *MSK_BENCH_ENVS), default="all")
    parser.add_argument("--list-envs", action="store_true")
    parser.add_argument("--episodes", "-n", type=int, default=50 if EVAL_MODE == "success" else 30)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--norm-path", type=Path, default=None)
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

