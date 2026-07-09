"""depRL robustness evaluation for MSK-Bench.

The script evaluates depRL/Tonic checkpoints. By default it searches for the
newest run under:
    D:/MSK-Bench/depRL/baselines_MSKBench/<tonic-name>/<timestamp>/checkpoints

You can also pass --run-path or --checkpoint-file explicitly.
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

ALGORITHM_NAME = "depRL"
EVAL_MODE = "robustness"
DEFAULT_DEPRL_ROOT = Path("D:/MSK-Bench/depRL")
DEFAULT_BENCHMARK_ROOT = Path("D:/MSK-Bench")
DEFAULT_MODEL_ROOT = Path("D:/MSK-Bench/depRL/baselines_MSKBench")
DEFAULT_CONFIG_DIR = Path("D:/MSK-Bench/depRL/experiments/msk_bench_training_files")
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
yaml = None
deprl = None
load_checkpoint = None
env_wrappers = None


def env_slug(env_id: str) -> str:
    name = env_id.removeprefix("MSKBench").removesuffix("-v0")
    return re.sub(r"(?<!^)([A-Z])", r"_\1", name).lower()


def selected_envs(env_arg: str) -> tuple[str, ...]:
    return MSK_BENCH_ENVS if env_arg == "all" else (env_arg,)


def parse_scales(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def max_steps_for(env_id: str, cli_max_steps: int | None) -> int:
    return cli_max_steps if cli_max_steps is not None else int(TASK_CONFIGS[env_id].get("steps", 1000))


def ensure_import_paths(args: argparse.Namespace) -> None:
    candidates = (
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
    from deprl import env_wrappers as wrappers_mod
    from deprl.utils import load_checkpoint as load_checkpoint_func

    np = np_mod
    yaml = yaml_mod
    deprl = deprl_mod
    env_wrappers = wrappers_mod
    load_checkpoint = load_checkpoint_func


def config_path_for(args: argparse.Namespace, env_id: str) -> Path:
    return Path(args.config_dir) / f"msk_bench_{env_slug(env_id)}.yaml"


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)


def latest_child(path: Path) -> Path | None:
    children = [child for child in path.iterdir() if child.is_dir()] if path.is_dir() else []
    if not children:
        return None
    return max(children, key=lambda item: item.stat().st_mtime)


def resolve_run_path(args: argparse.Namespace, env_id: str) -> Path:
    if args.run_path:
        return Path(args.run_path)
    config = read_yaml(config_path_for(args, env_id))
    tonic_name = config["tonic"]["name"]
    root = Path(args.model_root) / tonic_name
    latest = latest_child(root)
    if latest is not None and (latest / "config.yaml").exists():
        return latest
    raise FileNotFoundError(f"No depRL run found for {env_id}; pass --run-path or --model-root.")


def checkpoint_reference(args: argparse.Namespace, run_path: Path) -> tuple[dict[str, Any], Path]:
    if args.checkpoint_file:
        checkpoint_file = Path(args.checkpoint_file)
        run_path = Path(str(checkpoint_file).split("checkpoints")[0])
        checkpoint = checkpoint_file.stem.replace("step_", "")
    else:
        checkpoint = args.checkpoint
    config, checkpoint_path, _ = load_checkpoint(str(run_path / "checkpoints"), checkpoint)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No depRL checkpoint found in {run_path / 'checkpoints'}")
    return config, Path(checkpoint_path)


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

    @property
    def muscle_states(self):
        return self.env.muscle_states

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._obs_noise(obs)

    def step(self, action):
        if self.noise_type == "action" and self.noise_scale > 0:
            action = np.clip(action + np.random.normal(0.0, self.noise_scale, size=np.asarray(action).shape), -1.0, 1.0)
        obs, reward, done, info = self.env.step(action)
        return self._obs_noise(obs), reward, done, info

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


@dataclass
class EpisodeResult:
    env_id: str
    success: bool
    reward: float
    steps: int
    terminated: bool
    metric: float


def run_episode(agent: Any, env: Any, env_id: str, noisy: bool, max_steps: int) -> EpisodeResult:
    obs = env.reset()
    tracker = EpisodeTracker(env_id)
    tracker.on_reset(env)
    total_reward = 0.0
    steps = 0
    done = False
    while steps < max_steps:
        muscle_states = env.muscle_states
        if noisy and hasattr(agent, "noisy_test_step"):
            action = agent.noisy_test_step(obs, muscle_states=muscle_states, steps=1_000_000)
        else:
            action = agent.test_step(obs, muscle_states=muscle_states, steps=1_000_000)
        if len(np.asarray(action).shape) > 1:
            action = np.asarray(action)[0, :]
        obs, reward, done, info = env.step(action)
        steps += 1
        total_reward += float(reward)
        tracker.observe(env, info or {}, steps)
        if tracker.success(steps, False):
            break
        if done:
            break
    return EpisodeResult(env_id, tracker.success(steps, done), total_reward, steps, bool(done), tracker.summary_metric())


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
    parser = argparse.ArgumentParser(description=f"depRL {EVAL_MODE} evaluation for MSK-Bench.")
    parser.add_argument("--env", choices=("all", *MSK_BENCH_ENVS), default="all")
    parser.add_argument("--list-envs", action="store_true")
    parser.add_argument("--episodes", "-n", type=int, default=50 if EVAL_MODE == "success" else 30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noisy", action="store_true", help="Use depRL noisy_test_step when available.")
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
