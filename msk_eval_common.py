"""Shared smoothness, energy, and rendering helpers for MSK-Bench evaluators."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Callable


def configure_headless_rendering() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _np():
    import numpy as np

    return np


def as_array(value: Any, *, default_shape: tuple[int, ...] = (0,)):
    np = _np()
    if value is None:
        return np.zeros(default_shape, dtype=float)
    try:
        return np.asarray(value, dtype=float).copy()
    except Exception:
        return np.zeros(default_shape, dtype=float)


def unwrap_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    if hasattr(current, "unwrapped"):
        try:
            return current.unwrapped
        except Exception:
            return current
    return current


def env_dt(env: Any, default: float = 0.01) -> float:
    base = unwrap_env(env)
    for candidate in (base, getattr(base, "sim", None)):
        if candidate is None:
            continue
        value = getattr(candidate, "dt", None)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
    try:
        timestep = float(base.sim.model.opt.timestep)
        frame_skip = float(getattr(base, "frame_skip", 1))
        return timestep * frame_skip
    except Exception:
        return default


def physics_data(env: Any) -> Any | None:
    base = unwrap_env(env)
    sim = getattr(base, "sim", None)
    if sim is not None and hasattr(sim, "data"):
        return sim.data
    return getattr(base, "data", None)


def extract_physics_snapshot(env: Any, action: Any) -> dict[str, Any]:
    data = physics_data(env)
    fallback_action = as_array(action)
    if data is None:
        return {
            "qvel": as_array(None),
            "torque": as_array(None),
            "activation": fallback_action,
        }
    qvel = as_array(getattr(data, "qvel", None))
    torque = as_array(getattr(data, "qfrc_actuator", None), default_shape=qvel.shape)
    activation = getattr(data, "act", None)
    activation_array = as_array(activation) if activation is not None else fallback_action
    return {"qvel": qvel, "torque": torque, "activation": activation_array}


def evaluate_smoothness(trajectory: Any, dt: float = 0.01) -> dict[str, float]:
    np = _np()
    data = np.asarray(trajectory, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[0] < 4 or data.shape[1] == 0:
        return {
            "dims": int(data.shape[1]) if data.ndim == 2 else 0,
            "rate_energy": 0.0,
            "accel_energy": 0.0,
            "jerk_energy": 0.0,
            "max_jerk_peak": 0.0,
            "mean_absolute_jerk": 0.0,
        }
    safe_dt = max(float(dt), 1e-9)
    vel = np.diff(data, n=1, axis=0) / safe_dt
    accel = np.diff(data, n=2, axis=0) / (safe_dt**2)
    jerk = np.diff(data, n=3, axis=0) / (safe_dt**3)
    return {
        "dims": int(data.shape[1]),
        "rate_energy": float(np.mean(np.square(vel))),
        "accel_energy": float(np.mean(np.square(accel))),
        "jerk_energy": float(np.mean(np.square(jerk))),
        "max_jerk_peak": float(np.max(np.abs(jerk))),
        "mean_absolute_jerk": float(np.mean(np.abs(jerk))),
    }


def summarize_energy_episode(actions: Any, torques: Any, activations: Any, qvels: Any, dt: float) -> dict[str, float]:
    np = _np()
    action_arr = np.asarray(actions, dtype=float)
    torque_arr = np.asarray(torques, dtype=float)
    activation_arr = np.asarray(activations, dtype=float)
    qvel_arr = np.asarray(qvels, dtype=float)
    safe_dt = max(float(dt), 1e-9)

    if action_arr.size == 0:
        action_arr = np.zeros((0, 1), dtype=float)
    if torque_arr.size == 0:
        torque_arr = np.zeros((0, 1), dtype=float)
    if activation_arr.size == 0:
        activation_arr = action_arr

    mechanical_energy = 0.0
    if torque_arr.shape == qvel_arr.shape and torque_arr.size:
        mechanical_power = np.sum(np.abs(torque_arr * qvel_arr), axis=1)
        mechanical_energy = float(np.sum(mechanical_power) * safe_dt)

    return {
        "joint_torque_abs_sum": float(np.sum(np.abs(torque_arr))),
        "joint_torque_squared_energy": float(np.sum(np.square(torque_arr)) * safe_dt),
        "mechanical_energy": mechanical_energy,
        "muscle_effort_mean": float(np.mean(np.square(activation_arr))) if activation_arr.size else 0.0,
        "muscle_activation_abs_sum": float(np.sum(np.abs(activation_arr))),
        "action_l2_mean": float(np.mean(np.square(action_arr))) if action_arr.size else 0.0,
    }


def write_rows(rows: list[dict[str, Any]], json_path: str | Path | None, csv_path: str | Path | None) -> None:
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


def _row_mean(rows: list[dict[str, Any]], key: str) -> float:
    np = _np()
    values = [float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float))]
    return float(np.mean(values)) if values else 0.0


def print_smooth_summary(algorithm: str, env_id: str, rows: list[dict[str, Any]]) -> None:
    print(
        f"{env_id:<28} {algorithm:<6} "
        f"action_jerk={_row_mean(rows, 'action_jerk_energy'):12.3f} "
        f"joint_jerk={_row_mean(rows, 'joint_jerk_energy'):12.3f} "
        f"steps={_row_mean(rows, 'steps'):7.1f}"
    )


def print_energy_summary(algorithm: str, env_id: str, rows: list[dict[str, Any]]) -> None:
    print(
        f"{env_id:<28} {algorithm:<6} "
        f"torque={_row_mean(rows, 'joint_torque_abs_sum'):12.3f} "
        f"effort={_row_mean(rows, 'muscle_effort_mean'):10.5f} "
        f"mech={_row_mean(rows, 'mechanical_energy'):12.3f}"
    )


def _episode_result_row(
    algorithm: str,
    env_id: str,
    artifact_key: str,
    artifact_path: Path,
    episode: int,
    steps: int,
    reward: float,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "algorithm": algorithm,
        "env_id": env_id,
        "episode": episode,
        "steps": steps,
        "reward": reward,
        artifact_key: str(artifact_path),
        **metrics,
    }


def _stack(values: list[Any]):
    np = _np()
    if not values:
        return np.zeros((0, 1), dtype=float)
    try:
        return np.asarray(values, dtype=float)
    except ValueError:
        max_len = max(np.asarray(item).size for item in values)
        padded = np.zeros((len(values), max_len), dtype=float)
        for index, item in enumerate(values):
            flat = np.asarray(item, dtype=float).reshape(-1)
            padded[index, : flat.size] = flat
        return padded


def collect_sb3_episode(model: Any, venv: Any, deterministic: bool, max_steps: int) -> dict[str, Any]:
    np = _np()
    obs = venv.reset()
    env = venv.envs[0]
    dt = env_dt(env)
    actions: list[Any] = []
    torques: list[Any] = []
    activations: list[Any] = []
    qvels: list[Any] = []
    total_reward = 0.0
    steps = 0
    done = False
    while steps < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        action_one = np.asarray(action)[0]
        obs, rewards, dones, infos = venv.step(action)
        snapshot = extract_physics_snapshot(env, action_one)
        actions.append(action_one.copy())
        qvels.append(snapshot["qvel"])
        torques.append(snapshot["torque"])
        activations.append(snapshot["activation"])
        total_reward += float(np.asarray(rewards).reshape(-1)[0])
        steps += 1
        done = bool(np.asarray(dones).reshape(-1)[0])
        if done:
            break
    return {
        "actions": _stack(actions),
        "qvels": _stack(qvels),
        "torques": _stack(torques),
        "activations": _stack(activations),
        "reward": total_reward,
        "steps": steps,
        "done": done,
        "dt": dt,
    }


def collect_deprl_episode(agent: Any, env: Any, noisy: bool, max_steps: int) -> dict[str, Any]:
    np = _np()
    obs = env.reset()
    dt = env_dt(env)
    actions: list[Any] = []
    torques: list[Any] = []
    activations: list[Any] = []
    qvels: list[Any] = []
    total_reward = 0.0
    steps = 0
    done = False
    while steps < max_steps:
        muscle_states = getattr(env, "muscle_states", None)
        if noisy and hasattr(agent, "noisy_test_step"):
            action = agent.noisy_test_step(obs, muscle_states=muscle_states, steps=1_000_000)
        else:
            action = agent.test_step(obs, muscle_states=muscle_states, steps=1_000_000)
        if len(np.asarray(action).shape) > 1:
            action = np.asarray(action)[0, :]
        obs, reward, done, info = env.step(action)
        snapshot = extract_physics_snapshot(env, action)
        actions.append(np.asarray(action).copy())
        qvels.append(snapshot["qvel"])
        torques.append(snapshot["torque"])
        activations.append(snapshot["activation"])
        total_reward += float(reward)
        steps += 1
        if done:
            break
    return {
        "actions": _stack(actions),
        "qvels": _stack(qvels),
        "torques": _stack(torques),
        "activations": _stack(activations),
        "reward": total_reward,
        "steps": steps,
        "done": bool(done),
        "dt": dt,
    }


def _smooth_metrics(episode: dict[str, Any]) -> dict[str, float]:
    action = evaluate_smoothness(episode["actions"], episode["dt"])
    joint = evaluate_smoothness(episode["qvels"], episode["dt"])
    return {
        "action_dims": action["dims"],
        "action_rate_energy": action["rate_energy"],
        "action_accel_energy": action["accel_energy"],
        "action_jerk_energy": action["jerk_energy"],
        "action_max_jerk_peak": action["max_jerk_peak"],
        "action_mean_absolute_jerk": action["mean_absolute_jerk"],
        "joint_dims": joint["dims"],
        "joint_rate_energy": joint["rate_energy"],
        "joint_accel_energy": joint["accel_energy"],
        "joint_jerk_energy": joint["jerk_energy"],
        "joint_max_jerk_peak": joint["max_jerk_peak"],
        "joint_mean_absolute_jerk": joint["mean_absolute_jerk"],
    }


def _energy_metrics(episode: dict[str, Any]) -> dict[str, float]:
    return summarize_energy_episode(
        episode["actions"],
        episode["torques"],
        episode["activations"],
        episode["qvels"],
        episode["dt"],
    )


def _model_artifact_key(base_module: Any) -> str:
    name = str(getattr(base_module, "ALGORITHM_NAME", "")).lower()
    if name == "deprl":
        return "run_path"
    if name == "msgym":
        return "log_path"
    return "model_dir"


def _prepare_eval_parser(base_module: Any, description: str, default_episodes: int):
    parser = base_module.build_parser()
    parser.description = description
    parser.set_defaults(episodes=default_episodes)
    if hasattr(base_module, "MODEL_CLASS") or str(getattr(base_module, "ALGORITHM_NAME", "")).lower() in {"sac", "ppo", "msgym"}:
        parser.set_defaults(deterministic=True)
    return parser


def main_sb3_smooth(base_module: Any, argv: list[str] | None = None) -> int:
    configure_headless_rendering()
    algorithm = base_module.ALGORITHM_NAME
    parser = _prepare_eval_parser(base_module, f"{algorithm} smoothness evaluation for 22 MSK-Bench tasks.", 5)
    args = parser.parse_args(argv)
    args.algorithm = algorithm
    if args.list_envs:
        for env_id in base_module.MSK_BENCH_ENVS:
            print(env_id)
        return 0
    base_module.load_runtime(Path(args.benchmark_root))
    rows: list[dict[str, Any]] = []
    artifact_key = _model_artifact_key(base_module)
    for env_id in base_module.selected_envs(args.env):
        model, venv, artifact_path = base_module.load_model_and_env(args, env_id)
        env_rows: list[dict[str, Any]] = []
        max_steps = base_module.max_steps_for(env_id, args.max_steps)
        for episode_index in range(args.episodes):
            episode = collect_sb3_episode(model, venv, bool(args.deterministic), max_steps)
            row = _episode_result_row(
                algorithm,
                env_id,
                artifact_key,
                Path(artifact_path),
                episode_index + 1,
                int(episode["steps"]),
                float(episode["reward"]),
                _smooth_metrics(episode),
            )
            rows.append(row)
            env_rows.append(row)
        print_smooth_summary(algorithm, env_id, env_rows)
        venv.close()
    write_rows(rows, args.json, args.csv)
    return 0


def main_sb3_energy(base_module: Any, argv: list[str] | None = None) -> int:
    configure_headless_rendering()
    algorithm = base_module.ALGORITHM_NAME
    parser = _prepare_eval_parser(base_module, f"{algorithm} energy evaluation for 22 MSK-Bench tasks.", 10)
    args = parser.parse_args(argv)
    args.algorithm = algorithm
    if args.list_envs:
        for env_id in base_module.MSK_BENCH_ENVS:
            print(env_id)
        return 0
    base_module.load_runtime(Path(args.benchmark_root))
    rows: list[dict[str, Any]] = []
    artifact_key = _model_artifact_key(base_module)
    for env_id in base_module.selected_envs(args.env):
        model, venv, artifact_path = base_module.load_model_and_env(args, env_id)
        env_rows: list[dict[str, Any]] = []
        max_steps = base_module.max_steps_for(env_id, args.max_steps)
        for episode_index in range(args.episodes):
            episode = collect_sb3_episode(model, venv, bool(args.deterministic), max_steps)
            row = _episode_result_row(
                algorithm,
                env_id,
                artifact_key,
                Path(artifact_path),
                episode_index + 1,
                int(episode["steps"]),
                float(episode["reward"]),
                _energy_metrics(episode),
            )
            rows.append(row)
            env_rows.append(row)
        print_energy_summary(algorithm, env_id, env_rows)
        venv.close()
    write_rows(rows, args.json, args.csv)
    return 0


def main_deprl_smooth(base_module: Any, argv: list[str] | None = None) -> int:
    configure_headless_rendering()
    algorithm = base_module.ALGORITHM_NAME
    parser = _prepare_eval_parser(base_module, f"{algorithm} smoothness evaluation for 22 MSK-Bench tasks.", 5)
    args = parser.parse_args(argv)
    args.algorithm = algorithm
    if args.list_envs:
        for env_id in base_module.MSK_BENCH_ENVS:
            print(env_id)
        return 0
    base_module.load_runtime(args)
    rows: list[dict[str, Any]] = []
    for env_id in base_module.selected_envs(args.env):
        agent, env, run_path = base_module.build_agent_env(args, env_id)
        env_rows: list[dict[str, Any]] = []
        max_steps = base_module.max_steps_for(env_id, args.max_steps)
        for episode_index in range(args.episodes):
            episode = collect_deprl_episode(agent, env, bool(args.noisy), max_steps)
            row = _episode_result_row(
                algorithm,
                env_id,
                "run_path",
                Path(run_path),
                episode_index + 1,
                int(episode["steps"]),
                float(episode["reward"]),
                _smooth_metrics(episode),
            )
            rows.append(row)
            env_rows.append(row)
        print_smooth_summary(algorithm, env_id, env_rows)
        if hasattr(env, "close"):
            env.close()
    write_rows(rows, args.json, args.csv)
    return 0


def main_deprl_energy(base_module: Any, argv: list[str] | None = None) -> int:
    configure_headless_rendering()
    algorithm = base_module.ALGORITHM_NAME
    parser = _prepare_eval_parser(base_module, f"{algorithm} energy evaluation for 22 MSK-Bench tasks.", 10)
    args = parser.parse_args(argv)
    args.algorithm = algorithm
    if args.list_envs:
        for env_id in base_module.MSK_BENCH_ENVS:
            print(env_id)
        return 0
    base_module.load_runtime(args)
    rows: list[dict[str, Any]] = []
    for env_id in base_module.selected_envs(args.env):
        agent, env, run_path = base_module.build_agent_env(args, env_id)
        env_rows: list[dict[str, Any]] = []
        max_steps = base_module.max_steps_for(env_id, args.max_steps)
        for episode_index in range(args.episodes):
            episode = collect_deprl_episode(agent, env, bool(args.noisy), max_steps)
            row = _episode_result_row(
                algorithm,
                env_id,
                "run_path",
                Path(run_path),
                episode_index + 1,
                int(episode["steps"]),
                float(episode["reward"]),
                _energy_metrics(episode),
            )
            rows.append(row)
            env_rows.append(row)
        print_energy_summary(algorithm, env_id, env_rows)
        if hasattr(env, "close"):
            env.close()
    write_rows(rows, args.json, args.csv)
    return 0


def render_rgb_frame(env: Any, width: int = 640, height: int = 480, camera_id: int | None = None):
    np = _np()
    base = unwrap_env(env)
    render_calls: list[Callable[[], Any]] = []
    render_method = getattr(base, "render", None)
    if callable(render_method):
        render_calls.extend(
            [
                lambda: render_method(),
                lambda: render_method(mode="rgb_array"),
                lambda: render_method(width=width, height=height, camera_id=camera_id),
            ]
        )
    sim = getattr(base, "sim", None)
    if sim is not None:
        sim_render = getattr(sim, "render", None)
        if callable(sim_render):
            render_calls.extend(
                [
                    lambda: sim_render(width=width, height=height, camera_id=camera_id),
                    lambda: sim_render(height, width, camera_id=camera_id),
                ]
            )
        renderer = getattr(sim, "renderer", None)
        if renderer is not None:
            render_calls.append(lambda: _render_with_msk_renderer(sim, width, height, camera_id))
    for call in render_calls:
        try:
            frame = call()
        except Exception:
            continue
        if frame is None:
            continue
        array = np.asarray(frame)
        if array.ndim == 3 and array.shape[2] >= 3:
            return array[:, :, :3].astype("uint8")
    raise RuntimeError("Could not render an RGB frame from this environment.")


def _render_with_msk_renderer(sim: Any, width: int, height: int, camera_id: int | None):
    renderer = sim.renderer
    if getattr(renderer, "_renderer", None) is None and hasattr(renderer, "setup_renderer"):
        renderer.setup_renderer(sim.model.ptr, height=height, width=width)
    real_renderer = getattr(renderer, "_renderer", None)
    if real_renderer is None:
        raise RuntimeError("MSK renderer is not initialized.")
    scene_option = getattr(renderer, "_scene_option", None)
    real_renderer.update_scene(sim.data.ptr, camera=camera_id, scene_option=scene_option)
    return real_renderer.render()


class MetricPanel:
    def __init__(self, height: int, width: int, dt: float):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        self.plt = plt
        self.dt = dt
        self.steps: list[int] = []
        self.efforts: list[float] = []
        self.torque_energy: list[float] = []
        self.total_torque = 0.0
        dpi = 100
        self.fig, self.axes = plt.subplots(2, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
        self.canvas = FigureCanvasAgg(self.fig)
        self.fig.subplots_adjust(left=0.23, right=0.95, top=0.9, bottom=0.12, hspace=0.35)

    def update(self, step: int, activation: Any, torque: Any) -> None:
        np = _np()
        effort = float(np.mean(np.square(np.asarray(activation, dtype=float)))) if np.asarray(activation).size else 0.0
        torque_step = float(np.sum(np.abs(np.asarray(torque, dtype=float))) * self.dt) if np.asarray(torque).size else 0.0
        self.total_torque += torque_step
        self.steps.append(step)
        self.efforts.append(effort)
        self.torque_energy.append(self.total_torque)

    def frame(self):
        np = _np()
        ax1, ax2 = self.axes
        ax1.clear()
        ax2.clear()
        if self.steps:
            ax1.plot(self.steps, self.efforts, color="#1f77b4", linewidth=1.2)
            ax1.fill_between(self.steps, self.efforts, color="#1f77b4", alpha=0.12)
            ax2.plot(self.steps, self.torque_energy, color="#d62728", linewidth=1.2)
            ax2.fill_between(self.steps, self.torque_energy, color="#d62728", alpha=0.12)
        ax1.set_title(f"Muscle effort: {self.efforts[-1]:.4f}" if self.efforts else "Muscle effort", fontsize=10)
        ax1.set_ylabel("act^2", fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax2.set_title(f"Torque integral: {self.total_torque:.2f}", fontsize=10)
        ax2.set_xlabel("Steps", fontsize=8)
        ax2.set_ylabel("sum |tau| dt", fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.5)
        self.canvas.draw()
        image = np.asarray(self.canvas.buffer_rgba())
        return image[:, :, :3].astype("uint8")

    def close(self) -> None:
        self.plt.close(self.fig)


def save_video(frames: list[Any], path: Path, fps: int) -> Path:
    if not frames:
        raise RuntimeError("No frames were rendered.")
    import imageio

    np = _np()
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_frames = [np.asarray(frame).astype("uint8") for frame in frames]
    try:
        imageio.mimsave(path, clean_frames, fps=fps, macro_block_size=1)
        return path
    except Exception:
        fallback = path.with_suffix(".gif")
        imageio.mimsave(fallback, clean_frames, fps=fps)
        return fallback


def _prepare_render_parser(base_module: Any, description: str):
    parser = _prepare_eval_parser(base_module, description, 1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--plot-width", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-id", type=int, default=None)
    parser.add_argument("--with-plots", action="store_true", help="Append effort/torque plots to the video.")
    return parser


def _video_output_dir(args: Any, artifact_path: Path) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    return Path(artifact_path) / "renders"


def _video_name(algorithm: str, env_id: str, episode_index: int) -> str:
    slug = env_id.removeprefix("MSKBench").removesuffix("-v0")
    return f"{algorithm.lower()}_{slug}_episode{episode_index}.mp4"


def main_sb3_render(base_module: Any, argv: list[str] | None = None) -> int:
    configure_headless_rendering()
    algorithm = base_module.ALGORITHM_NAME
    parser = _prepare_render_parser(base_module, f"{algorithm} rendering for 22 MSK-Bench tasks.")
    args = parser.parse_args(argv)
    args.algorithm = algorithm
    if args.list_envs:
        for env_id in base_module.MSK_BENCH_ENVS:
            print(env_id)
        return 0
    base_module.load_runtime(Path(args.benchmark_root))
    rows: list[dict[str, Any]] = []
    for env_id in base_module.selected_envs(args.env):
        model, venv, artifact_path = base_module.load_model_and_env(args, env_id)
        max_steps = base_module.max_steps_for(env_id, args.max_steps)
        for episode_index in range(1, args.episodes + 1):
            video_path = render_sb3_episode(model, venv, env_id, args, artifact_path, max_steps, episode_index)
            rows.append({"algorithm": algorithm, "env_id": env_id, "episode": episode_index, "video": str(video_path)})
            print(f"{env_id:<28} video={video_path}")
        venv.close()
    write_rows(rows, args.json, args.csv)
    return 0


def render_sb3_episode(model: Any, venv: Any, env_id: str, args: Any, artifact_path: Path, max_steps: int, episode_index: int) -> Path:
    np = _np()
    obs = venv.reset()
    env = venv.envs[0]
    dt = env_dt(env)
    panel = MetricPanel(args.height, args.plot_width, dt) if args.with_plots else None
    frames: list[Any] = []
    try:
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=bool(getattr(args, "deterministic", True)))
            obs, rewards, dones, infos = venv.step(action)
            frame = render_rgb_frame(env, args.width, args.height, args.camera_id)
            if panel is not None:
                snapshot = extract_physics_snapshot(env, np.asarray(action)[0])
                panel.update(step, snapshot["activation"], snapshot["torque"])
                frame = np.concatenate((frame, panel.frame()), axis=1)
            frames.append(frame)
            if bool(np.asarray(dones).reshape(-1)[0]):
                break
    finally:
        if panel is not None:
            panel.close()
    out_dir = _video_output_dir(args, Path(artifact_path))
    return save_video(frames, out_dir / _video_name(str(getattr(args, "algorithm", "")) or "policy", env_id, episode_index), args.fps)


def main_deprl_render(base_module: Any, argv: list[str] | None = None) -> int:
    configure_headless_rendering()
    algorithm = base_module.ALGORITHM_NAME
    parser = _prepare_render_parser(base_module, f"{algorithm} rendering for 22 MSK-Bench tasks.")
    args = parser.parse_args(argv)
    args.algorithm = algorithm
    if args.list_envs:
        for env_id in base_module.MSK_BENCH_ENVS:
            print(env_id)
        return 0
    base_module.load_runtime(args)
    rows: list[dict[str, Any]] = []
    for env_id in base_module.selected_envs(args.env):
        agent, env, run_path = base_module.build_agent_env(args, env_id)
        max_steps = base_module.max_steps_for(env_id, args.max_steps)
        for episode_index in range(1, args.episodes + 1):
            video_path = render_deprl_episode(agent, env, env_id, args, run_path, max_steps, episode_index)
            rows.append({"algorithm": algorithm, "env_id": env_id, "episode": episode_index, "video": str(video_path)})
            print(f"{env_id:<28} video={video_path}")
        if hasattr(env, "close"):
            env.close()
    write_rows(rows, args.json, args.csv)
    return 0


def render_deprl_episode(agent: Any, env: Any, env_id: str, args: Any, run_path: Path, max_steps: int, episode_index: int) -> Path:
    np = _np()
    obs = env.reset()
    dt = env_dt(env)
    panel = MetricPanel(args.height, args.plot_width, dt) if args.with_plots else None
    frames: list[Any] = []
    try:
        for step in range(max_steps):
            muscle_states = getattr(env, "muscle_states", None)
            if bool(getattr(args, "noisy", False)) and hasattr(agent, "noisy_test_step"):
                action = agent.noisy_test_step(obs, muscle_states=muscle_states, steps=1_000_000)
            else:
                action = agent.test_step(obs, muscle_states=muscle_states, steps=1_000_000)
            if len(np.asarray(action).shape) > 1:
                action = np.asarray(action)[0, :]
            obs, reward, done, info = env.step(action)
            frame = render_rgb_frame(env, args.width, args.height, args.camera_id)
            if panel is not None:
                snapshot = extract_physics_snapshot(env, action)
                panel.update(step, snapshot["activation"], snapshot["torque"])
                frame = np.concatenate((frame, panel.frame()), axis=1)
            frames.append(frame)
            if done:
                break
    finally:
        if panel is not None:
            panel.close()
    out_dir = _video_output_dir(args, Path(run_path))
    return save_video(frames, out_dir / _video_name(str(getattr(args, "algorithm", "deprl")), env_id, episode_index), args.fps)

