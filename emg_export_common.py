"""Target-muscle EMG export helpers for MSK-Bench algorithms."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import msk_eval_common as common

TARGET_EMG_MUSCLES = (
    "soleus_r",
    "soleus_l",
    "gasmed_r",
    "gaslat_r",
    "tibant_r",
    "perlong_r",
    "perbrev_r",
    "recfem_r",
    "vaslat_r",
    "vasmed_r",
    "bflh_r",
    "semiten_r",
)
EMG_METADATA_COLUMNS = ("algorithm", "env_id", "episode", "step", "time_s", "reward", "done")


def _np():
    import numpy as np

    return np


def _physics_model(env: Any) -> Any | None:
    base = common.unwrap_env(env)
    sim = getattr(base, "sim", None)
    if sim is not None and hasattr(sim, "model"):
        return sim.model
    return getattr(base, "model", None)


def _decode_model_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def actuator_names(env: Any) -> list[str]:
    model = _physics_model(env)
    if model is None:
        return []

    for attr in ("actuator_names", "names_actuator"):
        names = getattr(model, attr, None)
        if names:
            try:
                return [_decode_model_name(name) for name in names]
            except TypeError:
                pass

    count = int(getattr(model, "nu", 0) or getattr(model, "na", 0) or 0)
    if count <= 0:
        try:
            count = int(len(getattr(model, "actuator_ctrlrange")))
        except Exception:
            count = 0
    if count <= 0:
        return []

    names = []
    for index in range(count):
        name = None
        for method_name in ("actuator_id2name", "id2name"):
            method = getattr(model, method_name, None)
            if not callable(method):
                continue
            try:
                name = method(index, "actuator") if method_name == "id2name" else method(index)
            except TypeError:
                try:
                    name = method("actuator", index)
                except Exception:
                    name = None
            except Exception:
                name = None
            if name:
                break
        if not name:
            actuator_method = getattr(model, "actuator", None)
            if callable(actuator_method):
                try:
                    name = getattr(actuator_method(index), "name", None)
                except Exception:
                    name = None
        names.append(_decode_model_name(name) if name else "")

    if any(names):
        return names

    try:
        import mujoco

        obj_type = mujoco.mjtObj.mjOBJ_ACTUATOR
        mj_model = getattr(model, "ptr", model)
        return [mujoco.mj_id2name(mj_model, obj_type, index) or "" for index in range(count)]
    except Exception:
        return []


def target_muscle_indices(env: Any, activation_size: int) -> dict[str, int | None]:
    names = actuator_names(env)
    normalized = {name.strip().lower(): index for index, name in enumerate(names) if name}
    indices: dict[str, int | None] = {}
    for muscle in TARGET_EMG_MUSCLES:
        index = normalized.get(muscle.lower())
        indices[muscle] = index if index is not None and index < activation_size else None

    if not names and activation_size == len(TARGET_EMG_MUSCLES):
        return {muscle: index for index, muscle in enumerate(TARGET_EMG_MUSCLES)}
    return indices


def target_emg_values(env: Any, activation: Any) -> dict[str, float | str]:
    np = _np()
    activation_array = np.asarray(activation, dtype=float).reshape(-1)
    indices = target_muscle_indices(env, int(activation_array.size))
    values: dict[str, float | str] = {}
    for muscle in TARGET_EMG_MUSCLES:
        index = indices.get(muscle)
        values[muscle] = float(activation_array[index]) if index is not None else ""
    return values


def target_emg_row(
    *,
    algorithm: str,
    env_id: str,
    episode: int,
    step: int,
    time_s: float,
    reward: float,
    done: bool,
    env: Any,
    activation: Any,
) -> dict[str, Any]:
    return {
        "algorithm": algorithm,
        "env_id": env_id,
        "episode": episode,
        "step": step,
        "time_s": float(time_s),
        "reward": float(reward),
        "done": bool(done),
        **target_emg_values(env, activation),
    }


def write_emg_csv(rows: list[dict[str, Any]], csv_path: str | Path) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EMG_METADATA_COLUMNS) + list(TARGET_EMG_MUSCLES)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_sb3_target_emg_episode(
    model: Any,
    venv: Any,
    env_id: str,
    algorithm: str,
    deterministic: bool,
    max_steps: int,
    episode_index: int,
) -> list[dict[str, Any]]:
    np = _np()
    obs = venv.reset()
    env = venv.envs[0]
    dt = common.env_dt(env)
    rows: list[dict[str, Any]] = []
    for step_index in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        action_one = np.asarray(action)[0]
        obs, rewards, dones, infos = venv.step(action)
        done = bool(np.asarray(dones).reshape(-1)[0])
        reward = float(np.asarray(rewards).reshape(-1)[0])
        snapshot = common.extract_physics_snapshot(env, action_one)
        rows.append(
            target_emg_row(
                algorithm=algorithm,
                env_id=env_id,
                episode=episode_index,
                step=step_index + 1,
                time_s=(step_index + 1) * dt,
                reward=reward,
                done=done,
                env=env,
                activation=snapshot["activation"],
            )
        )
        if done:
            break
    return rows


def collect_deprl_target_emg_episode(
    agent: Any,
    env: Any,
    env_id: str,
    algorithm: str,
    noisy: bool,
    max_steps: int,
    episode_index: int,
) -> list[dict[str, Any]]:
    np = _np()
    obs = env.reset()
    dt = common.env_dt(env)
    rows: list[dict[str, Any]] = []
    for step_index in range(max_steps):
        muscle_states = getattr(env, "muscle_states", None)
        if noisy and hasattr(agent, "noisy_test_step"):
            action = agent.noisy_test_step(obs, muscle_states=muscle_states, steps=1_000_000)
        else:
            action = agent.test_step(obs, muscle_states=muscle_states, steps=1_000_000)
        if len(np.asarray(action).shape) > 1:
            action = np.asarray(action)[0, :]
        obs, reward, done, info = env.step(action)
        snapshot = common.extract_physics_snapshot(env, action)
        rows.append(
            target_emg_row(
                algorithm=algorithm,
                env_id=env_id,
                episode=episode_index,
                step=step_index + 1,
                time_s=(step_index + 1) * dt,
                reward=float(reward),
                done=bool(done),
                env=env,
                activation=snapshot["activation"],
            )
        )
        if done:
            break
    return rows


def _prepare_emg_parser(base_module: Any, description: str):
    parser = common._prepare_eval_parser(base_module, description, 1)
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for per-episode EMG CSV files.")
    return parser


def _emg_output_dir(args: Any, artifact_path: Path) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    return Path(artifact_path) / "emg"


def _emg_name(algorithm: str, env_id: str, episode_index: int) -> str:
    slug = env_id.removeprefix("MSKBench").removesuffix("-v0")
    return f"{algorithm.lower()}_{slug}_episode{episode_index}_emg.csv"


def main_sb3_emg_export(base_module: Any, argv: list[str] | None = None) -> int:
    common.configure_headless_rendering()
    algorithm = base_module.ALGORITHM_NAME
    parser = _prepare_emg_parser(base_module, f"{algorithm} 12-target-muscle EMG export for MSK-Bench tasks.")
    args = parser.parse_args(argv)
    args.algorithm = algorithm
    if args.list_envs:
        for env_id in base_module.MSK_BENCH_ENVS:
            print(env_id)
        return 0

    base_module.load_runtime(Path(args.benchmark_root))
    summary_rows: list[dict[str, Any]] = []
    artifact_key = common._model_artifact_key(base_module)
    for env_id in base_module.selected_envs(args.env):
        model, venv, artifact_path = base_module.load_model_and_env(args, env_id)
        max_steps = base_module.max_steps_for(env_id, args.max_steps)
        out_dir = _emg_output_dir(args, Path(artifact_path))
        for episode_index in range(1, args.episodes + 1):
            rows = collect_sb3_target_emg_episode(
                model,
                venv,
                env_id,
                algorithm,
                bool(getattr(args, "deterministic", True)),
                max_steps,
                episode_index,
            )
            csv_path = out_dir / _emg_name(algorithm, env_id, episode_index)
            write_emg_csv(rows, csv_path)
            summary_rows.append(
                {
                    "algorithm": algorithm,
                    "env_id": env_id,
                    "episode": episode_index,
                    "steps": len(rows),
                    artifact_key: str(artifact_path),
                    "emg_csv": str(csv_path),
                }
            )
            print(f"{env_id:<28} episode={episode_index:<3} emg_csv={csv_path}")
        venv.close()

    common.write_rows(summary_rows, args.json, args.csv)
    return 0


def main_deprl_emg_export(base_module: Any, argv: list[str] | None = None) -> int:
    common.configure_headless_rendering()
    algorithm = base_module.ALGORITHM_NAME
    parser = _prepare_emg_parser(base_module, f"{algorithm} 12-target-muscle EMG export for MSK-Bench tasks.")
    args = parser.parse_args(argv)
    args.algorithm = algorithm
    if args.list_envs:
        for env_id in base_module.MSK_BENCH_ENVS:
            print(env_id)
        return 0

    base_module.load_runtime(args)
    summary_rows: list[dict[str, Any]] = []
    for env_id in base_module.selected_envs(args.env):
        agent, env, run_path = base_module.build_agent_env(args, env_id)
        max_steps = base_module.max_steps_for(env_id, args.max_steps)
        out_dir = _emg_output_dir(args, Path(run_path))
        for episode_index in range(1, args.episodes + 1):
            rows = collect_deprl_target_emg_episode(
                agent,
                env,
                env_id,
                algorithm,
                bool(getattr(args, "noisy", False)),
                max_steps,
                episode_index,
            )
            csv_path = out_dir / _emg_name(algorithm, env_id, episode_index)
            write_emg_csv(rows, csv_path)
            summary_rows.append(
                {
                    "algorithm": algorithm,
                    "env_id": env_id,
                    "episode": episode_index,
                    "steps": len(rows),
                    "run_path": str(run_path),
                    "emg_csv": str(csv_path),
                }
            )
            print(f"{env_id:<28} episode={episode_index:<3} emg_csv={csv_path}")
        if hasattr(env, "close"):
            env.close()

    common.write_rows(summary_rows, args.json, args.csv)
    return 0
