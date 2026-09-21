"""Durable pause/observe/act harness for the direct-muscle-control case study.

No LLM client lives here.  Phase 2 is intentionally file mediated: this process
exports a controller-visible observation, exits, and later accepts exactly one
validated action.  Recovery rebuilds seed 0 and deterministically replays all
committed actions, which also restores the stair task's private reference state.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

import numpy as np

from .action_io import atomic_write_json, load_action_file, parse_utc_timestamp, validate_action
from .observation_serializer import ObservationLayout, serialize_policy_observation


TASKS: dict[str, tuple[str, str, str]] = {
    "walk": (
        "rl_paradigms.residualrl.walk",
        "MSKBenchResidualWalkEnvV0",
        "DEFAULT_WALK_MOTION_PATH",
    ),
    "run": (
        "rl_paradigms.residualrl.run",
        "MSKBenchResidualRunEnvV0",
        "DEFAULT_RUN_MOTION_PATH",
    ),
    "stairs": (
        "rl_paradigms.residualrl.stair",
        "MSKBenchResidualStairEnvV0",
        "DEFAULT_STAIR_MOTION_PATH",
    ),
}
TASK_ALIASES = {"stair": "stairs", "walking": "walk", "running": "run"}
PUBLIC_ENV_IDS = {
    "walk": "MSKBenchResidualWalk-v0",
    "run": "MSKBenchResidualRun-v0",
    "stairs": "MSKBenchResidualStair-v0",
}
DEFAULT_SEED = 0
DEFAULT_HOLD_SECONDS = 0.2
DEFAULT_MAX_DURATION = 3.0
DEFAULT_MAX_DECISIONS = 15


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_task(task: str) -> str:
    value = TASK_ALIASES.get(str(task).strip().lower(), str(task).strip().lower())
    if value not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {tuple(TASKS)}.")
    return value


def _task_module(task: str):
    return importlib.import_module(TASKS[normalize_task(task)][0])


def task_motion_path(task: str) -> Path:
    module = _task_module(task)
    return Path(getattr(module, TASKS[normalize_task(task)][2])).resolve()


def unwrap_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    return getattr(current, "unwrapped", current)


def make_direct_env(task: str):
    """Construct the no-base-policy environment used by GPT and mock control."""

    normalized = normalize_task(task)
    module = _task_module(normalized)
    return getattr(module, TASKS[normalized][1])()


def make_residual_env(task: str, checkpoint: str):
    """Construct the existing wrapper with a real pretrained base policy."""

    if not checkpoint:
        raise ValueError("Residual runs require an explicit resolved checkpoint path.")
    module = _task_module(task)
    return module.make_env(base_model_dir=str(checkpoint))


def simulation_time(env: Any) -> float:
    candidate = env
    if hasattr(candidate, "_data"):
        return float(candidate._data.time)
    base = unwrap_env(env)
    if hasattr(base, "_data"):
        return float(base._data.time)
    sim = getattr(base, "sim", None)
    if sim is not None and hasattr(sim, "data"):
        return float(sim.data.time)
    raise AttributeError("Environment does not expose MuJoCo simulation time.")


def decision_deadline(decision_index: int, *, hold_seconds: float = DEFAULT_HOLD_SECONDS) -> float:
    if decision_index < 0:
        raise ValueError("decision_index must be non-negative.")
    if not np.isfinite(hold_seconds) or hold_seconds <= 0.0:
        raise ValueError("hold_seconds must be finite and positive.")
    return float((decision_index + 1) * hold_seconds)


@dataclass(frozen=True)
class StepTransition:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    simulation_time: float
    native_step: int


@dataclass(frozen=True)
class StepBatchResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    simulation_time: float
    native_steps: int
    stop_reason: str | None = None


def step_until_deadline(
    env: Any,
    action: np.ndarray,
    *,
    deadline: float,
    on_step: Callable[[StepTransition], None] | None = None,
    stop_if: Callable[[StepTransition], str | None] | None = None,
    tolerance: float = 1e-10,
    max_native_steps: int = 100_000,
) -> StepBatchResult:
    """Hold one action until absolute simulator time crosses ``deadline``."""

    if not np.isfinite(deadline) or deadline < simulation_time(env) - tolerance:
        raise ValueError("Deadline must be finite and not precede current simulation time.")
    observation: Any = None
    reward = 0.0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    steps = 0
    stop_reason: str | None = None
    while simulation_time(env) < deadline - tolerance:
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        transition = StepTransition(
            observation=observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
            simulation_time=simulation_time(env),
            native_step=steps,
        )
        if on_step is not None:
            on_step(transition)
        if stop_if is not None:
            stop_reason = stop_if(transition)
            if stop_reason is not None:
                break
        if terminated or truncated:
            break
        if steps >= max_native_steps:
            raise RuntimeError(f"Simulator did not reach deadline {deadline} in {max_native_steps} steps.")
    return StepBatchResult(
        observation=observation,
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=dict(info),
        simulation_time=simulation_time(env),
        native_steps=steps,
        stop_reason=stop_reason,
    )


def _actuator_names(base: Any) -> list[str]:
    return [base._model.actuator(index).name or f"actuator_{index}" for index in range(base._model.nu)]


def environment_info(env: Any, task: str, controller: str, checkpoint: str | None = None) -> dict[str, Any]:
    import mujoco

    base = unwrap_env(env)
    normalized_task = normalize_task(task)
    layout = ObservationLayout.from_env(base)
    return {
        "task": normalized_task,
        "controller": controller,
        "env_id": PUBLIC_ENV_IDS[normalized_task],
        "environment_class": type(base).__name__,
        "simulator": "MuJoCo",
        "mujoco_version": getattr(mujoco, "__version__", "unknown"),
        "seed": DEFAULT_SEED,
        "observation_dim": int(np.prod(env.observation_space.shape)),
        "action_dim": int(np.prod(env.action_space.shape)),
        "action_low": np.asarray(env.action_space.low, dtype=float).reshape(-1).tolist(),
        "action_high": np.asarray(env.action_space.high, dtype=float).reshape(-1).tolist(),
        "actuator_names": _actuator_names(base),
        "observation_layout": layout.as_dict(),
        "physics_timestep_seconds": float(base._model.opt.timestep),
        "physics_substeps_per_control": int(base._n_substeps),
        "control_timestep_seconds": float(base.physics_control_dt),
        "decision_frequency_hz": 1.0 / DEFAULT_HOLD_SECONDS,
        "action_hold_seconds": DEFAULT_HOLD_SECONDS,
        "max_duration_seconds": DEFAULT_MAX_DURATION,
        "max_decisions": DEFAULT_MAX_DECISIONS,
        "motion_path": str(task_motion_path(task)),
        "motion_frames": int(base.th.n_frames),
        "motion_timestep_seconds": float(base.th.dt),
        "base_policy_enabled": bool(getattr(env, "_base_policy_fn", None) is not None),
        "checkpoint": checkpoint,
        "controller_visible_contract": (
            "Only env_info.json and observations/decision_XXX_controller_visible.json may inform GPT actions."
        ),
        "logging_only_contract": (
            "trajectory.npz, diagnostics, rendered frames and analysis must not be read before the next GPT action."
        ),
    }


def _tilt_degrees(qpos: np.ndarray) -> float:
    if qpos.size < 7:
        return float("nan")
    quat = np.asarray(qpos[3:7], dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0 or not np.isfinite(norm):
        return 180.0
    w, x, y, z = quat / norm
    del w, z
    up_z = float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
    return float(np.degrees(np.arccos(up_z)))


def logging_diagnostics(env: Any, observation: np.ndarray, layout: ObservationLayout) -> dict[str, Any]:
    """Compute privileged diagnostics that are never placed in controller-visible files."""

    base = unwrap_env(env)
    data = base._data
    qpos = np.asarray(data.qpos, dtype=float)
    root_height = float(qpos[2])
    tilt = _tilt_degrees(qpos)
    ref_step = int(base.ref_step)
    reference = base.th.get_traj_data_at(0, ref_step)
    reference_root = np.asarray(reference.qpos[:3], dtype=float)
    tracking_rmse = float("nan")
    try:
        wrapper = base._goal_wrapper
        wrapper._lazy_init()
        ids = np.asarray(wrapper._rel_site_ids, dtype=int)
        delta = np.asarray(data.site_xpos[ids], dtype=float) - np.asarray(reference.site_xpos[ids], dtype=float)
        tracking_rmse = float(np.sqrt(np.mean(np.square(delta))))
    except Exception:
        pass
    obs = np.asarray(observation, dtype=float).reshape(-1)
    touch_start = layout.base_dim - len(layout.touch_names)
    touch = obs[touch_start : layout.base_dim]
    finite_state = bool(np.all(np.isfinite(qpos)) and np.all(np.isfinite(data.qvel)))
    fallen = bool((not finite_state) or root_height < 0.6 or tilt > 70.0)
    return {
        "visibility": "logging-only",
        "reference_step": ref_step,
        "root_height": root_height,
        "root_tilt_degrees": tilt,
        "tracking_site_rmse": tracking_rmse,
        "touch": [float(value) for value in touch],
        "contact_count": int(data.ncon),
        "fallen": fallen,
        "root_xyz": [float(value) for value in qpos[:3]],
        "reference_root_xyz": [float(value) for value in reference_root],
    }


class TrajectoryRecorder:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata
        self.rows: list[dict[str, Any]] = []

    def append(
        self,
        env: Any,
        observation: np.ndarray,
        commanded_action: np.ndarray,
        layout: ObservationLayout,
        *,
        decision_index: int,
        native_step: int,
    ) -> None:
        base = unwrap_env(env)
        data = base._data
        diagnostic = logging_diagnostics(env, observation, layout)
        self.rows.append(
            {
                "time": simulation_time(env),
                "decision_index": int(decision_index),
                "native_step": int(native_step),
                "observation": np.asarray(observation, dtype=np.float32).copy(),
                "commanded_action": np.asarray(commanded_action, dtype=np.float32).copy(),
                "qpos": np.asarray(data.qpos, dtype=np.float64).copy(),
                "qvel": np.asarray(data.qvel, dtype=np.float64).copy(),
                "act": np.asarray(data.act, dtype=np.float64).copy(),
                "ctrl": np.asarray(data.ctrl, dtype=np.float64).copy(),
                "qfrc_actuator": np.asarray(data.qfrc_actuator, dtype=np.float64).copy(),
                "root_height": diagnostic["root_height"],
                "root_tilt_degrees": diagnostic["root_tilt_degrees"],
                "tracking_site_rmse": diagnostic["tracking_site_rmse"],
                "touch": np.asarray(diagnostic["touch"], dtype=np.float32),
                "contact_count": diagnostic["contact_count"],
                "fallen": diagnostic["fallen"],
                "root_xyz": np.asarray(diagnostic["root_xyz"], dtype=np.float64),
                "reference_root_xyz": np.asarray(diagnostic["reference_root_xyz"], dtype=np.float64),
                "reference_step": diagnostic["reference_step"],
            }
        )

    def save(self, path: str | Path) -> Path:
        if not self.rows:
            raise ValueError("Cannot save an empty trajectory.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.stem}.{uuid4().hex}.tmp.npz")
        scalar_keys = (
            "time",
            "decision_index",
            "native_step",
            "root_height",
            "root_tilt_degrees",
            "tracking_site_rmse",
            "contact_count",
            "fallen",
            "reference_step",
        )
        array_keys = (
            "observation",
            "commanded_action",
            "qpos",
            "qvel",
            "act",
            "ctrl",
            "qfrc_actuator",
            "touch",
            "root_xyz",
            "reference_root_xyz",
        )
        payload = {key: np.asarray([row[key] for row in self.rows]) for key in scalar_keys}
        payload.update({key: np.stack([row[key] for row in self.rows]) for key in array_keys})
        payload["metadata_json"] = np.asarray(json.dumps(self.metadata, ensure_ascii=False, allow_nan=False))
        try:
            np.savez_compressed(temporary, **payload)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination


@dataclass(frozen=True)
class RolloutResult:
    trajectory_path: Path
    summary_path: Path
    final_observation: np.ndarray
    final_simulation_time: float
    decisions_executed: int
    native_steps: int
    terminated: bool
    truncated: bool
    fallen: bool
    stop_reason: str | None
    layout: ObservationLayout
    env_info: dict[str, Any]


def run_rollout(
    task: str,
    actions: Iterable[Iterable[float] | np.ndarray],
    output_dir: str | Path,
    *,
    controller: str,
    checkpoint: str | None = None,
    seed: int = DEFAULT_SEED,
    hold_seconds: float = DEFAULT_HOLD_SECONDS,
    max_duration: float = DEFAULT_MAX_DURATION,
) -> RolloutResult:
    """Run a finite action schedule and persist every native control state."""

    normalized = normalize_task(task)
    action_list = list(actions)
    if len(action_list) > DEFAULT_MAX_DECISIONS:
        raise ValueError(f"At most {DEFAULT_MAX_DECISIONS} decisions are allowed.")
    if controller == "residual_pretrained_base_zero_correction":
        env = make_residual_env(normalized, str(checkpoint or ""))
    else:
        env = make_direct_env(normalized)
    output = Path(output_dir)
    try:
        observation, _ = env.reset(seed=seed)
        layout = ObservationLayout.from_env(env)
        info = environment_info(env, normalized, controller, checkpoint)
        metadata = {
            "format_version": 1,
            "task": normalized,
            "controller": controller,
            "seed": int(seed),
            "hold_seconds": float(hold_seconds),
            "max_duration": float(max_duration),
            "requested_decisions": len(action_list),
            "requested_duration": min(len(action_list) * hold_seconds, max_duration),
            "created_at_utc": utc_now(),
            "checkpoint": checkpoint,
            "visibility": "logging-only",
        }
        recorder = TrajectoryRecorder(metadata)
        zero = np.zeros(env.action_space.shape, dtype=np.float32)
        recorder.append(env, observation, zero, layout, decision_index=-1, native_step=0)
        global_step = 0
        decisions = 0
        terminated = False
        truncated = False
        stop_reason: str | None = None
        for decision_index, proposed in enumerate(action_list):
            action = validate_action(proposed, env.action_space)
            deadline = min(decision_deadline(decision_index, hold_seconds=hold_seconds), max_duration)

            def record_transition(transition: StepTransition) -> None:
                nonlocal global_step
                global_step += 1
                recorder.append(
                    env,
                    transition.observation,
                    action,
                    layout,
                    decision_index=decision_index,
                    native_step=global_step,
                )

            result = step_until_deadline(
                env,
                action,
                deadline=deadline,
                on_step=record_transition,
                stop_if=lambda transition: (
                    "fall"
                    if logging_diagnostics(env, transition.observation, layout)["fallen"]
                    else None
                ),
            )
            if result.observation is not None:
                observation = np.asarray(result.observation, dtype=np.float32)
            decisions += 1
            terminated = result.terminated
            truncated = result.truncated
            stop_reason = result.stop_reason
            if terminated or truncated or stop_reason is not None or result.simulation_time >= max_duration - 1e-10:
                break
        trajectory_path = recorder.save(output / "trajectory.npz")
        diagnostics = logging_diagnostics(env, observation, layout)
        fallen = bool(diagnostics["fallen"])
        if fallen and stop_reason is None:
            stop_reason = "fall"
        summary = {
            "task": normalized,
            "controller": controller,
            "seed": int(seed),
            "decisions_executed": decisions,
            "native_steps": global_step,
            "final_simulation_time": simulation_time(env),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "fallen": fallen,
            "stop_reason": stop_reason,
            "final_logging_only_diagnostics": diagnostics,
            "trajectory": str(trajectory_path),
        }
        summary_path = atomic_write_json(output / "rollout.json", summary)
        atomic_write_json(output / "env_info.json", info)
        return RolloutResult(
            trajectory_path=trajectory_path,
            summary_path=summary_path,
            final_observation=np.asarray(observation, dtype=np.float32).copy(),
            final_simulation_time=simulation_time(env),
            decisions_executed=decisions,
            native_steps=global_step,
            terminated=bool(terminated),
            truncated=bool(truncated),
            fallen=fallen,
            stop_reason=stop_reason,
            layout=layout,
            env_info=info,
        )
    finally:
        if hasattr(env, "close"):
            env.close()


def collect_environment_info(output_path: str | Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "phase": 1,
        "real_gpt_actions_generated": 0,
        "tasks": {},
    }
    for task in TASKS:
        env = make_direct_env(task)
        try:
            observation, _ = env.reset(seed=DEFAULT_SEED)
            info = environment_info(env, task, "direct_no_base_policy")
            info["initial_observation_finite"] = bool(np.all(np.isfinite(observation)))
            result["tasks"][task] = info
        finally:
            env.close()
    atomic_write_json(output_path, result)
    return result


def _observation_path(session_dir: Path, decision_index: int) -> Path:
    return session_dir / "observations" / f"decision_{decision_index:03d}_controller_visible.json"


def initialize_session(task: str, session_dir: str | Path) -> dict[str, Any]:
    """Create a fresh Phase 2 session and export observation zero, then stop."""

    directory = Path(session_dir)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Session already exists: {manifest_path}")
    rollout = run_rollout(task, [], directory, controller="gpt_direct", seed=DEFAULT_SEED)
    exported_at = utc_now()
    observation_path = _observation_path(directory, 0)
    payload = serialize_policy_observation(
        rollout.final_observation,
        rollout.layout,
        task=normalize_task(task),
        decision_index=0,
        simulation_time=rollout.final_simulation_time,
        exported_at_utc=exported_at,
    )
    atomic_write_json(observation_path, payload, indent=None)
    manifest = {
        "format_version": 1,
        "phase": 2,
        "task": normalize_task(task),
        "controller": "gpt_direct",
        "seed": DEFAULT_SEED,
        "hold_seconds": DEFAULT_HOLD_SECONDS,
        "max_duration": DEFAULT_MAX_DURATION,
        "max_decisions": DEFAULT_MAX_DECISIONS,
        "status": "awaiting_action",
        "decisions": [],
        "pending_observation": str(observation_path),
        "pending_observation_exported_at_utc": exported_at,
        "controller_may_read": [str(directory / "env_info.json"), str(observation_path)],
        "controller_must_not_read": [str(directory / "trajectory.npz"), str(directory / "rollout.json")],
    }
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(directory / "actions.json", {"task": normalize_task(task), "decisions": []})
    return manifest


def submit_action(session_dir: str | Path, action_file: str | Path) -> dict[str, Any]:
    """Commit one action, replay the session, export the next observation, then stop."""

    directory = Path(session_dir)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "awaiting_action":
        raise RuntimeError(f"Session is not awaiting an action: {manifest.get('status')!r}.")
    decision_index = len(manifest["decisions"])
    if decision_index >= int(manifest["max_decisions"]):
        raise RuntimeError("Decision budget exhausted.")
    received_at = utc_now()

    validation_env = make_direct_env(manifest["task"])
    try:
        action = load_action_file(action_file, validation_env.action_space)
    finally:
        validation_env.close()
    actions = [entry["action"] for entry in manifest["decisions"]] + [action]
    rollout = run_rollout(
        manifest["task"],
        actions,
        directory,
        controller="gpt_direct",
        seed=int(manifest["seed"]),
        hold_seconds=float(manifest["hold_seconds"]),
        max_duration=float(manifest["max_duration"]),
    )
    rollout_fallen = bool(getattr(rollout, "fallen", False))
    rollout_stop_reason = getattr(rollout, "stop_reason", None)
    exported_timestamp = parse_utc_timestamp(manifest["pending_observation_exported_at_utc"])
    received_timestamp = parse_utc_timestamp(received_at)
    latency = max(0.0, received_timestamp - exported_timestamp)
    manifest["decisions"].append(
        {
            "decision_index": decision_index,
            "action": [float(value) for value in action],
            "observation_exported_at_utc": manifest["pending_observation_exported_at_utc"],
            "action_received_at_utc": received_at,
            "wall_clock_latency_seconds": latency,
            "simulation_deadline": decision_deadline(
                decision_index, hold_seconds=float(manifest["hold_seconds"])
            ),
            "simulation_time_after_action": rollout.final_simulation_time,
            "stop_reason_after_action": rollout_stop_reason,
        }
    )
    complete = bool(
        rollout.terminated
        or rollout.truncated
        or rollout_fallen
        or len(manifest["decisions"]) >= int(manifest["max_decisions"])
        or rollout.final_simulation_time >= float(manifest["max_duration"]) - 1e-10
    )
    if complete:
        manifest["status"] = "complete"
        manifest["pending_observation"] = None
        manifest["pending_observation_exported_at_utc"] = None
        manifest["controller_may_read"] = [str(directory / "env_info.json")]
        if rollout_stop_reason is not None:
            manifest["stop_reason"] = rollout_stop_reason
        elif rollout.terminated:
            manifest["stop_reason"] = "terminated"
        elif rollout.truncated:
            manifest["stop_reason"] = "truncated"
        elif len(manifest["decisions"]) >= int(manifest["max_decisions"]):
            manifest["stop_reason"] = "decision_budget"
        else:
            manifest["stop_reason"] = "simulation_horizon"
    else:
        next_index = len(manifest["decisions"])
        next_exported_at = utc_now()
        next_path = _observation_path(directory, next_index)
        payload = serialize_policy_observation(
            rollout.final_observation,
            rollout.layout,
            task=manifest["task"],
            decision_index=next_index,
            simulation_time=rollout.final_simulation_time,
            exported_at_utc=next_exported_at,
        )
        atomic_write_json(next_path, payload, indent=None)
        manifest["pending_observation"] = str(next_path)
        manifest["pending_observation_exported_at_utc"] = next_exported_at
        manifest["controller_may_read"] = [str(directory / "env_info.json"), str(next_path)]
    manifest["updated_at_utc"] = utc_now()
    atomic_write_json(
        directory / "actions.json",
        {"task": manifest["task"], "controller": "gpt_direct", "decisions": manifest["decisions"]},
    )
    atomic_write_json(manifest_path, manifest)
    return manifest


def run_mock(task: str, output_dir: str | Path, decisions: int = 2) -> RolloutResult:
    """Run the deterministic zero-action mock.  This is never labeled as GPT."""

    if decisions < 1 or decisions > DEFAULT_MAX_DECISIONS:
        raise ValueError(f"decisions must be in [1, {DEFAULT_MAX_DECISIONS}].")
    env = make_direct_env(task)
    try:
        action_shape = env.action_space.shape
    finally:
        env.close()
    actions = [np.zeros(action_shape, dtype=np.float32) for _ in range(decisions)]
    return run_rollout(task, actions, output_dir, controller="mock_zero", seed=DEFAULT_SEED)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    env_info_parser = sub.add_parser("env-info", help="Inspect all three direct environments.")
    env_info_parser.add_argument("--output", type=Path, required=True)
    init_parser = sub.add_parser("init", help="Initialize one durable GPT-direct session.")
    init_parser.add_argument("--task", choices=tuple(TASKS), required=True)
    init_parser.add_argument("--session-dir", type=Path, required=True)
    submit_parser = sub.add_parser("submit", help="Submit exactly one action and pause again.")
    submit_parser.add_argument("--session-dir", type=Path, required=True)
    submit_parser.add_argument("--action-file", type=Path, required=True)
    mock_parser = sub.add_parser("mock", help="Run deterministic zero-action pipeline smoke.")
    mock_parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    mock_parser.add_argument("--output-dir", type=Path, required=True)
    mock_parser.add_argument("--decisions", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "env-info":
        collect_environment_info(args.output)
        print(f"Wrote {args.output}")
    elif args.command == "init":
        manifest = initialize_session(args.task, args.session_dir)
        print(json.dumps({"status": manifest["status"], "observation": manifest["pending_observation"]}, indent=2))
    elif args.command == "submit":
        manifest = submit_action(args.session_dir, args.action_file)
        print(json.dumps({"status": manifest["status"], "next": manifest["pending_observation"]}, indent=2))
    elif args.command == "mock":
        tasks = TASKS if args.task == "all" else (args.task,)
        for task in tasks:
            result = run_mock(task, args.output_dir / f"mock_{task}", decisions=args.decisions)
            print(f"MOCK_OK task={task} steps={result.native_steps} trajectory={result.trajectory_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
