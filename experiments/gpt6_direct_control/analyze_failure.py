"""Paired qualitative diagnostics for GPT-direct and residual rollouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .action_io import atomic_write_json
from .render_rollout import trim_arrays_at_first_fall

SUMMARY_COLUMNS = [
    "task",
    "controller",
    "seed",
    "success",
    "survived_full_3s_horizon",
    "sim_duration_seconds",
    "survival_steps",
    "tracking_rmse",
    "final_tracking_error",
    "max_tracking_error",
    "mean_action_delta",
    "max_action_delta",
    "mean_action_norm",
    "GPT_decisions",
    "mean_wall_clock_decision_time",
    "median_wall_clock_decision_time",
    "max_wall_clock_decision_time",
    "first_divergence_time",
    "termination_reason",
]

EXTRA_COLUMNS = [
    "protocol_analyzed_decisions",
    "raw_state_count",
    "protocol_state_count",
    "horizontal_displacement",
    "progress_ratio",
    "root_height_min",
    "torso_pitch_max_abs_degrees",
    "torso_roll_max_abs_degrees",
    "right_contact_fraction",
    "left_contact_fraction",
    "failure_mode",
    "protocol_note",
    "timing_note",
    "pause_excluded_mean_wall_clock_decision_time",
    "trajectory",
    "failure_trace",
]


def _finite(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _finite_mean(values: Any) -> float:
    finite = _finite(values)
    return float(np.mean(finite)) if finite.size else float("nan")


def normalized_action_change_series(actions: Any) -> np.ndarray:
    """Return ||a_t-a_(t-1)||_2/sqrt(M), aligned with the action rows."""

    values = np.asarray(actions, dtype=float)
    if values.ndim != 2:
        raise ValueError("actions must be a 2-D array")
    if values.shape[0] == 0:
        return np.zeros(0, dtype=float)
    result = np.zeros(values.shape[0], dtype=float)
    if values.shape[0] > 1:
        result[1:] = np.linalg.norm(np.diff(values, axis=0), axis=1) / np.sqrt(values.shape[1])
    return result


def quaternion_pitch_roll(qpos: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract pitch and roll in degrees from MuJoCo free-joint qpos (w, x, y, z)."""

    values = np.asarray(qpos, dtype=float)
    if values.ndim != 2 or values.shape[1] < 7:
        raise ValueError("qpos must have at least seven columns")
    w, x, y, z = (values[:, index] for index in range(3, 7))
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sin_roll, cos_roll)
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    return np.degrees(pitch), np.degrees(roll)


def contact_series(touch: Any) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(touch, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("touch must be a 2-D array with at least two sensors")
    split = values.shape[1] // 2
    right = np.any(values[:, :split] > 0.0, axis=1).astype(float)
    left = np.any(values[:, split:] > 0.0, axis=1).astype(float)
    return right, left


def first_sustained_divergence(
    gpt_time: Any,
    gpt_error: Any,
    residual_time: Any,
    residual_error: Any,
    *,
    min_steps: int = 5,
) -> float | None:
    """Find the first sustained GPT error above the residual 95th percentile.

    A candidate must also exceed the residual error interpolated at the same
    simulation time.  This data-derived threshold avoids treating one spike as
    divergence and avoids inventing a task-specific distance cutoff.
    """

    gt = np.asarray(gpt_time, dtype=float).reshape(-1)
    ge = np.asarray(gpt_error, dtype=float).reshape(-1)
    rt = np.asarray(residual_time, dtype=float).reshape(-1)
    re = np.asarray(residual_error, dtype=float).reshape(-1)
    if min_steps < 1:
        raise ValueError("min_steps must be positive")
    valid_g = np.isfinite(gt) & np.isfinite(ge)
    valid_r = np.isfinite(rt) & np.isfinite(re)
    gt, ge, rt, re = gt[valid_g], ge[valid_g], rt[valid_r], re[valid_r]
    if gt.size < min_steps or rt.size == 0:
        return None
    threshold = float(np.quantile(re, 0.95))
    aligned = np.interp(gt, rt, re, left=re[0], right=re[-1])
    above = (ge > threshold) & (ge > aligned)
    for start in range(0, above.size - min_steps + 1):
        if bool(np.all(above[start : start + min_steps])):
            return float(gt[start])
    return None


def select_keyframe_times(
    *,
    duration: float,
    first_divergence: float | None,
    decision_times: Iterable[float],
    fallen: bool,
) -> list[tuple[str, float]]:
    """Choose at most four ordered semantic frame times."""

    candidates: list[tuple[str, float]] = [("initial", 0.0)]
    decisions = sorted(float(value) for value in decision_times if 0.0 <= float(value) <= duration)
    if first_divergence is not None and np.isfinite(first_divergence):
        divergence = float(np.clip(first_divergence, 0.0, duration))
        candidates.append(("first_divergence", divergence))
        correction = next((value for value in decisions if value > divergence + 1e-9), None)
        if correction is not None and correction < duration - 1e-9:
            candidates.append(("attempted_correction", correction))
    candidates.append(("fall" if fallen else "final_3s", float(duration)))
    unique: list[tuple[str, float]] = []
    for label, timestamp in candidates:
        if not any(abs(timestamp - previous) < 1e-9 for _, previous in unique):
            unique.append((label, timestamp))
    return unique[:4]


def classify_failure(
    *,
    fallen: bool,
    progress_ratio: float,
    tracking_rmse_mean: float,
    max_tilt: float,
) -> str:
    """Assign one interpretable primary failure mode with fixed priority."""

    if fallen:
        return "fall"
    if np.isfinite(progress_ratio) and progress_ratio < 0.25:
        return "insufficient_progress"
    if np.isfinite(tracking_rmse_mean) and tracking_rmse_mean > 0.5:
        return "tracking_loss"
    if np.isfinite(max_tilt) and max_tilt > 45.0:
        return "balance_instability"
    return "none_observed"


def _effective_actions(arrays: dict[str, np.ndarray], controller: str) -> np.ndarray:
    key = "commanded_action" if controller == "gpt_direct" else "ctrl"
    return np.asarray(arrays[key], dtype=float)


def _decision_action_rows(arrays: dict[str, np.ndarray], controller: str) -> np.ndarray:
    actions = _effective_actions(arrays, controller)
    if controller != "gpt_direct":
        return actions
    indices = np.asarray(arrays["decision_index"], dtype=int).reshape(-1)
    selected: list[int] = []
    for decision in sorted(set(int(value) for value in indices if value >= 0)):
        matches = np.flatnonzero(indices == decision)
        if matches.size:
            selected.append(int(matches[0]))
    return actions[selected] if selected else actions[:0]


def _decision_times(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> list[float]:
    indices = np.asarray(arrays["decision_index"], dtype=int).reshape(-1)
    count = len(set(int(value) for value in indices if value >= 0))
    hold = float(metadata.get("hold_seconds", 0.2))
    return [index * hold for index in range(count)]


def _load_adjacent_json(trajectory: Path, name: str) -> dict[str, Any]:
    path = trajectory.parent / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _timing_metrics(trajectory: Path, controller: str) -> dict[str, Any]:
    if controller != "gpt_direct":
        return {
            "GPT_decisions": None,
            "mean_wall_clock_decision_time": None,
            "median_wall_clock_decision_time": None,
            "max_wall_clock_decision_time": None,
            "pause_excluded_mean_wall_clock_decision_time": None,
            "timing_note": "N/A (Residual controller; no GPT decisions)",
        }
    manifest = _load_adjacent_json(trajectory, "manifest.json")
    decisions = list(manifest.get("decisions", []))
    values = np.asarray([item.get("wall_clock_latency_seconds", np.nan) for item in decisions], dtype=float)
    values = values[np.isfinite(values)]
    filtered = values[values < 3600.0]
    has_pause = bool(values.size and np.any(values >= 3600.0))
    note = "Raw observation-to-action elapsed time."
    if has_pause:
        note += " Includes a documented multi-hour user/session pause; not pure inference latency."
    return {
        "GPT_decisions": len(decisions),
        "mean_wall_clock_decision_time": float(np.mean(values)) if values.size else None,
        "median_wall_clock_decision_time": float(np.median(values)) if values.size else None,
        "max_wall_clock_decision_time": float(np.max(values)) if values.size else None,
        "pause_excluded_mean_wall_clock_decision_time": float(np.mean(filtered)) if filtered.size else None,
        "timing_note": note,
    }


def _trajectory_metrics(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    trajectory: Path,
    *,
    raw_state_count: int,
    first_divergence: float | None,
) -> dict[str, Any]:
    times = np.asarray(arrays["time"], dtype=float).reshape(-1)
    root = np.asarray(arrays["root_xyz"], dtype=float)
    reference = np.asarray(arrays["reference_root_xyz"], dtype=float)
    if times.size == 0 or root.shape[0] != times.size or reference.shape[0] != times.size:
        raise ValueError("Trajectory arrays have inconsistent or empty lengths.")
    controller = str(metadata.get("controller", "unknown"))
    actual_delta = root[-1] - root[0]
    reference_delta = reference[-1] - reference[0]
    reference_norm_sq = float(np.dot(reference_delta, reference_delta))
    progress_ratio = (
        float(np.dot(actual_delta, reference_delta) / reference_norm_sq)
        if reference_norm_sq > 1e-12
        else float("nan")
    )
    duration = float(times[-1] - times[0])
    fallen = bool(np.any(np.asarray(arrays["fallen"], dtype=bool)))
    tracking = np.asarray(arrays["tracking_site_rmse"], dtype=float).reshape(-1)
    tracking_finite = tracking[np.isfinite(tracking)]
    tracking_rmse = float(np.sqrt(np.mean(np.square(tracking_finite)))) if tracking_finite.size else float("nan")
    pitch, roll = quaternion_pitch_roll(arrays["qpos"])
    right_contact, left_contact = contact_series(arrays["touch"])
    action_rows = _decision_action_rows(arrays, controller)
    changes = normalized_action_change_series(action_rows)
    non_initial_changes = changes[1:] if changes.size > 1 else np.zeros(0)
    action_norms = np.linalg.norm(action_rows, axis=1) if action_rows.size else np.zeros(0)
    max_tilt = max(float(np.max(np.abs(pitch))), float(np.max(np.abs(roll))))
    failure_mode = classify_failure(
        fallen=fallen,
        progress_ratio=progress_ratio,
        tracking_rmse_mean=_finite_mean(tracking),
        max_tilt=max_tilt,
    )
    survived = bool(duration >= 3.0 - 1e-6 and not fallen)
    success = bool(survived and failure_mode == "none_observed")
    rollout = _load_adjacent_json(trajectory, "rollout.json")
    if fallen:
        termination_reason = "fall"
    elif bool(rollout.get("terminated")):
        termination_reason = "terminated"
    elif bool(rollout.get("truncated")):
        termination_reason = "truncated"
    elif survived:
        termination_reason = "horizon_3s"
    else:
        termination_reason = str(rollout.get("stop_reason") or "incomplete")
    protocol_note = "Full raw trajectory used."
    if raw_state_count != times.size:
        protocol_note = (
            f"Protocol prefix only: {times.size}/{raw_state_count} states through first fall; "
            "post-fall raw states retained for audit but excluded here."
        )
    row = {
        "task": str(metadata.get("task", "unknown")),
        "controller": controller,
        "seed": int(metadata.get("seed", 0)),
        "success": success,
        "survived_full_3s_horizon": survived,
        "sim_duration_seconds": duration,
        "survival_steps": int(max(times.size - 1, 0)),
        "tracking_rmse": tracking_rmse,
        "final_tracking_error": float(tracking_finite[-1]) if tracking_finite.size else None,
        "max_tracking_error": float(np.max(tracking_finite)) if tracking_finite.size else None,
        "mean_action_delta": float(np.mean(non_initial_changes)) if non_initial_changes.size else 0.0,
        "max_action_delta": float(np.max(non_initial_changes)) if non_initial_changes.size else 0.0,
        "mean_action_norm": float(np.mean(action_norms)) if action_norms.size else 0.0,
        "first_divergence_time": first_divergence if controller == "gpt_direct" else None,
        "termination_reason": termination_reason,
        "protocol_analyzed_decisions": int(len(_decision_times(arrays, metadata))) if controller == "gpt_direct" else None,
        "raw_state_count": raw_state_count,
        "protocol_state_count": int(times.size),
        "horizontal_displacement": float(np.linalg.norm(actual_delta[:2])),
        "progress_ratio": progress_ratio,
        "root_height_min": float(np.nanmin(arrays["root_height"])),
        "torso_pitch_max_abs_degrees": float(np.max(np.abs(pitch))),
        "torso_roll_max_abs_degrees": float(np.max(np.abs(roll))),
        "right_contact_fraction": float(np.mean(right_contact)),
        "left_contact_fraction": float(np.mean(left_contact)),
        "failure_mode": failure_mode,
        "protocol_note": protocol_note,
        "trajectory": str(trajectory),
    }
    row.update(_timing_metrics(trajectory, controller))
    return row


def summarize_arrays(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible compact summary used by Phase 1 tests."""

    times = np.asarray(arrays["time"], dtype=float).reshape(-1)
    root = np.asarray(arrays["root_xyz"], dtype=float)
    reference = np.asarray(arrays["reference_root_xyz"], dtype=float)
    actual_delta = root[-1] - root[0]
    reference_delta = reference[-1] - reference[0]
    reference_norm_sq = float(np.dot(reference_delta, reference_delta))
    progress_ratio = (
        float(np.dot(actual_delta, reference_delta) / reference_norm_sq)
        if reference_norm_sq > 1e-12
        else float("nan")
    )
    duration = float(times[-1] - times[0])
    tracking_mean = _finite_mean(arrays["tracking_site_rmse"])
    max_tilt = float(np.nanmax(np.asarray(arrays["root_tilt_degrees"], dtype=float)))
    fallen = bool(np.any(arrays["fallen"]))
    failure_mode = classify_failure(
        fallen=fallen,
        progress_ratio=progress_ratio,
        tracking_rmse_mean=tracking_mean,
        max_tilt=max_tilt,
    )
    requested_duration = float(metadata.get("requested_duration", duration))
    completed = duration >= requested_duration - 1e-6
    actions = np.asarray(arrays["commanded_action"], dtype=float)
    return {
        "task": str(metadata.get("task", "unknown")),
        "controller": str(metadata.get("controller", "unknown")),
        "seed": int(metadata.get("seed", 0)),
        "duration_seconds": duration,
        "requested_duration_seconds": requested_duration,
        "states": len(arrays["time"]),
        "horizontal_displacement": float(np.linalg.norm(actual_delta[:2])),
        "vertical_displacement": float(actual_delta[2]),
        "reference_displacement": float(np.linalg.norm(reference_delta)),
        "progress_ratio": progress_ratio,
        "tracking_site_rmse_mean": tracking_mean,
        "root_height_min": float(np.nanmin(arrays["root_height"])),
        "root_tilt_max_degrees": max_tilt,
        "fallen": fallen,
        "command_effort_mean_square": float(np.mean(np.square(actions))) if actions.size else 0.0,
        "command_delta_rms": float(np.sqrt(np.mean(np.square(np.diff(actions, axis=0)))))
        if actions.shape[0] > 1
        else 0.0,
        "success": bool(completed and failure_mode == "none_observed"),
        "failure_mode": failure_mode,
        "single_seed_qualitative_only": True,
    }


def load_rollout(trajectory_path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(trajectory_path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files if key != "metadata_json"}
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    return arrays, metadata


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _plot_task_trace(
    task: str,
    pair: dict[str, tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]],
    output_path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    colors = {"gpt": "#b83280", "residual": "#2b6cb0"}
    for label in ("gpt", "residual"):
        arrays, metadata, row = pair[label]
        times = np.asarray(arrays["time"], dtype=float)
        name = "GPT-6 direct" if label == "gpt" else "Residual"
        axes[0].plot(times, arrays["tracking_site_rmse"], label=name, color=colors[label])
        effective = _effective_actions(arrays, str(metadata["controller"]))
        axes[1].plot(times, normalized_action_change_series(effective), label=name, color=colors[label])
        pitch, roll = quaternion_pitch_roll(arrays["qpos"])
        axes[2].plot(times, pitch, label=f"{name} pitch", color=colors[label])
        axes[2].plot(times, roll, label=f"{name} roll", color=colors[label], linestyle="--")
        right, left = contact_series(arrays["touch"])
        offset = 0.05 if label == "residual" else 0.0
        axes[3].step(times, right + offset, where="post", label=f"{name} right", color=colors[label])
        axes[3].step(times, -left - offset, where="post", label=f"{name} left", color=colors[label], linestyle="--")
        root = np.asarray(arrays["root_xyz"])
        reference = np.asarray(arrays["reference_root_xyz"])
        axes[4].plot(times, root[:, 0] - root[0, 0], label=f"{name} actual x", color=colors[label])
        if label == "gpt":
            axes[4].plot(
                times,
                reference[:, 0] - reference[0, 0],
                label="reference x",
                color="#4a5568",
                linestyle=":",
            )
        if row["termination_reason"] == "fall":
            for axis in axes:
                axis.axvline(times[-1], color=colors[label], linewidth=1.0, linestyle=":", alpha=0.8)
    gpt_arrays, gpt_metadata, _ = pair["gpt"]
    decision_times = _decision_times(gpt_arrays, gpt_metadata)
    hold = float(gpt_metadata.get("hold_seconds", 0.2))
    for index, boundary in enumerate(decision_times):
        for axis in axes:
            axis.axvline(boundary, color="#718096", linewidth=0.6, alpha=0.35)
            if index % 2 == 0:
                axis.axvspan(boundary, boundary + hold, color="#edf2f7", alpha=0.25)
    labels = [
        "site tracking RMSE (m)",
        "normalized action delta",
        "torso angle (deg)",
        "contact (R=+1, L=-1)",
        "forward displacement (m)",
    ]
    for axis, label in zip(axes, labels, strict=True):
        axis.set_ylabel(label)
        axis.grid(True, linestyle=":", alpha=0.35)
        axis.legend(loc="best", fontsize=7, ncol=2)
    axes[-1].set_xlabel("simulation time (s)")
    fig.suptitle(f"{task.title()}: GPT-6 direct vs Residual")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_summary(
    pairs: dict[str, dict[str, tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]]],
    output_path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = ("walk", "run", "stairs")
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex="col")
    for column, task in enumerate(tasks):
        for label, color in (("gpt", "#b83280"), ("residual", "#2b6cb0")):
            arrays, metadata, _ = pairs[task][label]
            times = np.asarray(arrays["time"], dtype=float)
            name = "GPT-6 direct" if label == "gpt" else "Residual"
            axes[0, column].plot(times, arrays["tracking_site_rmse"], label=name, color=color)
            changes = normalized_action_change_series(_effective_actions(arrays, str(metadata["controller"])))
            axes[1, column].plot(times, changes, label=name, color=color)
        axes[0, column].set_title(task.title())
        axes[0, column].set_ylabel("tracking RMSE (m)")
        axes[1, column].set_ylabel("normalized action delta")
        axes[1, column].set_xlabel("simulation time (s)")
        for axis in axes[:, column]:
            axis.grid(True, linestyle=":", alpha=0.35)
            axis.legend(fontsize=8)
    fig.suptitle("GPT-6 Direct Muscle Control — qualitative single-seed comparison")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def video_frame_count(reader: Any, metadata: dict[str, Any]) -> int:
    """Use a finite metadata frame count or ask the decoder to count."""

    raw = metadata.get("nframes", 0)
    try:
        numeric = float(raw)
    except (TypeError, ValueError):
        numeric = 0.0
    if np.isfinite(numeric) and numeric > 0.0:
        return int(numeric)
    return int(reader.count_frames())


def _extract_keyframes(
    video_path: Path,
    output_dir: Path,
    selected: list[tuple[str, float]],
) -> list[str]:
    import imageio.v2 as imageio

    if not video_path.exists():
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(video_path)
    try:
        metadata = reader.get_meta_data()
        fps = float(metadata.get("fps", 30.0))
        frame_count = video_frame_count(reader, metadata)
        outputs: list[str] = []
        used: set[int] = set()
        for order, (label, timestamp) in enumerate(selected):
            index = min(max(int(round(timestamp * fps)), 0), frame_count - 1)
            if index in used:
                continue
            used.add(index)
            destination = output_dir / f"{order:02d}_{label}.png"
            imageio.imwrite(destination, reader.get_data(index))
            outputs.append(str(destination))
        return outputs
    finally:
        reader.close()


def _hold_error_growth(arrays: dict[str, np.ndarray]) -> tuple[int, int]:
    indices = np.asarray(arrays["decision_index"], dtype=int)
    errors = np.asarray(arrays["tracking_site_rmse"], dtype=float)
    grew = total = 0
    for decision in sorted(set(int(value) for value in indices if value >= 0)):
        matches = np.flatnonzero(indices == decision)
        if matches.size > 1 and np.isfinite(errors[matches[[0, -1]]]).all():
            total += 1
            grew += int(errors[matches[-1]] > errors[matches[0]])
    return grew, total


def _mechanism(arrays: dict[str, np.ndarray], row: dict[str, Any]) -> str:
    grew, total = _hold_error_growth(arrays)
    fraction = grew / total if total else 0.0
    if row["max_action_delta"] and row["max_action_delta"] > 0.25:
        return "large inter-decision action discontinuity followed by balance loss"
    if fraction >= 0.6:
        return "tracking drift accumulated during multiple 0.2-s action holds"
    return "tracking and balance drift without evidence for a single dominant mechanism"


def _write_report(
    pairs: dict[str, dict[str, tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]]],
    output_path: Path,
    frame_paths: dict[str, list[str]],
    timing_caveat: dict[str, Any],
) -> None:
    lines = [
        "# Experimental setting",
        "",
        "This experiment is a qualitative, latency-decoupled direct-control diagnostic.",
        "The simulator is paused during Codex/GPT-6 Astra inference.",
        "GPT-generated muscle actions are updated at 5 Hz in simulation time and held for 0.2 s.",
        "Only one seed and a maximum 3 s simulated horizon are evaluated for walking, running, and stair climbing.",
        "The experiment is not intended as a statistical benchmark.",
        "",
        "First divergence is defined as the first five consecutive native control samples for which GPT tracking error",
        "exceeds both the paired Residual value and the Residual trajectory's 95th percentile. If this condition is not",
        "met, the value is reported as N/A. This fixed, data-derived rule avoids selecting a task-specific threshold.",
        "",
    ]
    for task in ("walk", "run", "stairs"):
        g_arrays, _, g = pairs[task]["gpt"]
        _, _, r = pairs[task]["residual"]
        grew, holds = _hold_error_growth(g_arrays)
        divergence = "N/A" if g["first_divergence_time"] is None else f"{g['first_divergence_time']:.3f} s"
        lines.extend(
            [
                f"# {task.title()}",
                "",
                f"- Did it survive 3 s? **{g['survived_full_3s_horizon']}** (Residual: {r['survived_full_3s_horizon']}).",
                f"- Did it satisfy task success? **{g['success']}** (Residual: {r['success']}).",
                f"- First sustained divergence: **{divergence}**.",
                f"- Tracking error: final {g['final_tracking_error']:.4f} m, maximum {g['max_tracking_error']:.4f} m, "
                f"trajectory RMSE {g['tracking_rmse']:.4f} m.",
                f"- Action continuity: mean normalized delta {g['mean_action_delta']:.4f}, maximum {g['max_action_delta']:.4f}.",
                f"- Error increased within {grew}/{holds} analyzed 0.2-s holds.",
                f"- Foot contact duty fractions: right {g['right_contact_fraction']:.3f}, left {g['left_contact_fraction']:.3f}.",
                f"- Torso excursion: |pitch| max {g['torso_pitch_max_abs_degrees']:.2f} deg; "
                f"|roll| max {g['torso_roll_max_abs_degrees']:.2f} deg.",
                f"- Recovery assessment: {_mechanism(g_arrays, g)}.",
                f"- Termination: `{g['termination_reason']}` at {g['sim_duration_seconds']:.3f} s.",
                f"- Protocol note: {g['protocol_note']}",
                f"- Extracted frames: {', '.join(frame_paths.get(task, [])) or 'N/A'}",
                "",
            ]
        )
    lines.extend(
        [
            "# Cross-task observations",
            "",
            "Under the tested 5-Hz direct-action interface, all three GPT-direct cases reached a logged fall before the",
            "3-s horizon. The paired Residual controller completed the Walk horizon but also fell early in the Run and",
            "Stairs cases, so the observed contrast is task-dependent rather than a blanket controller ranking.",
            "The traces expose action-hold boundaries, tracking drift, torso motion, and contact changes without treating",
            "wall-clock inference latency as a causal simulator delay.",
            "",
            "# Timing and audit caveats",
            "",
            "Wall-clock values are raw observation-export-to-action-receipt elapsed times. The simulator was paused throughout",
            "each interval. Run decision index 7 includes a multi-hour user/session pause and is not a pure inference-time",
            "measurement; the raw value remains in summary.csv, with a pause-excluded mean in an audit column.",
            f"Recorded timing caveat: {json.dumps(timing_caveat, ensure_ascii=False)}",
            "",
            "Walk contains a pre-repair logging deviation: its raw file continued beyond the first fall. All Phase 3 plots,",
            "metrics, video, and frames use only the inclusive prefix through the first fall; the raw file remains unchanged.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_rollouts(trajectory_paths: Iterable[str | Path], output_dir: str | Path) -> list[dict[str, Any]]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, dict[str, tuple[dict[str, np.ndarray], dict[str, Any], Path, int]]] = {}
    for item in trajectory_paths:
        path = Path(item)
        arrays, metadata = load_rollout(path)
        task = str(metadata.get("task"))
        label = "gpt" if str(metadata.get("controller")) == "gpt_direct" else "residual"
        if task not in ("walk", "run", "stairs") or label in loaded.setdefault(task, {}):
            raise ValueError(f"Expected one GPT and one Residual trajectory per task; got duplicate/unknown: {path}")
        raw_count = len(arrays["time"])
        if label == "gpt":
            arrays = trim_arrays_at_first_fall(arrays)
        loaded[task][label] = (arrays, metadata, path, raw_count)
    expected = {(task, label) for task in ("walk", "run", "stairs") for label in ("gpt", "residual")}
    present = {(task, label) for task, values in loaded.items() for label in values}
    if present != expected:
        raise ValueError(f"Expected six paired rollouts; missing={sorted(expected - present)}")

    rows: list[dict[str, Any]] = []
    pairs: dict[str, dict[str, tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]]] = {}
    for task in ("walk", "run", "stairs"):
        g_arrays, g_meta, g_path, g_raw = loaded[task]["gpt"]
        r_arrays, r_meta, r_path, r_raw = loaded[task]["residual"]
        divergence = first_sustained_divergence(
            g_arrays["time"],
            g_arrays["tracking_site_rmse"],
            r_arrays["time"],
            r_arrays["tracking_site_rmse"],
        )
        g_row = _trajectory_metrics(
            g_arrays,
            g_meta,
            g_path,
            raw_state_count=g_raw,
            first_divergence=divergence,
        )
        r_row = _trajectory_metrics(
            r_arrays,
            r_meta,
            r_path,
            raw_state_count=r_raw,
            first_divergence=None,
        )
        trace = destination / "figures" / f"{task}_failure_trace.png"
        g_row["failure_trace"] = str(trace)
        r_row["failure_trace"] = str(trace)
        pairs[task] = {"gpt": (g_arrays, g_meta, g_row), "residual": (r_arrays, r_meta, r_row)}
        _plot_task_trace(task, pairs[task], trace)
        rows.extend((g_row, r_row))
    _plot_summary(pairs, destination / "figures" / "gpt6_direct_control_summary.png")

    frame_paths: dict[str, list[str]] = {}
    for task in ("walk", "run", "stairs"):
        arrays, metadata, row = pairs[task]["gpt"]
        selected = select_keyframe_times(
            duration=float(row["sim_duration_seconds"]),
            first_divergence=row["first_divergence_time"],
            decision_times=_decision_times(arrays, metadata),
            fallen=row["termination_reason"] == "fall",
        )
        video = destination / "videos" / f"{task}_gpt6_seed0.mp4"
        frame_paths[task] = _extract_keyframes(video, destination / "failure_frames" / task, selected)

    clean_rows = [{key: _safe_json_value(value) for key, value in row.items()} for row in rows]
    atomic_write_json(destination / "summary.json", clean_rows)
    fields = SUMMARY_COLUMNS + EXTRA_COLUMNS
    with (destination / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in clean_rows)
    phase2 = destination / "PHASE2_STATUS.json"
    phase2_status = json.loads(phase2.read_text(encoding="utf-8")) if phase2.exists() else {}
    timing_caveat = dict(phase2_status.get("timing_caveat", {}))
    _write_report(
        pairs,
        destination / "QUALITATIVE_FAILURE_REPORT.md",
        frame_paths,
        timing_caveat,
    )
    return clean_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = analyze_rollouts(args.trajectory, args.output_dir)
    print(f"ANALYSIS_OK rows={len(rows)} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
