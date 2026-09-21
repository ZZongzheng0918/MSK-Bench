"""Offline MP4 rendering from persisted MuJoCo states (never policy execution)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from msk_bench.validation import validate_video

from .paused_env_controller import make_direct_env, normalize_task


def select_frame_indices(times: np.ndarray, fps: int) -> np.ndarray:
    """Select nearest recorded states on an evenly spaced simulation-time grid."""

    values = np.asarray(times, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Trajectory contains no timestamps.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if not bool(np.all(np.isfinite(values))) or bool(np.any(np.diff(values) < 0.0)):
        raise ValueError("Timestamps must be finite and monotonically non-decreasing.")
    start, end = float(values[0]), float(values[-1])
    targets = np.arange(start, end + 0.5 / fps, 1.0 / fps)
    candidates = np.searchsorted(values, targets, side="left")
    candidates = np.clip(candidates, 0, values.size - 1)
    for index, candidate in enumerate(candidates):
        if candidate > 0 and abs(values[candidate - 1] - targets[index]) <= abs(values[candidate] - targets[index]):
            candidates[index] = candidate - 1
    return np.unique(np.concatenate(([0], candidates, [values.size - 1]))).astype(int)


def trim_arrays_at_first_fall(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return an inclusive prefix ending at the first logged fallen sample."""

    times = np.asarray(arrays["time"])
    fallen = np.asarray(arrays.get("fallen", np.zeros(times.size, dtype=bool)), dtype=bool).reshape(-1)
    if fallen.size != times.size:
        raise ValueError("fallen and time arrays must have the same length.")
    indices = np.flatnonzero(fallen)
    stop = int(indices[0]) + 1 if indices.size else times.size
    return {
        key: np.asarray(value)[:stop] if np.asarray(value).ndim and np.asarray(value).shape[0] == times.size else np.asarray(value)
        for key, value in arrays.items()
    }


def _load(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files if key != "metadata_json"}
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    return arrays, metadata


def render_trajectory(
    trajectory_path: str | Path,
    output_path: str | Path,
    *,
    task: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    camera: int | str | None = None,
    stop_at_first_fall: bool = False,
) -> Path:
    """Restore each saved state and render it without advancing simulation."""

    from benchmark_eval.common import configure_headless_rendering

    configure_headless_rendering()
    import imageio.v2 as imageio
    import mujoco

    source = Path(trajectory_path)
    arrays, metadata = _load(source)
    if stop_at_first_fall:
        arrays = trim_arrays_at_first_fall(arrays)
    selected_task = normalize_task(task or metadata["task"])
    required = ("time", "qpos", "qvel", "act", "ctrl")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"Trajectory is missing required state arrays: {missing}")
    indices = select_frame_indices(arrays["time"], fps)
    env = make_direct_env(selected_task)
    renderer = None
    frames: list[np.ndarray] = []
    try:
        env.reset(seed=int(metadata.get("seed", 0)))
        data = env._data
        model = env._model
        if arrays["qpos"].shape[1] != model.nq or arrays["qvel"].shape[1] != model.nv:
            raise ValueError("Saved state dimensions do not match the task model.")
        renderer = mujoco.Renderer(model, height=height, width=width)
        for index in indices:
            data.qpos[:] = arrays["qpos"][index]
            data.qvel[:] = arrays["qvel"][index]
            if data.act.size:
                data.act[:] = arrays["act"][index]
            if data.ctrl.size:
                data.ctrl[:] = arrays["ctrl"][index]
            data.time = float(arrays["time"][index])
            mujoco.mj_forward(model, data)
            if camera is None:
                renderer.update_scene(data)
            else:
                renderer.update_scene(data, camera=camera)
            frame = np.asarray(renderer.render(), dtype=np.uint8)
            if frame.shape != (height, width, 3):
                raise RuntimeError(f"Unexpected rendered frame shape: {frame.shape}")
            frames.append(frame.copy())
    finally:
        if renderer is not None:
            renderer.close()
        env.close()
    if not frames:
        raise RuntimeError("No frames selected for rendering.")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(destination, frames, fps=fps, macro_block_size=1)
    validate_video(destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("walk", "run", "stairs"), default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera", default=None)
    parser.add_argument(
        "--stop-at-first-fall",
        action="store_true",
        help="Render only the inclusive protocol prefix ending at the first fallen sample.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    camera: int | str | None = args.camera
    if isinstance(camera, str) and camera.lstrip("-").isdigit():
        camera = int(camera)
    result = render_trajectory(
        args.trajectory,
        args.output,
        task=args.task,
        width=args.width,
        height=args.height,
        fps=args.fps,
        camera=camera,
        stop_at_first_fall=args.stop_at_first_fall,
    )
    print(f"VIDEO_OK {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
