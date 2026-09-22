"""Run the existing Residual wrapper with the official pretrained base policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .action_io import atomic_write_json
from .paused_env_controller import DEFAULT_MAX_DECISIONS, TASKS, make_direct_env, run_rollout


def resolve_checkpoint(source: str | None) -> str:
    """Resolve a local path or the official hf:// source to one Orbax checkpoint."""

    from musclemimic.integrations.msk_bench import resolve_checkpoint_source
    from musclemimic.runner.checkpointing import _canonicalize_resume_path

    return _canonicalize_resume_path(resolve_checkpoint_source(source))


def run_residual_task(
    task: str,
    output_dir: str | Path,
    *,
    checkpoint: str,
    decisions: int = DEFAULT_MAX_DECISIONS,
):
    if decisions < 1 or decisions > DEFAULT_MAX_DECISIONS:
        raise ValueError(f"decisions must be in [1, {DEFAULT_MAX_DECISIONS}].")
    probe = make_direct_env(task)
    try:
        action_shape = probe.action_space.shape
    finally:
        probe.close()
    zero_residuals = [np.zeros(action_shape, dtype=np.float32) for _ in range(decisions)]
    result = run_rollout(
        task,
        zero_residuals,
        output_dir,
        controller="residual_pretrained_base_zero_correction",
        checkpoint=checkpoint,
    )
    atomic_write_json(
        Path(output_dir) / "residual_actions.json",
        {
            "description": "Zero residual corrections; actual pretrained muscle commands are stored as ctrl in trajectory.npz.",
            "checkpoint": checkpoint,
            "decisions": [[float(value) for value in action] for action in zero_residuals],
        },
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--checkpoint", default=None, help="Local checkpoint path or hf:// source; default is official.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decisions", type=int, default=DEFAULT_MAX_DECISIONS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = resolve_checkpoint(args.checkpoint)
    tasks = TASKS if args.task == "all" else (args.task,)
    for task in tasks:
        result = run_residual_task(
            task,
            args.output_dir / f"residual_{task}",
            checkpoint=checkpoint,
            decisions=args.decisions,
        )
        print(f"RESIDUAL_OK task={task} steps={result.native_steps} trajectory={result.trajectory_path}")
    print(json.dumps({"checkpoint": checkpoint, "tasks": list(tasks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
