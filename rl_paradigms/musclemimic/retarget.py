from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


class InterceptSuccess(Exception):
    """Carry intercepted qpos data out of MuscleMimic's retargeting pipeline."""

    def __init__(self, qpos):
        self.qpos = qpos


def safe_intercept(qpos, fps, free_joint_name, model):
    print("\nCaptured the 89-dimensional pose data before velocity conversion.")
    raise InterceptSuccess(qpos)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retarget an AMASS motion for the MuscleMimic stair prior.")
    parser.add_argument(
        "--amass-root",
        type=Path,
        default=Path(os.environ["AMASS_PATH"]) if os.environ.get("AMASS_PATH") else None,
        help="AMASS dataset root. Defaults to the AMASS_PATH environment variable.",
    )
    parser.add_argument(
        "--motion-data",
        type=Path,
        default=None,
        help="Input AMASS .npz file. Defaults to KIT/316/13_35_poses.npz under --amass-root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "stair_prior_89d.npz",
        help="Output .npz path. Defaults next to this script.",
    )
    return parser


def resolve_paths(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[Path, Path, Path]:
    if args.amass_root is None:
        parser.error("--amass-root is required when AMASS_PATH is not set")

    amass_root = args.amass_root.expanduser().resolve()
    motion_data = (args.motion_data or amass_root / "KIT" / "316" / "13_35_poses.npz").expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not amass_root.is_dir():
        raise FileNotFoundError(f"AMASS dataset root does not exist: {amass_root}")
    if not motion_data.is_file():
        raise FileNotFoundError(f"AMASS motion file does not exist: {motion_data}")
    return amass_root, motion_data, output


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    amass_root, motion_data, output = resolve_paths(args, parser)
    os.environ["AMASS_PATH"] = str(amass_root)

    import loco_mujoco.smpl.retargeting as retargeting
    from loco_mujoco.smpl.retargeting import fit_gmr_motion, load_robot_conf_file

    retargeting._compute_qvel_from_qpos = safe_intercept
    logger = logging.getLogger()
    env_name = "MyoFullBody"
    robot_conf = load_robot_conf_file(env_name)
    gmr_config = {"target_fps": 100, "offset_to_ground": False}

    try:
        fit_gmr_motion(env_name, robot_conf, str(motion_data), logger, gmr_config)
    except InterceptSuccess as exc:
        final_qpos = exc.qpos
        dummy_qvel = np.zeros((final_qpos.shape[0], 88))
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, states=final_qpos, qvel=dummy_qvel)
        print(f"Saved the retargeted stair prior to: {output}")
        return 0

    raise RuntimeError("Retargeting completed without producing intercepted pose data")


if __name__ == "__main__":
    raise SystemExit(main())
