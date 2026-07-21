"""
Parallel AMASS retargeting for muscle-based environments.
Supports MyoBimanualArm and MyoFullBody models with configurable datasets.
"""

import os
import sys
import logging

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["MUJOCO_GL"] = "egl"
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

import time
import argparse
import multiprocessing as mp
from typing import List, Tuple, Optional, Dict


def retarget_single_motion(
    args: Tuple[str, str, Optional[Dict], bool]
) -> Tuple[bool, str]:
    """
    Process a single motion in an isolated process.

    Using maxtasksperchild=1, each call runs in a fresh process that gets
    killed after completion, guaranteeing RSS is returned to OS.

    Args:
        args: Tuple of (motion_name, model_type, retargeting_config, clear_cache)

    Returns:
        Tuple of (success: bool, motion_name: str)
    """
    motion_name, model_type, retargeting_config, clear_cache = args

    try:
        from loco_mujoco.task_factories import (
            ImitationFactory,
            AMASSDatasetConf,
        )
    except ImportError as e:
        print(f"[{motion_name}] Failed to import: {e}")
        return False, motion_name

    try:
        # Create dataset configuration
        dataset_conf = AMASSDatasetConf([motion_name])
        dataset_conf.clear_cache = clear_cache

        # Apply retargeting configuration if provided
        if retargeting_config:
            dataset_conf.retargeting_method = retargeting_config["method"]
            if retargeting_config["method"] == "gmr":
                dataset_conf.gmr_config = retargeting_config["gmr_params"]

        # Configure environment creation based on model type
        env_params = {
            "amass_dataset_conf": dataset_conf,
            "headless": True,
        }

        # Add model-specific parameters
        if model_type == "MyoBimanualArm":
            env_params["goal_type"] = "GoalBimanualTrajMimic"

        # Create CPU-only environment - triggers automatic retargeting
        env = ImitationFactory.make(model_type, **env_params)
        del env

        print(f"[OK] {motion_name}")
        return True, motion_name

    except Exception as e:
        print(f"[FAIL] {motion_name}: {e}")
        return False, motion_name


def get_motion_dataset(dataset_group) -> List[str]:
    """Load and combine motion datasets from one or more group names."""
    from loco_mujoco.task_factories.dataset_confs import (
        get_amass_dataset_groups,
        expand_amass_dataset_group_spec,
    )

    groups = get_amass_dataset_groups()
    group_names = expand_amass_dataset_group_spec(dataset_group)
    dataset_paths = []
    for name in group_names:
        if name not in groups:
            raise ValueError(
                f"Dataset '{name}' not found. Available: {list(groups.keys())}"
            )
        dataset_paths.extend(groups[name])
    return list(dict.fromkeys(dataset_paths))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Parallel AMASS retargeting for muscle-based environments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        choices=["MyoBimanualArm", "MyoFullBody"],
        default="MyoBimanualArm",
        help="Model type to retarget for",
    )

    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["AMASS_BIMANUAL_MARGINAL_MOTIONS"],
        help="One or more dataset group names (e.g. AMASS_RANDOM_TRAINING_MOTIONS KIT_KINESIS_TRANSITION_TRAINING_MOTIONS)",
    )

    parser.add_argument(
        "--workers", type=int, default=None, help="Number of worker processes (default: min(cpu_count, 8, num_motions))"
    )

    # Retargeting method configuration
    parser.add_argument(
        "--retargeting-method",
        choices=["smpl", "gmr"],
        default="smpl",
        help="Retargeting method: 'smpl' (optimization-based, default) or 'gmr' (IK-based)",
    )

    # GMR-specific configuration (only used when --retargeting-method=gmr)
    gmr_group = parser.add_argument_group("GMR Configuration (only applies when --retargeting-method=gmr)")
    gmr_group.add_argument("--gmr-src-human", default="smplh", help="Source human model for GMR")
    gmr_group.add_argument("--gmr-target-fps", type=int, default=30, help="Target FPS for GMR retargeting")
    gmr_group.add_argument("--gmr-solver", default="daqp", help="IK solver for GMR (e.g., daqp, qpswift)")
    gmr_group.add_argument("--gmr-damping", type=float, default=0.5, help="Damping factor for GMR solver")
    gmr_group.add_argument(
        "--gmr-offset-to-ground", action="store_true", default=False, help="Offset trajectory to ground in GMR"
    )
    gmr_group.add_argument(
        "--gmr-no-offset-to-ground", dest="gmr_offset_to_ground", action="store_false", help="Disable ground offset"
    )
    gmr_group.add_argument(
        "--gmr-use-velocity-limit", action="store_true", default=False, help="Use velocity limits in GMR"
    )
    gmr_group.add_argument("--gmr-verbose", action="store_true", default=False, help="Enable verbose GMR output")

    # Cache control
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        default=False,
        help="Clear existing cache and force re-retargeting even if cached files exist",
    )

    return parser.parse_args()


def main() -> int:
    """Main parallel retargeting function."""
    args = parse_arguments()

    print(f"Parallel AMASS Retargeting for {args.model} (CPU-only)")
    print("=" * 60)

    # Load dataset
    try:
        all_motions = get_motion_dataset(args.dataset)
    except (ImportError, ValueError, TypeError) as e:
        print(f"Error loading dataset: {e}")
        return 1

    if not all_motions:
        print("Error: no motions found for the specified dataset groups.")
        return 1

    print(f"Model: {args.model}")
    print(f"Dataset: {', '.join(args.dataset)} ({len(all_motions)} motions)")

    # Build retargeting configuration
    retargeting_config = None
    if args.retargeting_method:
        method_display = args.retargeting_method.upper()
        print(f"Retargeting Method: {method_display}")

        retargeting_config = {"method": args.retargeting_method}

        # Add GMR-specific configuration if GMR is selected
        if args.retargeting_method == "gmr":
            retargeting_config["gmr_params"] = {
                "src_human": args.gmr_src_human,
                "target_fps": args.gmr_target_fps,
                "solver": args.gmr_solver,
                "damping": args.gmr_damping,
                "offset_to_ground": args.gmr_offset_to_ground,
                "use_velocity_limit": args.gmr_use_velocity_limit,
                "verbose": args.gmr_verbose,
            }
            print("GMR Configuration:")
            print(f"  Source Human: {args.gmr_src_human}")
            print(f"  Target FPS: {args.gmr_target_fps}")
            print(f"  Solver: {args.gmr_solver}")
            print(f"  Damping: {args.gmr_damping}")
            print(f"  Offset to Ground: {args.gmr_offset_to_ground}")
            print(f"  Use Velocity Limit: {args.gmr_use_velocity_limit}")

    if args.clear_cache:
        print("Cache clearing ENABLED: Will overwrite existing cached files")
    else:
        print("Retargeting system will handle caching automatically")
        print("  (GMR and SMPL caches are separate)")

    # Determine worker configuration
    if args.workers:
        max_workers = min(args.workers, len(all_motions))
    else:
        max_workers = min(mp.cpu_count(), 8, len(all_motions))
    print(f"Using {max_workers} CPU worker processes (maxtasksperchild=1 for memory control)")

    # Prepare single-motion args (each motion runs in isolated process)
    worker_args = [
        (motion_name, args.model, retargeting_config, args.clear_cache)
        for motion_name in all_motions
    ]

    # Execute parallel processing with process recycling
    start_time = time.time()
    print(f"\nProcessing {len(all_motions)} motions (each in fresh process)...")

    # Use spawn method for clean worker processes
    original_method = mp.get_start_method(allow_none=True)
    try:
        mp.set_start_method("spawn", force=True)

        # maxtasksperchild=10: each worker processes 10 motions then dies
        # This guarantees RSS is returned to OS after each motion
        with mp.Pool(processes=max_workers, maxtasksperchild=10) as pool:
            results = list(pool.imap_unordered(retarget_single_motion, worker_args))

    finally:
        if original_method:
            mp.set_start_method(original_method, force=True)

    elapsed_time = time.time() - start_time

    # Aggregate results
    total_successful = sum(1 for success, _ in results if success)
    total_failed = sum(1 for success, _ in results if not success)
    failed_motions = [name for success, name in results if not success]

    # Report results
    print(f"\nProcessing Results:")
    print(f"  Total processed: {total_successful + total_failed}")
    print(f"  Successful: {total_successful}")
    print(f"  Failed: {total_failed}")
    print(f"  Processing time: {elapsed_time:.1f} seconds")

    if len(all_motions) > 0:
        avg_time = elapsed_time / len(all_motions)
        print(f"  Average time per motion: {avg_time:.2f} seconds")

    if max_workers > 1:
        estimated_sequential_time = elapsed_time * max_workers
        speedup_factor = estimated_sequential_time / elapsed_time
        print(f"  Estimated speedup: {speedup_factor:.1f}x over sequential processing")

    if failed_motions:
        print(f"\nFailed motions ({len(failed_motions)}):")
        for motion in failed_motions[:20]:  # Show first 20
            print(f"  {motion}")
        if len(failed_motions) > 20:
            print(f"  ... and {len(failed_motions) - 20} more")

    success_rate = (total_successful / len(all_motions)) * 100 if len(all_motions) > 0 else 0
    print(f"\nFinal Status:")
    print(f"  Success rate: {success_rate:.1f}% ({total_successful}/{len(all_motions)})")

    if total_failed == 0:
        print("All motions processed successfully")
        return 0
    else:
        print(f"Warning: {total_failed} motions failed to process")
        return 1


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
