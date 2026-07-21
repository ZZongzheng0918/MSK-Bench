import os
from pathlib import Path
import argparse
import time
import multiprocessing as mp
from typing import Dict, List, Tuple
from collections import defaultdict
import pandas as pd
from musclemimic.utils.retarget.msk_metrics import compute_msk_metrics, _get_base_and_gmr_dirs


# ================== THRESHOLDS ==================
TH_JOINT = 0.5     # %
TH_PEN_PCT = 1      # % frames
TH_PEN_MAX = 0.005    # meters (5 mm)
TH_FLOAT = 0.025   # meters (2.5 cm), the radius of the foot collision capsule is already 2cm
TH_TEND  = 0.05   # %
FAIL_SUMMARY = defaultdict(lambda: defaultdict(set))

def _record_fail(summary, method: str, category: str, motion_rel: str):
    if method not in summary:
        summary[method] = {}
    if category not in summary[method]:
        summary[method][category] = set()
    summary[method][category].add(motion_rel)

def _update_fail_summary_from_result(summary, method: str, motion_rel: str, r: Dict):
    # Only count a motion for a category if it crosses the threshold
    if r.get(f"{method}_joint_pct") is not None and r[f"{method}_joint_pct"] > TH_JOINT:
        _record_fail(summary, method.upper(), "joint violation", motion_rel)

    if r.get(f"{method}_pen_max_depth_m") is not None and r[f"{method}_pen_max_depth_m"] > TH_PEN_MAX:
        _record_fail(summary, method.upper(), "penetration max depth", motion_rel)

    if r.get(f"{method}_float_height") is not None and r[f"{method}_float_height"] > TH_FLOAT:
        _record_fail(summary, method.upper(), "floating", motion_rel)

    if r.get(f"{method}_tendon_max_jump_%") is not None and r[f"{method}_tendon_max_jump_%"] > TH_TEND:
        _record_fail(summary, method.upper(), "tendon jump", motion_rel)

def new_fail_summary():
    return {}

def merge_fail_summaries(dst, src):
    for method, cats in src.items():
        for cat, motions in cats.items():
            dst[method][cat].update(motions)

def print_fail_summary(summary):
    print("\n" + "=" * 60)
    print("Failure summary (count = #motions)")
    print("=" * 60)
    for method in sorted(summary.keys()):
        print(f"\n{method}:")
        for cat in sorted(summary[method].keys()):
            print(f"  {cat}: {len(summary[method][cat])}")
    print("=" * 60 + "\n")

def collect_dataset_files(dataset_name: str) -> List[Path]:
    """
    Resolve dataset motion list (relative paths) to absolute KIT npz paths.
    """
    motion_list = get_motion_dataset(dataset_name)
    base, _ = _get_base_and_gmr_dirs()

    files: List[Path] = []
    for rel in motion_list:
        p = base / rel
        if p.suffix != ".npz":
            p = p.with_suffix(".npz")
        files.append(p)

    return files

def violates_gmr(r: Dict) -> bool:
    return (
        (r["gmr_joint_pct"] is not None and r["gmr_joint_pct"] > TH_JOINT)
        or (r["gmr_pen_max_depth_m"] is not None and r["gmr_pen_max_depth_m"] > TH_PEN_MAX)
        or (r["gmr_float_height"] is not None and r["gmr_float_height"] > TH_FLOAT)
        or (r["gmr_tendon_max_jump_%"] is not None and r["gmr_tendon_max_jump_%"] > TH_TEND)
    )


def violates_mimic(r: Dict) -> bool:
    return (
        (r["mimic_joint_pct"] is not None and r["mimic_joint_pct"] > TH_JOINT)
        or (r["mimic_pen_max_depth_m"] is not None and r["mimic_pen_max_depth_m"] > TH_PEN_MAX)
        or (r["mimic_float_height"] is not None and r["mimic_float_height"] > TH_FLOAT)
        or (r["mimic_tendon_max_jump_%"] is not None and r["mimic_tendon_max_jump_%"] > TH_TEND)
    )

def worker_process(args: Tuple[int, List[Path]]) -> Tuple[
    int, int, List[Dict], List[Dict], List[Dict], List[str], List[str], Dict
]:
    worker_id, batch = args
    print(f"Worker {worker_id}: {len(batch)} motions")
    local_fail_summary = {}

    base, gmr_dir = _get_base_and_gmr_dirs()

    success = 0
    missing = 0
    all_records: List[Dict] = []
    gmr_bad: List[Dict] = []
    mimic_bad: List[Dict] = []
    errors: List[str] = []
    missing_files: List[str] = []

    for i, f in enumerate(batch):
        rel = f.relative_to(base)
        gmr_file = gmr_dir / rel

        if not f.exists() or not gmr_file.exists():
            print(f"Worker {worker_id}: Missing files for {rel}, skipping")
            missing += 1
            missing_files.append(str(rel))
            continue

        try:
            print(f"Worker {worker_id}: [{i+1}/{len(batch)}] {rel}")

            metrics = compute_msk_metrics(rel)

            result = {
                "file": str(rel),

                "gmr_joint_pct": metrics["gmr"].get("joint_pct"),
                "gmr_pen_pct": metrics["gmr"].get("pen_pct"),
                "gmr_pen_max_depth_m": metrics["gmr"].get("pen_max_depth_m"),
                "gmr_float_height": metrics["gmr"].get("float_height"),
                "gmr_tendon_max_jump_%": metrics["gmr"].get("tendon_max_jump_%"),
                "gmr_sec_per_frame": metrics["gmr"].get("sec_per_frame"),
                "gmr_pos_rmse": metrics["gmr"].get("pos_rmse"),

                "mimic_joint_pct": metrics["mimic"].get("joint_pct"),
                "mimic_pen_pct": metrics["mimic"].get("pen_pct"),
                "mimic_pen_max_depth_m": metrics["mimic"].get("pen_max_depth_m"),
                "mimic_float_height": metrics["mimic"].get("float_height"),
                "mimic_tendon_max_jump_%": metrics["mimic"].get("tendon_max_jump_%"),
                "mimic_sec_per_frame": metrics["mimic"].get("sec_per_frame"),
                "mimic_pos_rmse": metrics["mimic"].get("pos_rmse"),
            }

            success += 1
            all_records.append(result)

            motion_rel = str(rel)

            # populate per-category failure summary (by motion)
            _update_fail_summary_from_result(local_fail_summary, "gmr", motion_rel, result)
            _update_fail_summary_from_result(local_fail_summary, "mimic", motion_rel, result)

            if violates_gmr(result):
                gmr_bad.append(result)
            if violates_mimic(result):
                mimic_bad.append(result)

        except Exception as e:
            errors.append(f"{rel}: {e!r}")

    return success, missing, all_records, gmr_bad, mimic_bad, errors, missing_files, local_fail_summary


def distribute(lst: List[Path], num_workers: int) -> List[List[Path]]:
    batches = [[] for _ in range(num_workers)]
    for i, item in enumerate(lst):
        batches[i % num_workers].append(item)
    return batches

def get_motion_dataset(dataset_name: str) -> List[str]:
    """Load the specified motion dataset."""
    try:
        from loco_mujoco import smpl

        if hasattr(smpl.const, dataset_name):
            return list(getattr(smpl.const, dataset_name))
        else:
            raise ValueError(f"Dataset '{dataset_name}' not found in loco_mujoco.smpl.const")
    except ImportError as e:
        raise ImportError(f"Failed to import dataset {dataset_name}: {e}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--dataset",
        choices=[
            "AMASS_BIMANUAL_MARGINAL_MOTIONS",
            "KIT_KINESIS_TRAINING_MOTIONS",
            "AMASS_BIMANUAL_TRAIN_MOTIONS",
            "AMASS_BIMANUAL_TEST_MOTIONS",
            "KIT_KINESIS_TRANSITION_TRAINING_MOTIONS",
            "AMASS_TRANSITION_MOTIONS",
            "ACCAD_TRAINING_MOTIONS"
        ],
        default="KIT_KINESIS_TRAINING_MOTIONS",
        help="Dataset constant name from loco_mujoco.smpl.const",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    motion_files = collect_dataset_files(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"Found {len(motion_files)} motions")

    if not motion_files:
        print("No files found.")
        return 0

    num_workers = min(args.workers or mp.cpu_count(), 100, len(motion_files))
    print(f"Using {num_workers} workers")

    batches = distribute(motion_files, num_workers)
    worker_args = list(enumerate(batches))

    start = time.time()

    with mp.get_context("spawn").Pool(num_workers) as pool:
        results = pool.map(worker_process, worker_args)

    elapsed = time.time() - start

    total_success = sum(r[0] for r in results)
    total_missing = sum(r[1] for r in results)
    all_records   = [item for r in results for item in r[2]]
    gmr_failed    = [item for r in results for item in r[3]]
    mimic_failed  = [item for r in results for item in r[4]]
    all_errors    = [msg  for r in results for msg in r[5]]
    missing_files = [p for r in results for p in r[6]]

    print(f"\n=== RESULTS (PRE-SAVE) ===")
    print(f"Total discovered: {len(motion_files)}")
    print(f"Processed OK:     {total_success}")
    print(f"Missing pairs:    {total_missing}")
    print(f"Errors:           {len(all_errors)}")
    print(f"Time: {elapsed:.2f}s")

    if missing_files:
        print("\n=== MISSING (NOT COMPARED) ===")
        # Print one path per line.
        for p in missing_files:
            print(p)

    # Abort output generation if any worker failed.
    if all_errors:
        # Print the first error only.
        print("\nERROR DETECTED — not writing CSV outputs.")
        print(all_errors[0])
        return 1

    # ================== WRITE OUTPUTS ==================
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gmr_cols = [
        "file",
        "gmr_joint_pct",
        "gmr_pen_pct",
        "gmr_pen_max_depth_m",
        "gmr_float_height",
        "gmr_tendon_max_jump_%",
        "gmr_sec_per_frame",
        "gmr_pos_rmse",
    ]

    mimic_cols = [
        "file",
        "mimic_joint_pct",
        "mimic_pen_pct",
        "mimic_pen_max_depth_m",
        "mimic_float_height",
        "mimic_tendon_max_jump_%",
        "mimic_sec_per_frame",
        "mimic_pos_rmse",
    ]

    all_cols = [
        "file",
        "gmr_joint_pct",
        "gmr_pen_pct",
        "gmr_pen_max_depth_m",
        "gmr_float_height",
        "gmr_tendon_max_jump_%",
        "gmr_pos_rmse",
        "gmr_sec_per_frame",
        "mimic_joint_pct",
        "mimic_pen_pct",
        "mimic_pen_max_depth_m",
        "mimic_float_height",
        "mimic_tendon_max_jump_%",
        "mimic_sec_per_frame",   
        "mimic_pos_rmse",
    ]

    df_gmr = pd.DataFrame(gmr_failed, columns=gmr_cols)
    df_mimic = pd.DataFrame(mimic_failed, columns=mimic_cols)
    df_all = pd.DataFrame(all_records, columns=all_cols)

    missing_counts = df_all.isna().sum().sort_values(ascending=False)
    print("\n=== MISSING COUNTS PER COLUMN ===")
    print(missing_counts[missing_counts > 0])

    gmr_tendon_motion_pct = 100.0 * (df_all["gmr_tendon_max_jump_%"] > TH_TEND).mean()
    mimic_tendon_motion_pct = 100.0 * (df_all["mimic_tendon_max_jump_%"] > TH_TEND).mean()

    mean_vals = df_all.mean(numeric_only=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset_tag = args.dataset.lower()

    all_path   = args.out_dir / f"all_motions_{dataset_tag}.csv"
    gmr_path   = args.out_dir / f"failed_motions_gmr_{dataset_tag}.csv"
    mimic_path = args.out_dir / f"failed_motions_mimic_{dataset_tag}.csv"

    df_all.to_csv(all_path, index=False, float_format="%.5f")
    df_gmr.to_csv(gmr_path, index=False, float_format="%.5f")
    df_mimic.to_csv(mimic_path, index=False, float_format="%.5f")

    mean_vals = df_all.mean(numeric_only=True)

    print(f"Saved {all_path.name}")
    print(f"Saved {gmr_path.name}")
    print(f"Saved {mimic_path.name}")

    # For "max" metrics, compute mean over NON-ZERO entries only (zeros usually mean "no event")
    eps = 0.0  # set to e.g. 1e-12 if you have tiny numerical noise instead of exact zeros
    nonzero_mean_cols = [
        "gmr_pen_max_depth_m",
        "mimic_pen_max_depth_m",
        "gmr_tendon_max_jump_%",
        "mimic_tendon_max_jump_%",
    ]

    for c in nonzero_mean_cols:
        if c in df_all.columns:
            s = pd.to_numeric(df_all[c], errors="coerce")
            nz = s.abs() > eps
            mean_vals[c] = s[nz].mean() if nz.any() else 0.0

    print(
        "\n=== SUMMARY (MEAN OVER ALL MOTIONS) ===\n"
        f"GMR:  joint={mean_vals['gmr_joint_pct']:.5f}%  "
        f"pen={mean_vals['gmr_pen_pct']:.5f}%  "
        f"pen_max={mean_vals['gmr_pen_max_depth_m']:.5f}m  "
        f"float={mean_vals['gmr_float_height']:.5f}m  "
        f"tendon_max_change_mean={mean_vals['gmr_tendon_max_jump_%']:.5f}%  "
        f"tendon_motion_pct={gmr_tendon_motion_pct:.2f}%  "
        f"spf={mean_vals['gmr_sec_per_frame']:.5f}s "
        f"rmse={mean_vals['gmr_pos_rmse']:.5f}m \n"
        f"MIMIC: joint={mean_vals['mimic_joint_pct']:.5f}%  "
        f"pen={mean_vals['mimic_pen_pct']:.5f}%  "
        f"pen_max={mean_vals['mimic_pen_max_depth_m']:.5f}m  "
        f"float={mean_vals['mimic_float_height']:.5f}m  "
        f"tendon_max_change_mean={mean_vals['mimic_tendon_max_jump_%']:.5f}%  "
        f"tendon_motion_pct={mimic_tendon_motion_pct:.2f}%  "
        f"spf={mean_vals['mimic_sec_per_frame']:.5f}s "
        f"rmse={mean_vals['mimic_pos_rmse']:.5f}m  "
    )

    fail_summaries = [r[7] for r in results]

    merged_fail_summary = defaultdict(lambda: defaultdict(set))
    for fs in fail_summaries:
        merge_fail_summaries(merged_fail_summary, fs)
    
    print_fail_summary(merged_fail_summary)
    
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
