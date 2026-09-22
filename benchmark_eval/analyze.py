"""File-based analyses. Input artifacts remain outside the source distribution."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from msk_bench.analysis.emg import best_shifted_pearson, minmax_normalize, resample_to_gait_cycle
from msk_bench.analysis.efficiency import peak_efficiency_steps
from msk_bench.analysis.robustness import aggregate_robustness


def read_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("JSON input must be a list of records")
        return rows
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def compare_emg_csv(simulated, reference, mapping):
    """Compare explicit gait cycles to a preprocessed single-cycle human envelope.

    Simulated CSV requires episode, cycle and the mapped muscle columns. Boundaries
    must come from gait events; never join different episodes or guess cycles from
    activation peaks. Reference columns contain one cycle of envelope samples.
    """
    if not mapping:
        raise ValueError("Provide an explicit simulated-to-human muscle column map")
    sim, ref = read_rows(simulated), read_rows(reference)
    if not sim or len(ref) < 3:
        raise ValueError("Empty simulation or insufficient reference samples")
    identities = {(row.get("algorithm", ""), row.get("env_id", "")) for row in sim}
    if len(identities) != 1:
        raise ValueError("Compare one algorithm and task at a time")
    cycles = defaultdict(list)
    for row in sim:
        if not row.get("episode") or row.get("cycle") in (None, ""):
            raise ValueError("Simulated CSV requires explicit episode and cycle identifiers")
        cycles[row["episode"], row["cycle"]].append(row)
    results = {}
    for muscle, human_column in mapping.items():
        try:
            human = np.asarray([float(r[human_column]) for r in ref])
            sampled = []
            for rows in cycles.values():
                signal = np.asarray([float(r[muscle]) for r in rows])
                if len(signal) < 3 or not np.isfinite(signal).all():
                    raise ValueError("Each gait cycle needs at least three finite samples")
                sampled.append(resample_to_gait_cycle(signal, 101))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid or missing EMG data for {muscle}: {exc}") from exc
        if not np.isfinite(human).all():
            raise ValueError(f"Nonfinite human envelope for {muscle}")
        mean_raw = np.mean(sampled, axis=0)
        mean = minmax_normalize(mean_raw)
        human = minmax_normalize(resample_to_gait_cycle(human, 101))
        score, shift = best_shifted_pearson(mean, human)
        if not np.isfinite(score):
            raise ValueError(f"Undefined Pearson correlation (constant envelope) for {muscle}")
        scale = float(np.ptp(mean_raw))
        results[muscle] = dict(pearson=float(score), shift_points=int(shift), cycle_count=len(sampled),
                               simulated=np.roll(mean, shift).tolist(), reference=human.tolist(),
                               std=np.roll(np.std(sampled, axis=0) / scale, shift).tolist())
    return dict(points=101, muscles=results,
                mean_pearson=float(np.mean([r["pearson"] for r in results.values()])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", choices=("robustness", "peak", "emg", "recruitment", "reconstruction"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--mapping", type=Path, help="EMG column map, or actuator-to-family JSON for recruitment")
    parser.add_argument("--source", help="Human data source/version and preprocessing description (required for EMG)")
    parser.add_argument("--success-unit", choices=("fraction", "percent"))
    parser.add_argument("--custom-grid", action="store_true", help="Allow non-paper perturbation grids")
    parser.add_argument("--all-tasks", action="store_true", help="Require all 22 canonical environments")
    args = parser.parse_args(argv)
    if args.output.resolve() in {p.resolve() for p in (args.input, args.reference, args.mapping) if p}:
        parser.error("Output must not overwrite an input")
    if args.metric == "robustness":
        if not args.success_unit:
            parser.error("Declare --success-unit; benchmark evaluators export percent")
        from msk_bench.registry import CANONICAL_ENV_IDS
        result = aggregate_robustness(read_rows(args.input), success_unit=args.success_unit,
            require_paper_grid=not args.custom_grid,
            expected_tasks=CANONICAL_ENV_IDS if args.all_tasks else None)
    elif args.metric == "peak":
        grouped = defaultdict(list)
        for row in read_rows(args.input):
            if not row.get("env_id") or not row.get("algorithm"):
                parser.error("Peak logs must label env_id and algorithm to prevent pooling task returns")
            grouped[row["env_id"], row["algorithm"]].append(row)
        if not grouped:
            parser.error("No evaluation records")
        result = [dict(env_id=t, algorithm=a, peak_efficiency_steps=peak_efficiency_steps(rows))
                  for (t, a), rows in sorted(grouped.items())]
    elif args.metric == "emg":
        if not args.reference or not args.mapping or not args.source:
            parser.error("EMG requires --reference, --mapping and --source")
        result = compare_emg_csv(args.input, args.reference, json.loads(args.mapping.read_text(encoding="utf-8")))
        result["reference_source"] = args.source
    elif args.metric == "recruitment":
        if not args.mapping:
            parser.error("Recruitment requires an explicit actuator-to-family --mapping")
        mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
        with np.load(args.input, allow_pickle=False) as data:
            names, values = data["actuator_names"].tolist(), data["activations"]
        if values.ndim != 2 or values.shape[1] != len(names) or not len(values) or not np.isfinite(values).all():
            raise ValueError("Need finite (time, actuator) activations matching actuator_names")
        if len(set(names)) != len(names) or set(names) != set(mapping):
            raise ValueError("Mapping must cover every unique actuator name exactly once")
        masses = defaultdict(float)
        for name, mean in zip(names, np.mean(np.abs(values), axis=0)):
            masses[mapping[name]] += float(mean)
        total = sum(masses.values())
        if total <= 0:
            raise ValueError("Undefined recruitment shares for zero total activation")
        result = dict(actuator_count=len(names), frames=len(values), family_mean_mass=dict(masses),
                      family_share={key: value / total for key, value in masses.items()})
    else:
        with np.load(args.input, allow_pickle=False) as data:
            target, predicted = data["target"], data["reconstruction"]
        if target.shape != predicted.shape or target.ndim != 2 or len(target) < 2:
            raise ValueError("Need matching (sample, actuator) target and reconstruction")
        if not np.isfinite(target).all() or not np.isfinite(predicted).all():
            raise ValueError("Reconstruction arrays must be finite")
        denominator = float(np.var(target, axis=0).sum())
        if denominator <= 0:
            raise ValueError("Explained variance requires variable target activations")
        result = dict(samples=len(target), actuator_count=target.shape[1],
                      mse=float(np.mean((target - predicted)**2)),
                      explained_variance=float(1 - np.var(target - predicted, axis=0).sum() / denominator))
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (args.input, args.reference, args.mapping) if p}
    payload = dict(metric=args.metric, input_sha256=sources, result=result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
