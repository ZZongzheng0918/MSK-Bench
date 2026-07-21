from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


CANONICAL_ALGORITHMS = ("ppo", "sac", "deprl", "msgym", "middleware")

ALGORITHM_ALIASES = {
    "ppo": "ppo",
    "sac": "sac",
    "deprl": "deprl",
    "dynsyn": "msgym",
    "dynsyn-sac": "msgym",
    "msgym": "msgym",
    "middleware": "middleware",
    "latent": "middleware",
    "latent-action": "middleware",
}

METRIC_ALIASES = {
    "success": "success",
    "success_rate": "success",
    "robustness": "robustness",
    "smooth": "smooth",
    "smoothness": "smooth",
    "energy": "energy",
    "activation_energy": "energy",
    "emg": "emg",
    "emg_similarity": "emg",
    "render": "render",
}

SCRIPT_TEMPLATES = {
    "ppo": {
        "success": "ppo/eval_ppo_success.py",
        "robustness": "ppo/eval_ppo_robustness.py",
        "smooth": "ppo/eval_ppo_smooth.py",
        "energy": "ppo/eval_ppo_energy.py",
        "emg": "ppo/export_ppo_emg.py",
        "render": "ppo/render_ppo.py",
    },
    "sac": {
        "success": "sac/eval_sac_success.py",
        "robustness": "sac/eval_sac_robustness.py",
        "smooth": "sac/eval_sac_smooth.py",
        "energy": "sac/eval_sac_energy.py",
        "emg": "sac/export_sac_emg.py",
        "render": "sac/render_sac.py",
    },
    "deprl": {
        "success": "depRL/eval_deprl_success.py",
        "robustness": "depRL/eval_deprl_robustness.py",
        "smooth": "depRL/eval_deprl_smooth.py",
        "energy": "depRL/eval_deprl_energy.py",
        "emg": "depRL/export_deprl_emg.py",
        "render": "depRL/render_deprl.py",
    },
    "msgym": {
        "success": "msgym/eval_msgym_success.py",
        "robustness": "msgym/eval_msgym_robustness.py",
        "smooth": "msgym/eval_msgym_smooth.py",
        "energy": "msgym/eval_msgym_energy.py",
        "emg": "msgym/export_msgym_emg.py",
        "render": "msgym/render_msgym.py",
    },
    "middleware": {
        "success": "deprl_middleware_22tasks/eval_middleware_success.py",
        "robustness": "deprl_middleware_22tasks/eval_middleware_robustness.py",
        "smooth": "deprl_middleware_22tasks/eval_middleware_smooth.py",
        "energy": "deprl_middleware_22tasks/eval_middleware_energy.py",
        "emg": "deprl_middleware_22tasks/export_middleware_emg.py",
        "render": "deprl_middleware_22tasks/render_middleware.py",
    },
}


@dataclass(frozen=True)
class EvaluationRequest:
    metric: str = "success"
    algorithms: tuple[str, ...] = CANONICAL_ALGORITHMS
    env_id: str = "all"
    benchmark_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    episodes: int | None = None
    seed: int | None = None
    output_json: Path | None = None
    output_csv: Path | None = None
    output_dir: Path | None = None
    model_root: Path | None = None
    model_dir: Path | None = None
    model_path: Path | None = None
    norm_path: Path | None = None
    run_path: Path | None = None
    log_path: Path | None = None
    checkpoint: str | None = None
    checkpoint_file: Path | None = None
    max_steps: int | None = None
    deterministic: bool = False
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    execute: bool = False


def normalize_algorithm(value: str) -> str:
    key = value.strip().lower()
    if key == "all":
        return key
    try:
        return ALGORITHM_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(("all", *sorted(ALGORITHM_ALIASES)))
        raise KeyError(f"Unknown algorithm {value!r}. Expected one of: {valid}") from exc


def normalize_metric(value: str) -> str:
    key = value.strip().lower()
    try:
        return METRIC_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(METRIC_ALIASES))
        raise KeyError(f"Unknown metric {value!r}. Expected one of: {valid}") from exc


def normalize_algorithms(values: Iterable[str] | None) -> tuple[str, ...]:
    if not values:
        return CANONICAL_ALGORITHMS
    normalized: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            algorithm = normalize_algorithm(part)
            if algorithm == "all":
                for item in CANONICAL_ALGORITHMS:
                    if item not in normalized:
                        normalized.append(item)
            elif algorithm not in normalized:
                normalized.append(algorithm)
    return tuple(normalized or CANONICAL_ALGORITHMS)


def script_for(algorithm: str, metric: str) -> str:
    normalized_algorithm = normalize_algorithm(algorithm)
    normalized_metric = normalize_metric(metric)
    if normalized_algorithm == "all":
        raise KeyError("script_for() needs one algorithm, not 'all'")
    try:
        return SCRIPT_TEMPLATES[normalized_algorithm][normalized_metric]
    except KeyError as exc:
        raise KeyError(f"{algorithm!r} does not support metric {metric!r}") from exc


def _output_path(path: Path | None, algorithm: str, multiple_algorithms: bool) -> Path | None:
    if path is None:
        return None
    if not multiple_algorithms:
        return path
    return path.with_name(f"{path.stem}_{algorithm}{path.suffix}")


def _output_paths_from_dir(request: EvaluationRequest, algorithm: str, metric: str) -> tuple[Path | None, Path | None, Path | None]:
    if request.output_dir is None:
        return None, None, None
    base = Path(request.output_dir)
    if metric in {"emg", "render"}:
        return None, None, base / algorithm
    return base / f"{metric}_{algorithm}.json", base / f"{metric}_{algorithm}.csv", None


def _clean_extra_args(extra_args: tuple[str, ...]) -> list[str]:
    if extra_args and extra_args[0] == "--":
        return list(extra_args[1:])
    return list(extra_args)


def build_command(request: EvaluationRequest, algorithm: str, *, python: str = "python") -> list[str]:
    normalized_algorithm = normalize_algorithm(algorithm)
    normalized_metric = normalize_metric(request.metric)
    command = [
        python,
        script_for(normalized_algorithm, normalized_metric),
        "--env",
        request.env_id,
        "--benchmark-root",
        str(Path(request.benchmark_root)),
    ]
    if request.episodes is not None:
        command += ["--episodes", str(request.episodes)]
    if request.seed is not None and normalized_algorithm != "msgym":
        command += ["--seed", str(request.seed)]
    if request.max_steps is not None:
        command += ["--max-steps", str(request.max_steps)]
    if request.deterministic:
        command.append("--deterministic")

    algorithms = normalize_algorithms(request.algorithms)
    dir_json, dir_csv, dir_output = _output_paths_from_dir(request, normalized_algorithm, normalized_metric)
    json_path = _output_path(request.output_json, normalized_algorithm, len(algorithms) > 1) or dir_json
    csv_path = _output_path(request.output_csv, normalized_algorithm, len(algorithms) > 1) or dir_csv
    output_dir = dir_output
    if json_path is not None:
        command += ["--json", str(json_path)]
    if csv_path is not None:
        command += ["--csv", str(csv_path)]
    if output_dir is not None:
        command += ["--output-dir", str(output_dir)]

    if request.model_root is not None:
        command += ["--model-root", str(request.model_root)]
    if request.model_dir is not None and normalized_algorithm in {"ppo", "sac"}:
        command += ["--model-dir", str(request.model_dir)]
    if request.model_path is not None and normalized_algorithm in {"ppo", "sac", "msgym"}:
        command += ["--model-path", str(request.model_path)]
    if request.norm_path is not None and normalized_algorithm in {"ppo", "sac", "msgym"}:
        command += ["--norm-path", str(request.norm_path)]
    run_path = request.log_path if normalized_algorithm == "msgym" and request.log_path is not None else request.run_path
    if run_path is not None:
        command += ["--log-path" if normalized_algorithm == "msgym" else "--run-path", str(run_path)]
    if request.checkpoint is not None and normalized_algorithm in {"deprl", "middleware"}:
        command += ["--checkpoint", str(request.checkpoint)]
    if request.checkpoint_file is not None and normalized_algorithm in {"deprl", "middleware"}:
        command += ["--checkpoint-file", str(request.checkpoint_file)]

    command += _clean_extra_args(request.extra_args)
    return command


def build_commands(request: EvaluationRequest, *, python: str = "python") -> list[list[str]]:
    return [build_command(request, algorithm, python=python) for algorithm in normalize_algorithms(request.algorithms)]


def validate_scripts(request: EvaluationRequest) -> list[str]:
    root = Path(request.benchmark_root)
    missing: list[str] = []
    for algorithm in normalize_algorithms(request.algorithms):
        script = root / script_for(algorithm, request.metric)
        if not script.is_file():
            missing.append(f"{algorithm}: {script.relative_to(root).as_posix()}")
    return missing


def run_request(request: EvaluationRequest, *, python: str = "python") -> list[subprocess.CompletedProcess]:
    missing = validate_scripts(request)
    if missing:
        raise FileNotFoundError("Missing evaluator scripts:\n" + "\n".join(f"  - {item}" for item in missing))
    commands = build_commands(request, python=python)
    if not request.execute:
        metric = normalize_metric(request.metric)
        for algorithm, command in zip(normalize_algorithms(request.algorithms), commands):
            print(f"# benchmark_eval --metric {metric} --algorithm {algorithm}")
            print(" ".join(command))
        return []
    completed: list[subprocess.CompletedProcess] = []
    for command in commands:
        completed.append(subprocess.run(command, cwd=Path(request.benchmark_root), check=True))
    return completed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one MSK-Bench metric across one or more algorithms.")
    parser.add_argument("--metric", default="success", choices=tuple(sorted(METRIC_ALIASES)))
    parser.add_argument("--algorithm", dest="algorithms", action="append", help="Algorithm name. Can be repeated.")
    parser.add_argument("--algorithms", dest="algorithms", action="append", help="Comma-separated algorithm names or 'all'.")
    parser.add_argument("--env", dest="env_id", default="all")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--benchmark-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", dest="output_json", type=Path, default=None)
    parser.add_argument("--csv", dest="output_csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--norm-path", type=Path, default=None)
    parser.add_argument("--run-path", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-file", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser


def request_from_args(args: argparse.Namespace) -> EvaluationRequest:
    return EvaluationRequest(
        metric=args.metric,
        algorithms=normalize_algorithms(args.algorithms),
        env_id=args.env_id,
        benchmark_root=args.benchmark_root,
        episodes=args.episodes,
        seed=args.seed,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_dir=args.output_dir,
        model_root=args.model_root,
        model_dir=args.model_dir,
        model_path=args.model_path,
        norm_path=args.norm_path,
        run_path=args.run_path,
        log_path=args.log_path,
        checkpoint=args.checkpoint,
        checkpoint_file=args.checkpoint_file,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        extra_args=tuple(args.extra_args),
        execute=args.execute,
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_request(request_from_args(args), python="python")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALGORITHM_ALIASES",
    "CANONICAL_ALGORITHMS",
    "EvaluationRequest",
    "METRIC_ALIASES",
    "SCRIPT_TEMPLATES",
    "build_command",
    "build_commands",
    "build_parser",
    "normalize_algorithm",
    "normalize_algorithms",
    "normalize_metric",
    "request_from_args",
    "run_request",
    "script_for",
    "validate_scripts",
]
