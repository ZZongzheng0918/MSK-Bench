"""Unified benchmark command builder and launcher for MSK-Bench evaluators."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

from msk_bench.benchmarking.suites import SUITES, task_ids

UNIFIED_EVALUATOR_SCRIPT = "benchmark_eval/evaluate.py"

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

LEGACY_SCRIPT_TEMPLATES = {
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

SCRIPT_TEMPLATES = {
    algorithm: {metric: UNIFIED_EVALUATOR_SCRIPT for metric in metrics}
    for algorithm, metrics in LEGACY_SCRIPT_TEMPLATES.items()
}


@dataclass(frozen=True)
class BenchmarkRunRequest:
    suite: str
    algorithm: str
    metric: str = "success"
    env_id: str | None = None
    episodes: int | None = None
    seed: int | None = None
    output_json: Path | None = None
    output_csv: Path | None = None
    model_root: Path | None = None
    model_dir: Path | None = None
    model_path: Path | None = None
    norm_path: Path | None = None
    run_path: Path | None = None
    log_path: Path | None = None
    checkpoint: str | None = None
    checkpoint_file: Path | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    dry_run: bool = True


def normalize_algorithm(value: str) -> str:
    key = value.lower()
    try:
        return ALGORITHM_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(ALGORITHM_ALIASES))
        raise KeyError(f"Unknown algorithm {value!r}. Expected one of: {valid}") from exc


def normalize_metric(value: str) -> str:
    key = value.lower()
    try:
        return METRIC_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(METRIC_ALIASES))
        raise KeyError(f"Unknown metric {value!r}. Expected one of: {valid}") from exc


def legacy_script_for(algorithm: str, metric: str) -> str:
    normalized_algorithm = normalize_algorithm(algorithm)
    normalized_metric = normalize_metric(metric)
    try:
        return LEGACY_SCRIPT_TEMPLATES[normalized_algorithm][normalized_metric]
    except KeyError as exc:
        raise KeyError(f"{algorithm!r} does not support metric {metric!r}") from exc


def script_for(algorithm: str, metric: str) -> str:
    legacy_script_for(algorithm, metric)
    return UNIFIED_EVALUATOR_SCRIPT


def expand_suite(request: BenchmarkRunRequest) -> tuple[BenchmarkRunRequest, ...]:
    if request.env_id:
        return (request,)
    if request.suite == "all":
        return (replace(request, env_id="all"),)
    return tuple(replace(request, env_id=env_id) for env_id in task_ids(request.suite))


def build_command(request: BenchmarkRunRequest, repo_root: Path | str = Path("."), python: str = "python") -> list[str]:
    repo_root = Path(repo_root)
    algorithm = normalize_algorithm(request.algorithm)
    metric = normalize_metric(request.metric)
    command = [
        python,
        UNIFIED_EVALUATOR_SCRIPT,
        "--algorithm",
        algorithm,
        "--metric",
        metric,
        "--env",
        request.env_id or "all",
        "--benchmark-root",
        str(repo_root),
    ]
    if request.episodes is not None:
        command += ["--episodes", str(request.episodes)]
    if request.seed is not None:
        command += ["--seed", str(request.seed)]
    if request.output_json is not None:
        command += ["--json", str(request.output_json)]
    if request.output_csv is not None:
        command += ["--csv", str(request.output_csv)]
    if request.model_root is not None:
        command += ["--model-root", str(request.model_root)]
    if request.model_dir is not None:
        command += ["--model-dir", str(request.model_dir)]
    if request.model_path is not None:
        command += ["--model-path", str(request.model_path)]
    if request.norm_path is not None:
        command += ["--norm-path", str(request.norm_path)]
    if request.run_path is not None:
        command += ["--run-path", str(request.run_path)]
    if request.log_path is not None:
        command += ["--log-path", str(request.log_path)]
    if request.checkpoint is not None:
        command += ["--checkpoint", str(request.checkpoint)]
    if request.checkpoint_file is not None:
        command += ["--checkpoint-file", str(request.checkpoint_file)]
    if not request.dry_run:
        command.append("--execute")
    command += list(request.extra_args)
    return command


def build_commands(request: BenchmarkRunRequest, repo_root: Path | str = Path("."), python: str = "python") -> list[list[str]]:
    return [build_command(item, repo_root=repo_root, python=python) for item in expand_suite(request)]


def validate_request_assets(request: BenchmarkRunRequest, repo_root: Path | str = Path(".")) -> list[str]:
    repo_root = Path(repo_root)
    missing: list[str] = []
    script = repo_root / script_for(request.algorithm, request.metric)
    if not script.is_file():
        missing.append(f"script: {script.relative_to(repo_root).as_posix()}")
    for label, path in (
        ("model_root", request.model_root),
        ("model_dir", request.model_dir),
        ("model_path", request.model_path),
        ("norm_path", request.norm_path),
        ("run_path", request.run_path),
        ("log_path", request.log_path),
        ("checkpoint_file", request.checkpoint_file),
    ):
        if path is not None and not Path(path).exists():
            missing.append(f"{label}: {path}")
    return missing


def run_request(request: BenchmarkRunRequest, repo_root: Path | str = Path("."), python: str = "python") -> list[subprocess.CompletedProcess]:
    missing = validate_request_assets(request, repo_root=repo_root)
    if missing:
        raise FileNotFoundError("Missing benchmark assets:\n" + "\n".join(f"  - {item}" for item in missing))
    completed = []
    for command in build_commands(request, repo_root=repo_root, python=python):
        if request.dry_run:
            print(" ".join(command))
            continue
        completed.append(subprocess.run(command, cwd=repo_root, check=True))
    return completed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or run MSK-Bench evaluator commands.")
    parser.add_argument("--suite", choices=tuple(SUITES), default="all")
    parser.add_argument("--algorithm", required=True, choices=tuple(sorted(ALGORITHM_ALIASES)))
    parser.add_argument("--metric", default="success", choices=tuple(sorted(METRIC_ALIASES)))
    parser.add_argument("--env", dest="env_id", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json", dest="output_json", type=Path, default=None)
    parser.add_argument("--csv", dest="output_csv", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--norm-path", type=Path, default=None)
    parser.add_argument("--run-path", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-file", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", default="python")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = BenchmarkRunRequest(
        suite=args.suite,
        algorithm=args.algorithm,
        metric=args.metric,
        env_id=args.env_id,
        episodes=args.episodes,
        seed=args.seed,
        output_json=args.output_json,
        output_csv=args.output_csv,
        model_root=args.model_root,
        model_dir=args.model_dir,
        model_path=args.model_path,
        norm_path=args.norm_path,
        run_path=args.run_path,
        log_path=args.log_path,
        checkpoint=args.checkpoint,
        checkpoint_file=args.checkpoint_file,
        extra_args=tuple(args.extra_args),
        dry_run=not args.execute,
    )
    run_request(request, repo_root=args.repo_root, python=args.python)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALGORITHM_ALIASES",
    "BenchmarkRunRequest",
    "LEGACY_SCRIPT_TEMPLATES",
    "METRIC_ALIASES",
    "SCRIPT_TEMPLATES",
    "UNIFIED_EVALUATOR_SCRIPT",
    "build_command",
    "build_commands",
    "expand_suite",
    "legacy_script_for",
    "normalize_algorithm",
    "normalize_metric",
    "run_request",
    "script_for",
    "validate_request_assets",
]
