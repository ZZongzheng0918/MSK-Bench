"""Benchmark suite metadata and command builders for MSK-Bench."""

from importlib import import_module

from .metrics import METRICS
from .suites import SUITES, suite_for_env_id, task_ids, tasks

_RUNNER_EXPORTS = {
    "ALGORITHM_ALIASES",
    "BenchmarkRunRequest",
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
}


def __getattr__(name: str):
    if name in _RUNNER_EXPORTS:
        runner = import_module(".runner", __name__)
        value = getattr(runner, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "METRICS",
    "SUITES",
    "suite_for_env_id",
    "task_ids",
    "tasks",
    *_RUNNER_EXPORTS,
]