"""Benchmark suite metadata aligned with the MSK-Bench task taxonomy."""

from __future__ import annotations

from msk_bench.registry import CANONICAL_TASKS, TASK_FAMILIES, TaskSpec, suite_for_env_id as registry_suite_for_env_id

SUITES: dict[str, tuple[TaskSpec, ...]] = {
    **TASK_FAMILIES,
    "all": CANONICAL_TASKS,
}


def tasks(suite: str = "all") -> tuple[TaskSpec, ...]:
    try:
        return SUITES[suite]
    except KeyError as exc:
        valid = ", ".join(SUITES)
        raise KeyError(f"Unknown MSK-Bench suite {suite!r}. Expected one of: {valid}") from exc


def task_ids(suite: str = "all") -> list[str]:
    return [task.env_id for task in tasks(suite)]


def suite_for_env_id(env_id: str) -> str:
    return registry_suite_for_env_id(env_id)


__all__ = ["SUITES", "suite_for_env_id", "task_ids", "tasks"]
