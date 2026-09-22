"""Middleware package for depRL training on all 22 MSK-Bench tasks."""

from .registry import MSK_BENCH_TASKS, register_all

__all__ = ["MSK_BENCH_TASKS", "register_all"]
