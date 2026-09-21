"""Middleware robustness evaluation for all 22 MSK-Bench tasks."""

from __future__ import annotations

import eval_middleware_success as base


base.EVAL_MODE = "robustness"
MSK_BENCH_ENVS = base.MSK_BENCH_ENVS


def main(argv: list[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
