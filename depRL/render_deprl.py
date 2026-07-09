"""Render depRL policies for all 22 MSK-Bench tasks."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_deprl_success as base
import msk_eval_common as common

MSK_BENCH_ENVS = base.MSK_BENCH_ENVS
# CLI uses default="all" so the script covers all 22 canonical tasks.


def main(argv: list[str] | None = None) -> int:
    return common.main_deprl_render(base, argv)


if __name__ == "__main__":
    raise SystemExit(main())
