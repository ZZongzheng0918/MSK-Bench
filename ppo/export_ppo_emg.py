"""Export 12 target-muscle EMG CSV files from PPO checkpoints."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import emg_export_common as emg_common
import eval_ppo_success as base

MSK_BENCH_ENVS = base.MSK_BENCH_ENVS


def main(argv: list[str] | None = None) -> int:
    return emg_common.main_sb3_emg_export(base, argv)


if __name__ == "__main__":
    raise SystemExit(main())
