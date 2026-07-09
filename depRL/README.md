# MSK-Bench

MSK-Bench is a compact benchmark project for full-body musculoskeletal control tasks in MuJoCo. It exposes the 22 benchmark tasks from the MSK-Bench paper with one canonical naming scheme:

```python
import msk_bench
from msk_bench.utils import gym

env = gym.make("MSKBenchWalkTurn-v0")
```

Environment IDs use `MSKBench{Name}-v0`, and implementation classes use `MSKBench{Name}EnvV0`.

## Installation

```bash
pip install -e .
```

This checkout keeps `deprl/` in this directory and the `msk_bench` package in the sibling `../MSK-Bench/msk_bench` directory. The editable install is configured in `pyproject.toml` to include both packages without moving files.
## Task Families

Stabilization: Stand, Powerlift, SingleLegStand, Sit, Balance, Squat.

Locomotion: Walk, Crawl, Run, Jump, WalkTurn, Sidestep.

Interaction: Stairs, Hurdle, StepStones, Slide, DoorOpen, Reach, WalkAndSit, ChinUp, Catch, PoleWalk.

## Smoke Test

```bash
python test.py
```

The smoke test creates `MSKBenchPowerlift-v0`, runs a short random rollout, and writes `MSKBenchPowerlift-v0_test.gif` when rendering is available.
## depRL Training

MSK-Bench includes the depRL training code under `deprl/` and MSK-Bench-specific configs under `experiments/msk_bench_training_files/`.

```bash
python -m deprl.main experiments/msk_bench_training_files/msk_bench_powerlift.yaml
```
