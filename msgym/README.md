# DynSyn/msgym Baseline

This directory contains the MSK-Bench integration for DynSyn/msgym baseline workflows. It includes configuration files, evaluator wrappers, Stable-Baselines3-compatible artifact loading, and benchmark-specific command entry points.

## Attribution

- Upstream: https://github.com/Beanpow/DynSyn
- License: preserve `LICENSE` (Apache-2.0) and any nested upstream notice files if additional upstream code is synced later.
- Local MSK-Bench changes: evaluator entry points, benchmark environment wiring, model-root/model-path/norm-path/log-path argument compatibility, JSON/CSV result export, EMG export, and rendering helpers.

Do not remove this attribution section or replace this README wholesale with the upstream README. This file documents the modified MSK-Bench integration surface, not only the original upstream project.

## Training

```powershell
python msgym\SB3-Scripts\train.py --list-configs
python msgym\SB3-Scripts\train.py -f configs\msk_bench_walk.json
```

Bundled configs default to `./runs/msgym_logs`. From the repository root, trained runs are written under `msgym\runs\msgym_logs\<env-name>\<timestamp_seed>\checkpoint\`.

The evaluator loads `best_model.zip` and `best_env.zip` from the run's `checkpoint/` directory. You can pass `--log-path` for the run directory, or pass `--model-path` and `--norm-path` directly.

## Evaluation

```powershell
python msgym\eval_msgym_success.py --env MSKBenchWalk-v0 --episodes 10 --benchmark-root D:\MSK-Bench --log-path msgym\runs\msgym_logs\MSKBenchWalk-v0\<timestamp_seed>
python msgym\eval_msgym_energy.py --env MSKBenchWalk-v0 --episodes 5 --benchmark-root D:\MSK-Bench --model-path msgym\runs\msgym_logs\MSKBenchWalk-v0\<timestamp_seed>\checkpoint\best_model.zip --norm-path msgym\runs\msgym_logs\MSKBenchWalk-v0\<timestamp_seed>\checkpoint\best_env.zip
```

Prefer `benchmark_eval/evaluate.py` when comparing msgym against other algorithms under the same metric.

See `../THIRD_PARTY_NOTICES.md` and `../PATCHES.md` for the release-facing third-party notice and local patch summary.