# DynSyn/msgym Baseline

This directory contains the MSK-Bench integration for DynSyn/msgym baseline workflows. It includes configuration files, evaluator wrappers, Stable-Baselines3-compatible artifact loading, and benchmark-specific command entry points.

## Attribution

- Upstream: https://github.com/Beanpow/DynSyn
- License: preserve `LICENSE` (Apache-2.0) and any nested upstream notice files if additional upstream code is synced later.
- Local MSK-Bench changes: evaluator entry points, benchmark environment wiring, model-root/model-path/norm-path argument compatibility, JSON/CSV result export, EMG export, and rendering helpers.

Do not remove this attribution section or replace this README wholesale with the upstream README. This file documents the modified MSK-Bench integration surface, not only the original upstream project.

## Useful Commands

```powershell
python msgym\eval_msgym_success.py --env MSKBenchWalk-v0 --episodes 10 --benchmark-root D:\MSK-Bench
python msgym\eval_msgym_energy.py --env MSKBenchWalk-v0 --episodes 5 --benchmark-root D:\MSK-Bench
```

Use `--model-root`, `--model-path`, `--norm-path`, or `--log-path` to point evaluators at trained artifacts.

See `../THIRD_PARTY_NOTICES.md` and `../PATCHES.md` for the release-facing third-party notice and local patch summary.
