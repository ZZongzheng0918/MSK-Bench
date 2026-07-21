# depRL Middleware for 22 MSK-Bench Tasks

This directory contains the latent-action middleware wrapper, config generation utilities, expert-data collection script, transformer training script, and evaluator wrappers for all canonical MSK-Bench tasks.

## Attribution

- Upstream: this package is a local MSK-Bench integration built around depRL-style latent-action workflows. Related depRL upstream code is tracked at https://github.com/martius-lab/depRL.
- License: depRL-derived components retain the license files in `../depRL/`; MSK-Bench wrapper code follows the repository's top-level license status until the final license is confirmed.
- Local MSK-Bench changes: 22-task middleware registration, benchmark config generation, expert-data collection, transformer training, encoder/decoder artifact handling, evaluator entry points, JSON/CSV result export, EMG export, and rendering helpers.

Do not remove this attribution section. This README describes the MSK-Bench middleware integration and should not be replaced by an upstream README from a related dependency.

## Useful Commands

```powershell
python deprl_middleware_22tasks\generate_configs.py --output-dir deprl_middleware_22tasks\configs
python deprl_middleware_22tasks\collect_expert_data.py --env MSKBenchReach-v0 --checkpoint-dir checkpoints\deprl\reach --samples 50000
python deprl_middleware_22tasks\train_expert_transformer.py --data artifacts\expert_synergy.pt --output-dir artifacts --latent-dim 64
python deprl_middleware_22tasks\eval_middleware_success.py --env MSKBenchWalk-v0 --episodes 10 --benchmark-root D:\MSK-Bench
```

Use `--encoder-path`, `--decoder-path`, and `--strict-weights` when evaluating middleware models that require explicit weights.

See `../THIRD_PARTY_NOTICES.md` and `../PATCHES.md` for the release-facing third-party notice and local patch summary.
