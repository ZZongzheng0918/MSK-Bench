# SAC Baseline

This directory contains the Stable-Baselines3 SAC training, evaluation, EMG export, and rendering entry points for MSK-Bench.

Useful commands:

```powershell
python sac\train_sac_msk_bench.py --list-envs
python sac\train_sac_msk_bench.py --env MSKBenchWalk-v0 --dry-run
python sac\eval_sac_success.py --env MSKBenchWalk-v0 --episodes 10 --benchmark-root D:\MSK-Bench
```

Use `--model-root`, `--model-path`, and `--norm-path` to point evaluators at trained artifacts. Prefer `benchmark_eval/evaluate.py` when comparing SAC against other algorithms under the same metric.
