# PPO Baseline

This directory contains the Stable-Baselines3 PPO training, evaluation, EMG export, and rendering entry points for MSK-Bench.

Useful commands:

```powershell
python ppo\train_ppo_msk_bench.py --list-envs
python ppo\train_ppo_msk_bench.py --env MSKBenchWalk-v0 --dry-run
python ppo\eval_ppo_success.py --env MSKBenchWalk-v0 --episodes 10 --benchmark-root D:\MSK-Bench
```

Use `--model-root`, `--model-path`, and `--norm-path` to point evaluators at trained artifacts. Prefer `benchmark_eval/evaluate.py` when comparing PPO against other algorithms under the same metric.
