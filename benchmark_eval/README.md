# Unified MSK-Bench Evaluation

Use `benchmark_eval/evaluate.py` when you want one metric across multiple control methods. The command is a dry run by default: it prints the underlying algorithm-specific evaluator commands without running simulation.

## Dry Run

```powershell
python benchmark_eval/evaluate.py --metric success --algorithms ppo,sac,deprl --env MSKBenchWalk-v0 --episodes 10
```

## Execute

```powershell
python benchmark_eval/evaluate.py `
  --metric energy `
  --algorithms ppo,sac,deprl,msgym,middleware `
  --env MSKBenchWalk-v0 `
  --episodes 5 `
  --output-dir results\energy `
  --execute
```

## Loading Trained Weights

The unified evaluator does not keep checkpoints in the repository. Pass trained artifacts explicitly or place them in the default run layouts used by the individual baseline scripts.

For PPO and SAC, use a per-task model root or direct Stable-Baselines3 files:

```powershell
python benchmark_eval/evaluate.py `
  --metric success `
  --algorithm ppo `
  --env MSKBenchWalk-v0 `
  --episodes 10 `
  --model-path checkpoints\ppo\walk\best_model.zip `
  --norm-path checkpoints\ppo\walk\vec_normalize.pkl `
  --execute
```

For depRL and middleware, use a Tonic run directory and checkpoint selector, or pass a checkpoint file directly:

```powershell
python benchmark_eval/evaluate.py `
  --metric success `
  --algorithm deprl `
  --env MSKBenchWalk-v0 `
  --episodes 10 `
  --run-path checkpoints\deprl\walk_run `
  --checkpoint last `
  --execute
```

For DynSyn/msgym, use the log directory plus optional explicit model and normalization paths:

```powershell
python benchmark_eval/evaluate.py `
  --metric success `
  --algorithm msgym `
  --env MSKBenchWalk-v0 `
  --episodes 10 `
  --log-path checkpoints\msgym\walk_log `
  --execute
```

## Artifact Arguments

| Argument | Algorithms | Purpose |
|---|---|---|
| `--model-root` | PPO, SAC, msgym, depRL, middleware | Root directory containing per-task trained artifacts. |
| `--model-dir` | PPO, SAC | Single run/checkpoint directory searched before `--model-root`. |
| `--model-path` | PPO, SAC, msgym | Direct model file. |
| `--norm-path` | PPO, SAC, msgym | Observation-normalization file. |
| `--run-path` | depRL, middleware | depRL/Tonic run directory. |
| `--log-path` | msgym | DynSyn/msgym log directory. |
| `--checkpoint` | depRL, middleware | Checkpoint selector such as `last` or a training step. |
| `--checkpoint-file` | depRL, middleware | Explicit checkpoint file. |

Supported metrics are `success`, `robustness`, `smooth`, `energy`, `emg`, and `render`.
The old algorithm-specific scripts are still present for compatibility, but this entrypoint is the canonical command surface for comparing methods under the same metric.