# Unified MSK-Bench Evaluation

Use `benchmark_eval/evaluate.py` when you want one metric across multiple control methods.

Dry run:

```powershell
python benchmark_eval/evaluate.py --metric success --algorithms ppo,sac,deprl --env MSKBenchWalk-v0 --episodes 10
```

Execute:

```powershell
python benchmark_eval/evaluate.py `
  --metric energy `
  --algorithms ppo,sac,deprl,msgym,middleware `
  --env MSKBenchWalk-v0 `
  --episodes 5 `
  --output-dir results\energy `
  --execute
```

Supported metrics are `success`, `robustness`, `smooth`, `energy`, `emg`, and `render`.
The old algorithm-specific scripts are still present for compatibility, but this entrypoint is the canonical command surface for comparing methods under the same metric.
