# Contributing to MSK-Bench

Thank you for helping improve MSK-Bench. This project aims to stay useful as a benchmark: changes should be reproducible, documented, and easy to compare across algorithms.

## Development Setup

```powershell
cd D:\MSK-Bench
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install optional extras only for the baseline you are working on:

```powershell
python -m pip install -e ".[sb3]"
python -m pip install -e ".[deprl]"
python -m pip install -e ".[dynsyn]"
python -m pip install -e ".[residual]"
```

## Before Opening a Change

Run the regression suite from the repository root:

```powershell
python -B -m unittest discover -s tests -v
```

Use `python -B` or `PYTHONDONTWRITEBYTECODE=1` so Python cache directories are not written into the repository.

## Contribution Guidelines

- Keep benchmark task IDs stable unless the change is intentionally breaking.
- Add or update tests for task metadata, metrics, command builders, and public APIs.
- Keep generated artifacts out of git: checkpoints, run logs, videos, TensorBoard data, and local datasets should stay local.
- Prefer small, reviewable changes over broad rewrites.
- Preserve third-party license files in nested upstream projects.
- Document new metrics, baselines, wrappers, and required artifacts in the relevant README.

## Baseline Changes

When adding or changing a baseline, make sure the corresponding evaluator supports the common arguments used by `benchmark_eval/evaluate.py`: `--env`, `--episodes`, `--benchmark-root`, `--json`, `--csv`, and algorithm-specific artifact paths.

## Reporting Issues

When reporting a bug, include:

- Environment ID and algorithm.
- Exact command.
- Python version and operating system.
- Relevant package versions.
- Full error message or traceback.
- Whether the failure happens in a clean environment.
