# depRL Baseline

This directory contains the MSK-Bench integration for depRL baseline workflows. It includes evaluator wrappers, artifact-loading compatibility, and benchmark-specific command entry points.

## Attribution

- Upstream: https://github.com/martius-lab/depRL
- License: preserve `LICENSE` (Apache-2.0), `LICENSE.depRL` (MIT), and nested dependency licenses such as `deprl/vendor/tonic/LICENSE`.
- Local MSK-Bench changes: evaluator entry points, benchmark environment wiring, artifact argument compatibility, JSON/CSV result export, EMG export, and rendering helpers.

Do not remove this attribution section or replace this README wholesale with the upstream README. This file documents the modified MSK-Bench integration surface, not only the original upstream project.

## Training

MSK-Bench depRL configs live in `experiments/msk_bench_training_files/`. The depRL entry point reads the YAML or JSON config from the final command-line argument.

```powershell
cd D:\MSK-Bench\depRL
python -m deprl.main experiments\msk_bench_training_files\msk_bench_walk.yaml
```

The bundled configs use `working_dir: ./baselines_MSKBench`, so checkpoints are written under `baselines_MSKBench\<tonic-name>\<timestamp>\checkpoints\`.

## Evaluation

```powershell
python depRL\eval_deprl_success.py --env MSKBenchWalk-v0 --episodes 10 --benchmark-root D:\MSK-Bench --run-path depRL\baselines_MSKBench\MSKBenchWalk_DEP\<timestamp> --checkpoint last
python depRL\eval_deprl_energy.py --env MSKBenchWalk-v0 --episodes 5 --benchmark-root D:\MSK-Bench --checkpoint-file depRL\baselines_MSKBench\MSKBenchWalk_DEP\<timestamp>\checkpoints\step_<n>.pt
```

Use depRL-style artifact arguments such as `--run-path`, `--checkpoint`, `--checkpoint-file`, `--header`, `--agent`, and `--environment` when evaluating trained agents.

See `../THIRD_PARTY_NOTICES.md` and `../PATCHES.md` for the release-facing third-party notice and local patch summary.