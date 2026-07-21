# depRL Baseline

This directory contains the MSK-Bench integration for depRL baseline workflows. It includes evaluator wrappers, artifact-loading compatibility, and benchmark-specific command entry points.

## Attribution

- Upstream: https://github.com/martius-lab/depRL
- License: preserve `LICENSE` (Apache-2.0), `LICENSE.depRL` (MIT), and nested dependency licenses such as `deprl/vendor/tonic/LICENSE`.
- Local MSK-Bench changes: evaluator entry points, benchmark environment wiring, artifact argument compatibility, JSON/CSV result export, EMG export, and rendering helpers.

Do not remove this attribution section or replace this README wholesale with the upstream README. This file documents the modified MSK-Bench integration surface, not only the original upstream project.

## Useful Commands

```powershell
python depRL\eval_deprl_success.py --env MSKBenchWalk-v0 --episodes 10 --benchmark-root D:\MSK-Bench
python depRL\eval_deprl_energy.py --env MSKBenchWalk-v0 --episodes 5 --benchmark-root D:\MSK-Bench
```

Use depRL-style artifact arguments such as `--run-path`, `--checkpoint`, `--checkpoint-file`, `--header`, `--agent`, and `--environment` when evaluating trained agents.

See `../THIRD_PARTY_NOTICES.md` and `../PATCHES.md` for the release-facing third-party notice and local patch summary.
