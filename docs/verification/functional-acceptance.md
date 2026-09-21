# Reproducing functional acceptance

Data prerequisite: this source release excludes two restricted walking motions. See [the README](../../README.md#restricted-motion-references) for authorized acquisition and the unresolved exact Walk-reference version. The full 26-environment commands below require both motions; without them, use the documented separate training/evaluation/rendering phases and pytest's explicit data-dependent skips. `--skip-pretrained` does not waive motion-data requirements.

Use Python 3.11 in a fresh virtual environment and install only this repository and its declared extras:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[all,dev]"
.venv\Scripts\python -B tools/smoke_all.py --phase all --env MSKBenchSquat-v0 --output-dir .smoke/all --keep-going
```

Use a new output directory for each training run. The runner records the command, exit code, duration, log and artifact paths for every job in `report.json`. It validates 26 environment IDs, five eight-step baseline training runs, 25 metric exports and five MP4 renderers. Short training checks operation, not convergence; this is not every algorithm/task combination.

Residual acceptance uses network-downloaded `hf://amathislab/mm-fullbody-base` weights through the existing MuscleMimic checkpoint loader. These weights are not trained locally. No external source checkout is needed. The first run needs network access and cache space. `--skip-pretrained` explicitly selects offline zero-base dynamics checks and must not be described as full pretrained-policy acceptance.

Run individual phases or reuse saved training artifacts after interruption:

```powershell
python -B tools/smoke_all.py --phase environments --output-dir .smoke/envs
python -B tools/smoke_all.py --phase training --output-dir .smoke/run
python -B tools/smoke_all.py --phase evaluation --training-dir .smoke/run/training --output-dir .smoke/eval --keep-going
python -B tools/smoke_all.py --phase rendering --training-dir .smoke/run/training --output-dir .smoke/render --keep-going
$env:MSK_BENCH_TEST_PRETRAINED = "1"
python -B -m pytest -q --junitxml=.smoke/pytest.xml
python -m ruff check . --no-respect-gitignore
python -m pip check
```

The training task defaults to `MSKBenchSquat-v0`; environment checks always cover all 26 IDs. `--algorithm` filters training/evaluation/rendering. `--json-report` selects the report path and `--timeout` limits each job. Incomplete reports are not passing reports, even if all completed jobs succeeded.

The unified evaluator accepts `--artifact-manifest` to load exact saved model and normalization paths. Evaluation files must contain finite, nonempty data. Rendered MP4s must decode at the requested dimensions.

Children use the current Python interpreter in isolated mode with `PYTHONPATH` removed. Rendering defaults to GLFW on Windows/macOS and EGL on Linux unless explicitly configured. An OS graphics driver or working headless EGL implementation remains necessary.

## Scope and known boundaries

- Five standard baseline training/evaluation adapters: PPO, SAC, depRL, msgym/DynSyn and middleware.
- Four extension environments: ResidualRun, ResidualWalk, ResidualStair and AgenticWalk.
- Existing upstream MuscleMimic training, datasets and visualization tools are not rewritten by this repair; real pretrained inference and the three residual environment steps are checked separately.
- Core environment tests do not establish task performance or the quality of a learned policy.
