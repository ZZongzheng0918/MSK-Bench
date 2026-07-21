# MSK-Bench

MSK-Bench is an open-source musculoskeletal-control benchmark for full-body MyoSuite-style tasks. It provides Gymnasium-compatible environments, a canonical 22-task benchmark suite, baseline training and evaluation entry points, metric utilities, and shared tooling for comparing control algorithms on muscle-driven humanoid tasks.

The repository focuses on reusable benchmark code. Model checkpoints, long-running experiment outputs, videos, logs, and local run artifacts are intentionally kept out of version control.

## Contents

- [Features](#features)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Environment Registration](#environment-registration)
- [Benchmark Tasks](#benchmark-tasks)
- [Python APIs](#python-apis)
- [Metrics](#metrics)
- [Unified Evaluation](#unified-evaluation)
- [Baseline Scripts](#baseline-scripts)
- [Training](#training)
- [Development And Testing](#development-and-testing)
- [Artifact And Cache Policy](#artifact-and-cache-policy)
- [Third-Party Components And Local Patches](#third-party-components-and-local-patches)
- [Troubleshooting](#troubleshooting)
- [License Status](#license-status)

## Features

MSK-Bench currently includes:

- 22 canonical musculoskeletal benchmark environments grouped into `stabilization`, `locomotion`, `interaction`, and `all` suites.
- Gymnasium/Gym registration for canonical tasks plus residual-control and agentic extension environments.
- Metadata APIs for task IDs, task specs, suite lookup, model-path resolution, and evaluator command construction.
- Metric helpers for success-oriented result schemas, cumulative reward logs, peak-efficiency steps, muscle activation energy, joint smoothness, muscle recruitment grouping, and EMG-envelope similarity.
- A unified evaluation CLI for running one metric across PPO, SAC, depRL, DynSyn/msgym, and latent-action middleware baselines.
- Algorithm-specific compatibility scripts for success, robustness, smoothness, activation energy, EMG export, rendering, and selected training workflows.
- Shared evaluation utilities for headless rendering, episode rollout, energy summaries, smoothness summaries, JSON/CSV writing, and target-muscle EMG export.
- Regression tests covering task metadata, metrics, packaging dependencies, evaluator command construction, baseline command behavior, open-source metadata, and repository layout.

Not included by default:

- Trained checkpoints.
- Large generated datasets.
- Local run directories, videos, TensorBoard logs, or experiment tracking outputs.
- A finalized unified top-level open-source license. See [License Status](#license-status).

## Repository Layout

```text
MSK-Bench/
  .github/workflows/               Continuous integration workflow.
  benchmark_eval/                  Unified multi-algorithm evaluation entry point.
  depRL/                           depRL baseline wrappers and evaluators.
  deprl_middleware_22tasks/        Latent-action middleware for all 22 tasks.
  msgym/                           DynSyn/msgym baseline and SB3 integration files.
  msk_bench/                       Main Python package.
    analysis/                      Metric and aggregation helpers.
    benchmarking/                  Suite metadata and benchmark command builders.
    envs/msk/benchmark/            Environment registration and task implementations.
    integrations/                  Integration-facing package area.
    physics/, renderer/, robot/    Simulation support modules.
    simhive/                       Bundled MuJoCo/MSK model assets.
    utils/                         Gym compatibility and utility helpers.
  ppo/                             PPO training, evaluation, EMG, and rendering scripts.
  sac/                             SAC training, evaluation, EMG, and rendering scripts.
  tests/                           Unit and regression tests.
  third_party/                     Vendored or mirrored third-party project code.
  emg_export_common.py             Shared target-muscle EMG export utilities.
  msk_eval_common.py               Shared evaluation, rendering, energy, and smoothness utilities.
  pyproject.toml                   Package metadata, dependencies, optional extras, and lint config.
  CONTRIBUTING.md                  Development and contribution guide.
  CITATION.cff                     Software citation metadata.
  THIRD_PARTY_NOTICES.md           Third-party component notice index.
  PATCHES.md                       Local summary of modifications to upstream-derived code.
  README.md                        This file.
```

## Installation

### Requirements

- Python 3.11 or newer for the top-level package.
- A MuJoCo-compatible runtime for full environment execution.
- `gymnasium`, `mujoco`, `numpy`, and `scipy` for the default installation.
- Optional baseline dependencies for Stable-Baselines3, depRL, DynSyn/msgym, residual-control workflows, and development checks.

### Editable Install

From the repository root:

```powershell
cd D:\MSK-Bench
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Install optional dependency groups as needed:

```powershell
python -m pip install -e ".[sb3]"       # PPO/SAC Stable-Baselines3 workflows
python -m pip install -e ".[deprl]"     # depRL wrappers and YAML config support
python -m pip install -e ".[dynsyn]"    # DynSyn/msgym workflows
python -m pip install -e ".[residual]"  # Residual-control dependencies
python -m pip install -e ".[dev]"       # Test and lint tooling
```

Some nested third-party projects may require their own setup steps, CUDA choices, MuJoCo rendering configuration, or upstream dependency pins.

### Source-Only Usage

For metadata checks or quick local scripts without installing the package:

```powershell
$env:PYTHONPATH = "D:\MSK-Bench"
python -B -m unittest discover -s tests -v
```

Use `python -B` or `PYTHONDONTWRITEBYTECODE=1` when running repository tests. The layout tests expect that Python cache directories are not left in the repository.

## Environment Registration

Importing `msk_bench` registers environments when `gymnasium` or compatible `gym` is installed:

```python
import gymnasium as gym
import msk_bench

env = gym.make("MSKBenchWalk-v0")
obs, info = env.reset(seed=0)
```

If `gymnasium` or `gym` is unavailable, `msk_bench.gym` is set to `None`, and environment registration is skipped. Install the runtime dependencies before calling `gym.make()`.

The registration layer uses task metadata from `msk_bench.registry` and model files from `msk_bench/simhive/msk_sim/body`. Residual-control extension tasks use resources from `msk_bench/envs/msk/benchmark/residualrl`.

Additional registered environments beyond the 22 canonical tasks:

| Environment ID | Purpose | Horizon |
|---|---|---:|
| `MSKBenchResidualRun-v0` | Residual-control run extension. | 5000 |
| `MSKBenchResidualStair-v0` | Residual-control stair extension. | 2000 |
| `MSKBenchResidualWalk-v0` | Residual-control walk extension. | 1000 |
| `MSKBenchAgenticWalk-v0` | Agentic walk variant using `full_body.xml`. | 1000 |

## Benchmark Tasks

The public suite names are `stabilization`, `locomotion`, `interaction`, and `all`. Individual task specs also keep implementation-oriented families such as `posture`, `balance`, `locomotion`, and `manipulation`.

| Suite | Task | Environment ID | Implementation family | Horizon | Success metric | Model file |
|---|---|---|---|---:|---|---|
| stabilization | Stand | `MSKBenchStand-v0` | posture | 1000 | solved | `full_body.xml` |
| stabilization | Powerlift | `MSKBenchPowerlift-v0` | manipulation | 1000 | hold_bonus | `powerlift.xml` |
| stabilization | SingleLegStand | `MSKBenchSingleLegStand-v0` | posture | 1000 | solved | `full_body.xml` |
| stabilization | Sit | `MSKBenchSit-v0` | posture | 1000 | solved | `sit.xml` |
| stabilization | Balance | `MSKBenchBalance-v0` | balance | 1000 | solved | `balance.xml` |
| stabilization | Squat | `MSKBenchSquat-v0` | locomotion | 1000 | solved | `squat.xml` |
| locomotion | Walk | `MSKBenchWalk-v0` | locomotion | 1000 | solved | `full_body.xml` |
| locomotion | Crawl | `MSKBenchCrawl-v0` | locomotion | 1000 | solved | `full_body.xml` |
| locomotion | Run | `MSKBenchRun-v0` | locomotion | 1000 | solved | `full_body.xml` |
| locomotion | Jump | `MSKBenchJump-v0` | locomotion | 1000 | solved | `full_body.xml` |
| locomotion | WalkTurn | `MSKBenchWalkTurn-v0` | locomotion | 2000 | solved | `walk_turn.xml` |
| locomotion | Sidestep | `MSKBenchSidestep-v0` | locomotion | 1000 | solved | `full_body.xml` |
| interaction | Stairs | `MSKBenchStairs-v0` | locomotion | 2000 | solved | `stairs.xml` |
| interaction | Hurdle | `MSKBenchHurdle-v0` | locomotion | 1000 | solved | `hurdle.xml` |
| interaction | StepStones | `MSKBenchStepStones-v0` | locomotion | 1000 | solved | `step_stones.xml` |
| interaction | Slide | `MSKBenchSlide-v0` | locomotion | 1500 | solved | `slide.xml` |
| interaction | DoorOpen | `MSKBenchDoorOpen-v0` | manipulation | 1000 | solved | `door_open.xml` |
| interaction | Reach | `MSKBenchReach-v0` | manipulation | 500 | solved | `reach.xml` |
| interaction | WalkAndSit | `MSKBenchWalkAndSit-v0` | locomotion | 500 | solved | `sit.xml` |
| interaction | ChinUp | `MSKBenchChinUp-v0` | manipulation | 500 | solved | `chin_up.xml` |
| interaction | Catch | `MSKBenchCatch-v0` | manipulation | 200 | solved | `catch.xml` |
| interaction | PoleWalk | `MSKBenchPoleWalk-v0` | locomotion | 1000 | solved | `pole_walk.xml` |

## Python APIs

### Task Metadata

```python
from msk_bench.benchmarking import task_ids, tasks, suite_for_env_id

all_envs = task_ids("all")
locomotion_envs = task_ids("locomotion")
interaction_specs = tasks("interaction")
family = suite_for_env_id("MSKBenchWalk-v0")
```

Direct registry access is also available:

```python
from msk_bench.registry import CANONICAL_TASKS, CANONICAL_TASK_BY_ID, model_path_for

walk = CANONICAL_TASK_BY_ID["MSKBenchWalk-v0"]
print(walk.slug)          # walk
print(walk.entry_point)   # msk_bench.envs.msk.benchmark.msk_bench_v0:MSKBenchWalkEnvV0
```

### Actions

The canonical policy-action helpers convert between policy outputs in `[-1, 1]` and muscle excitations in `[0, 1]`:

```python
from msk_bench.action_transform import canonical_policy_action_to_excitation, excitation_to_canonical_policy_action

excitation = canonical_policy_action_to_excitation(action)
action = excitation_to_canonical_policy_action(excitation)
```

### Powerlift Reward Helper

```python
from msk_bench.rewards import PowerliftRewardConfig, compute_powerlift_reward

result = compute_powerlift_reward(
    hold_bonus=1.0,
    overhead_height=0.8,
    torso_upright=0.9,
    spine_straight=0.7,
    lumbar_effort=0.3,
    bend_over_penalty=0.0,
    activation=[0.1, 0.2, 0.3],
    done=False,
    config=PowerliftRewardConfig(),
)
print(result.dense, result.terms)
```

## Metrics

Metric names exposed by `msk_bench.benchmarking.metrics.METRICS`:

- `success_rate`
- `cumulative_reward`
- `peak_efficiency_steps`
- `action_noise_robustness`
- `observation_noise_robustness`
- `dynamics_randomization_robustness`
- `muscle_activation_energy`
- `joint_smoothness`
- `emg_similarity`

Analysis helpers are available from `msk_bench.analysis`:

```python
from msk_bench.analysis import (
    activation_energy,
    activation_abs_sum,
    timestep_activation_energy,
    finite_difference_jerk,
    log_mean_squared_jerk,
    process_and_align_simulated_emg,
    mean_emg_similarity,
    peak_efficiency_steps,
    family_activation_mean,
)
```

Metric details:

- Activation energy: `activation_energy(x)` returns the mean squared muscle activation over all timesteps and muscles.
- Absolute activation mass: `activation_abs_sum(x)` sums absolute activations.
- Per-timestep activation energy: `timestep_activation_energy(x)` returns mean squared activation per timestep.
- Smoothness: `finite_difference_jerk(joint_velocity, dt)` computes mean squared second finite differences of joint velocity; `log_mean_squared_jerk()` returns `log10` of that value with an epsilon floor.
- Peak efficiency: `peak_efficiency_steps(rows)` returns the largest numeric logged step. It accepts step aliases `step`, `steps`, `environment_step`, `env_step`, `timestep`, and `timesteps`.
- EMG similarity: `process_and_align_simulated_emg()` normalizes simulated EMG to 101 gait-cycle points, detects peak-to-peak cycles, averages cycles, min-max normalizes the envelope, circularly shifts it to maximize Pearson correlation with the reference, and returns `(aligned_curve, aligned_spread, score)`.
- Mean EMG similarity: `mean_emg_similarity(simulated_by_muscle, reference_by_muscle)` averages finite per-muscle Pearson scores.
- Muscle recruitment: `family_activation_mean()` and `family_activation_mass()` group actuator activations by anatomical keyword families: `paraspinals_deep`, `iliocostalis`, `abdominals`, `deltoids`, `elbow_flexors`, `elbow_extensors`, `quadriceps`, `hamstrings`, `calves`, and `unclassified`.

The EMG exporter targets these 12 muscle channels when actuator names are available:

```text
soleus_r, soleus_l, gasmed_r, gaslat_r, tibant_r, perlong_r, perbrev_r,
recfem_r, vaslat_r, vasmed_r, bflh_r, semiten_r
```

## Unified Evaluation

Use `benchmark_eval/evaluate.py` to build or run one metric across one or more algorithms. By default it is a dry run that prints the commands it would execute.

```powershell
python benchmark_eval\evaluate.py --metric success --algorithms ppo,sac,deprl --env MSKBenchWalk-v0 --episodes 10
```

Run the generated evaluator subprocesses with `--execute`:

```powershell
python benchmark_eval\evaluate.py `
  --metric energy `
  --algorithms ppo,sac,deprl,msgym,middleware `
  --env MSKBenchWalk-v0 `
  --episodes 5 `
  --output-dir results\energy `
  --execute
```

Supported canonical algorithms:

| Canonical name | Accepted aliases | Notes |
|---|---|---|
| `ppo` | `ppo` | Stable-Baselines3 PPO baseline. |
| `sac` | `sac` | Stable-Baselines3 SAC baseline. |
| `deprl` | `deprl` | depRL baseline. |
| `msgym` | `msgym`, `dynsyn`, `dynsyn-sac` | DynSyn/msgym baseline. |
| `middleware` | `middleware`, `latent`, `latent-action` | Latent-action depRL middleware. |
| all algorithms | `all` | Expands to `ppo,sac,deprl,msgym,middleware`. |

Supported evaluator metrics and aliases:

| Canonical metric | Accepted aliases |
|---|---|
| `success` | `success`, `success_rate` |
| `robustness` | `robustness` |
| `smooth` | `smooth`, `smoothness` |
| `energy` | `energy`, `activation_energy` |
| `emg` | `emg`, `emg_similarity` |
| `render` | `render` |

Useful shared evaluator arguments:

| Argument | Purpose |
|---|---|
| `--env` | Environment ID or `all`. |
| `--episodes` | Number of evaluation episodes. |
| `--seed` | Evaluation seed where supported. |
| `--max-steps` | Optional episode step cap. |
| `--deterministic` | Deterministic action selection for algorithms that support it. |
| `--benchmark-root` | Repository root used to resolve scripts and assets. |
| `--json` / `--csv` | Explicit output files. Multi-algorithm runs suffix the algorithm name. |
| `--output-dir` | Output directory. Scalar metrics write `metric_algorithm.json/csv`; EMG and render use per-algorithm subdirectories. |
| `--model-root` | Root directory containing model artifacts. |
| `--model-path` | Direct model file for PPO, SAC, or msgym. |
| `--norm-path` | Observation-normalization file for PPO, SAC, or msgym. |
| `--run-path` | depRL or middleware run directory. |
| `--log-path` | msgym log path. |
| `--checkpoint-file` | depRL or middleware checkpoint file. |
| `--execute` | Actually run subprocesses instead of printing commands. |

Extra metric-specific arguments can be passed after `--`. Robustness scripts accept noise controls such as `--noise-type`, `--action-scales`, `--obs-scales`, and `--dynamics-scales`:

```powershell
python benchmark_eval\evaluate.py `
  --metric robustness `
  --algorithms ppo,sac `
  --env MSKBenchWalk-v0 `
  --episodes 30 `
  --execute `
  -- --noise-type action --action-scales 0,0.02,0.05,0.08,0.12
```

The package runner builds the same unified command surface:

```powershell
python -m msk_bench.benchmarking.runner --algorithm ppo --metric success --env MSKBenchWalk-v0 --episodes 10
```

## Baseline Scripts

Algorithm-specific scripts remain available and are used by the unified evaluator.

| Algorithm | Success | Robustness | Smoothness | Energy | EMG | Render |
|---|---|---|---|---|---|---|
| PPO | `ppo/eval_ppo_success.py` | `ppo/eval_ppo_robustness.py` | `ppo/eval_ppo_smooth.py` | `ppo/eval_ppo_energy.py` | `ppo/export_ppo_emg.py` | `ppo/render_ppo.py` |
| SAC | `sac/eval_sac_success.py` | `sac/eval_sac_robustness.py` | `sac/eval_sac_smooth.py` | `sac/eval_sac_energy.py` | `sac/export_sac_emg.py` | `sac/render_sac.py` |
| depRL | `depRL/eval_deprl_success.py` | `depRL/eval_deprl_robustness.py` | `depRL/eval_deprl_smooth.py` | `depRL/eval_deprl_energy.py` | `depRL/export_deprl_emg.py` | `depRL/render_deprl.py` |
| msgym | `msgym/eval_msgym_success.py` | `msgym/eval_msgym_robustness.py` | `msgym/eval_msgym_smooth.py` | `msgym/eval_msgym_energy.py` | `msgym/export_msgym_emg.py` | `msgym/render_msgym.py` |
| middleware | `deprl_middleware_22tasks/eval_middleware_success.py` | `deprl_middleware_22tasks/eval_middleware_robustness.py` | `deprl_middleware_22tasks/eval_middleware_smooth.py` | `deprl_middleware_22tasks/eval_middleware_energy.py` | `deprl_middleware_22tasks/export_middleware_emg.py` | `deprl_middleware_22tasks/render_middleware.py` |

Common direct evaluator arguments include `--env`, `--list-envs`, `--episodes`, `--benchmark-root`, `--model-root`, `--max-steps`, `--json`, and `--csv`. PPO/SAC/msgym evaluators also accept `--model-path` and `--norm-path`. depRL and middleware evaluators accept depRL-style run/checkpoint arguments such as `--run-path`, `--checkpoint`, `--checkpoint-file`, `--header`, `--agent`, and `--environment`/`--env-expr`.

## Training

Full benchmark training is compute-intensive and is not part of the test suite. PPO and SAC scripts default to long runs, so start with `--list-envs` and `--dry-run` before launching a full experiment.

### PPO

```powershell
python ppo\train_ppo_msk_bench.py --list-envs
python ppo\train_ppo_msk_bench.py --env MSKBenchWalk-v0 --dry-run
python ppo\train_ppo_msk_bench.py `
  --env MSKBenchWalk-v0 `
  --total-timesteps 200000000 `
  --eval-freq 500000 `
  --eval-episodes 10 `
  --output-root ppo\runs
```

PPO defaults include 8 training envs, 2 eval envs, seed 0, learning rate `3e-4`, `n_steps=2048`, batch size 256, 10 epochs, gamma 0.99, GAE lambda 0.95, clip range 0.2, entropy coefficient 0.005, and resume enabled. Use `--no-resume` to start fresh.

### SAC

```powershell
python sac\train_sac_msk_bench.py --list-envs
python sac\train_sac_msk_bench.py --env MSKBenchWalk-v0 --dry-run
python sac\train_sac_msk_bench.py `
  --env MSKBenchWalk-v0 `
  --total-timesteps 200000000 `
  --eval-freq 200000 `
  --eval-episodes 10 `
  --output-root sac\runs
```

SAC defaults include 8 training envs, 2 eval envs, seed 0, learning rate `1e-4`, buffer size 1,000,000, batch size 256, gamma 0.99, tau 0.005, train frequency 1, gradient steps 1, entropy coefficient 0.05, and resume enabled. Use `--no-resume` to start fresh.

### Latent-Action Middleware

The middleware package registers `-Middleware-v0` variants for the 22 canonical tasks and wraps base environments with `BioMiddlewareWrapper`. Modes are selected from the registry: `residual` for Balance, DoorOpen, and ChinUp; `primate_bimanual` for Reach and Catch; `hard` for the remaining canonical tasks unless overridden.

Useful middleware commands:

```powershell
python deprl_middleware_22tasks\generate_configs.py `
  --output-dir deprl_middleware_22tasks\configs `
  --encoder-path artifacts\encoder.pt `
  --decoder-path artifacts\decoder.pt

python deprl_middleware_22tasks\collect_expert_data.py `
  --env MSKBenchReach-v0 `
  --checkpoint-dir checkpoints\deprl\reach `
  --samples 50000 `
  --output artifacts\expert_synergy.pt

python deprl_middleware_22tasks\train_expert_transformer.py `
  --data artifacts\expert_synergy.pt `
  --output-dir artifacts `
  --latent-dim 64 `
  --epochs 40 `
  --batch-size 256 `
  --learning-rate 1e-4
```

## Development And Testing

Run the full regression suite:

```powershell
cd D:\MSK-Bench
python -B -m unittest discover -s tests -v
```

Run focused suites:

```powershell
python -B -m unittest tests.test_benchmark_metadata_and_analysis -v
python -B -m unittest tests.test_emg_similarity_evaluator tests.test_peak_efficiency -v
python -B -m unittest tests.test_unified_evaluation_entrypoint tests.test_unified_runner_entrypoint -v
```

Important test coverage:

- `test_benchmark_metadata_and_analysis.py`: task taxonomy, suites, metric helpers, runner validation, and command construction.
- `test_emg_similarity_evaluator.py`: cycle splitting and phase-aligned EMG similarity.
- `test_peak_efficiency.py`: maximum logged-step semantics and step aliases.
- `test_packaging_runtime_dependencies.py`: required default runtime dependencies in `pyproject.toml`.
- `test_unified_evaluation_entrypoint.py`: multi-algorithm command construction.
- `test_unified_runner_entrypoint.py`: package runner command construction.
- 	est_minimal_layout.py: normalized repository layout and absence of committed Python caches.
- 	est_third_party_attribution.py: upstream URLs, license references, local patch summaries, and baseline attribution README coverage.

The Ruff configuration targets Python 3.11, uses a 120-character line length, and currently selects `E` and `F` lint rules while ignoring `E501` and `E402`. Large upstream or compatibility areas are excluded from linting in `pyproject.toml`.

## Third-Party Components And Local Patches

Some baseline directories include adapted or vendored open-source code. The local README files in those directories describe the MSK-Bench integration surface and should not be replaced wholesale with upstream README files.

Release-facing attribution is tracked in two places:

- `THIRD_PARTY_NOTICES.md`: component paths, upstream URLs, license-file locations, and a short description of local MSK-Bench changes.
- `PATCHES.md`: maintainer-oriented summaries of how upstream-derived baseline code was modified for this repository.

Before redistributing a modified baseline directory, preserve its upstream license files, keep the local attribution README in place, and update both notice files if the upstream source, license, path, or local integration changes.
## Artifact And Cache Policy

Generated files that should stay out of version control include:

- `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`.
- Build artifacts such as `build/`, `dist/`, and `*.egg-info/`.
- Local experiment directories such as `outputs/`, `runs/`, and `wandb/`.
- Local notebooks checkpoints, temporary logs, videos, and machine-specific editor files.

The repository layout tests also expect these legacy or nested paths not to exist at the root:

- `MSK-Bench/` nested inside the repository.
- `scripts/` as a legacy top-level folder.
- `draw_emg_12_muscles.py` as a legacy top-level script.

If cache directories appear after manual local commands, remove only repository-local `__pycache__/` directories and rerun tests with `python -B`.

## Troubleshooting

### `ModuleNotFoundError: gymnasium` Or No Registered Environments

Install the default runtime dependencies:

```powershell
python -m pip install -e .
```

Then import `msk_bench` before calling `gym.make()`.

### MuJoCo Rendering Fails On A Headless Machine

The shared evaluation helpers use these environment defaults for headless rendering where applicable:

```powershell
$env:JAX_PLATFORMS = "cpu"
$env:MUJOCO_GL = "egl"
$env:PYOPENGL_PLATFORM = "egl"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

Local machines with visible OpenGL may need a different MuJoCo GL backend.

### Evaluators Cannot Find A Model Or Checkpoint

Pass the appropriate artifact argument for the selected algorithm:

- PPO/SAC/msgym: use `--model-root`, `--model-path`, and `--norm-path` as needed.
- depRL/middleware: use `--run-path`, `--checkpoint`, or `--checkpoint-file`.
- Multi-algorithm runs: prefer `--output-dir` so each algorithm receives a distinct output path.

### EMG Similarity Returns `nan` Or `0.0`

Common causes are empty signals, non-finite values, constant signals with no variance, too few samples to form cycles, or a reference envelope that is also constant. The evaluator falls back to whole-signal resampling when it cannot detect at least two peaks.

### Layout Test Fails Because Of `__pycache__`

Run tests with `python -B` and remove only repository-local cache directories. Do not delete unrelated user files or external artifacts.

## License Status

The top-level `LICENSE` file is currently a license-status notice: the wrapper code is intended for open-source release, but the final unified license requires author confirmation before publication. Do not remove or overwrite upstream license files in nested third-party components. Preserve each upstream license in its original location and add a complete third-party notices file before public redistribution if required.


