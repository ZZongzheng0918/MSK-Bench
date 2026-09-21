# GPT-6 direct muscle-control diagnostic

This directory contains source code for a latency-decoupled, single-seed diagnostic of direct 354-dimensional muscle control. The simulator pauses while a controller selects an action. Each action is held for 0.2 seconds, so the decision rate is 5 Hz of simulation time. Each rollout is capped at 3 seconds.

## Release scope

The source release intentionally excludes generated actions, controller-visible observations, trajectories, diagnostics, videos, figures, summaries, and reports. Local outputs are written under `experiments/gpt6_direct_control/results/` and ignored by Git. The experiment code does not record chain-of-thought or hidden reasoning.

No LLM client is included. The code exports a controller-visible observation, exits, and later accepts one validated action file. To reproduce a new LLM-driven rollout, record the exact model identifier, model settings, complete prompts, and submitted action files. This source-only release does not reproduce prior GPT decisions.

## Environment and dependencies

- Tasks: Walk, Run, and Stairs.
- Action dimension: 354.
- Physics timestep: 0.001 s; native control timestep: 0.01 s (100 Hz).
- GPT decision interval: 0.2 simulated seconds (5 Hz), with at most 15 decisions.
- Residual comparison: the repository MuscleMimic integration and the official
  `hf://amathislab/mm-fullbody-base` checkpoint, using zero residual correction.

The checkpoint is downloaded through `huggingface_hub`; it is not trained locally. Walk and Run motion priors originate from `amathislab/musclemimic-retargeted` and are not redistributed. Follow the root README to acquire them. The stair prior and its dedicated two-step MuJoCo XML (MJCF) model are repository-provided. No external source checkout is used.

Install the declared dependencies from the repository root:

```bash
pip install -e ".[residual]"
```

## Run the included Stairs smoke test

The Stairs task uses the included prior and two-step MJCF model. This command runs without the restricted Walk and Run motions or the pretrained checkpoint:

```bash
python -m experiments.gpt6_direct_control.paused_env_controller mock --task stairs --output-dir experiments/gpt6_direct_control/results/smoke --decisions 1
```

This is a deterministic zero-action pipeline check. It is never labeled as a GPT rollout.

## Inspect all direct environments

The following command requires both restricted Walk and Run motion files. Complete the acquisition steps in the root README before running it.

Inspect all three direct environments:

```bash
python -m experiments.gpt6_direct_control.paused_env_controller env-info \
  --output experiments/gpt6_direct_control/results/env_info.json
```

## Run a manual action session

Initialize one paused direct-control session:

```bash
python -m experiments.gpt6_direct_control.paused_env_controller init \
  --task stairs \
  --session-dir experiments/gpt6_direct_control/results/gpt_stairs
```

An action file is a bare JSON array containing exactly 354 finite numeric values within the action-space bounds. This command creates a neutral action that verifies the file interface:

```bash
python -c "import json; from pathlib import Path; Path('action_000.json').write_text(json.dumps([0.0] * 354), encoding='utf-8')"
```

Replace those values with the controller's proposed muscle excitations for an LLM-driven run, then submit one action:

```bash
python -m experiments.gpt6_direct_control.paused_env_controller submit \
  --session-dir experiments/gpt6_direct_control/results/gpt_stairs \
  --action-file action_000.json
```

Repeat `submit` with the next action while the manifest status is `awaiting_action`. Before choosing each action, the controller may read only `env_info.json` and the current `observations/decision_XXX_controller_visible.json`.

Run the pretrained residual baseline on Stairs:

```bash
python -m experiments.gpt6_direct_control.run_residual_baseline \
  --task stairs \
  --output-dir experiments/gpt6_direct_control/results \
  --decisions 15
```

The first run downloads the official checkpoint. Use `--task all` only after obtaining both restricted Walk and Run motion files.

Render any persisted trajectory without advancing simulation:

```bash
python -m experiments.gpt6_direct_control.render_rollout \
  --trajectory experiments/gpt6_direct_control/results/gpt_stairs/trajectory.npz \
  --output experiments/gpt6_direct_control/results/videos/stairs_gpt6_seed0.mp4 \
  --fps 30 --width 640 --height 480 --stop-at-first-fall
```

`analyze_failure.py` accepts repeated `--trajectory` arguments and writes CSV/JSON summaries, plots, key frames, and a narrative report to the selected `--output-dir`. These files are local generated artifacts and are not part of the source release.
