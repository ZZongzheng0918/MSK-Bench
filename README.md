# MSK-Bench Workspace

This workspace integrates the MSK-Bench simulation environments, training algorithms, and a MuscleMimic-based residual-control extension. ResidualRL uses the MuscleMimic MyoFullBody model and a pretrained PPO base policy, then adds residual actions produced by the MSK-Bench agent to the base actions. This preserves the learned motion prior while allowing adaptation to new tasks and terrains.

## Directory Structure

- `MSK-Bench/`: Contains the `msk_bench` Python package, standard environments, and ResidualRL environments.
- `musclemimic-main/`: MuscleMimic source code with the full training, evaluation, and motion-retargeting functionality.
- `depRL/`, `ppo/`, `sac/`, `msgym/`: Training algorithms and related entry points used by MSK-Bench.
- `tests/`: Workspace interface tests that can run without loading a model checkpoint.

ResidualRL does not include MuscleMimic model checkpoints and no longer contains a bundled copy of the `musclemimic-models` source code.

## System Requirements

MuscleMimic requires Python 3.11 or later and uses `uv` for dependency management. According to the upstream MuscleMimic documentation:

- Training requires Linux, an NVIDIA GPU, and a compatible CUDA/JAX environment.
- Inference and evaluation are officially supported on Linux and macOS.
- Windows is not officially supported upstream. Static tests can be run on Windows, but a full JAX/MuJoCo setup will generally require Linux or WSL2.

## Installation

First, install [uv](https://docs.astral.sh/uv/). Then enter the MuscleMimic project directory and synchronize the official dependencies:

```bash
cd musclemimic-main
uv sync
```

ResidualRL also depends on Gymnasium. Because Gymnasium is an MSK-Bench integration-layer dependency rather than a core MuscleMimic dependency, install it into the same `uv` environment:

```bash
uv pip install gymnasium==0.29.1
```

If you run `uv sync` again later, reinstall Gymnasium using the command above. When running MSK-Bench commands, use `uv run --no-sync` to prevent `uv` from removing this integration-layer dependency before startup.

The MuscleMimic `pyproject.toml` already declares `musclemimic-models>=1.0.2`. Therefore, `uv sync` automatically installs the officially released `musclemimic-models` package and the `myofullbody.xml` file. There is no need to copy the model source code into the `residualrl` directory.

### CUDA Training Environment

On Linux x86_64 with an NVIDIA GPU, install the optional CUDA dependencies according to the upstream instructions:

```bash
uv sync --extra cuda
uv pip install gymnasium==0.29.1
```

## Configure the Python Path

When running from the `musclemimic-main` directory, add the MSK-Bench package directory to `PYTHONPATH`.

### PowerShell

```powershell
$env:PYTHONPATH = (Resolve-Path '..\MSK-Bench').Path
uv run --no-sync python -c "import msk_bench; print('MSK-Bench registered')"
```

### Linux/macOS

```bash
export PYTHONPATH="$(cd ../MSK-Bench && pwd)"
uv run --no-sync python -c "import msk_bench; print('MSK-Bench registered')"
```

## MuscleMimic Model Checkpoint

The default base policy is loaded from:

```text
hf://amathislab/mm-fullbody-base
```

The MuscleMimic checkpoint loader supports the following sources:

1. A Hugging Face URI;
2. A local `checkpoint_<step>` directory;
3. A parent directory containing multiple checkpoints.

The first time the default Hugging Face URI is used, the model will be downloaded from the internet. A Hugging Face login or permission to access the resource may be required.

It is recommended to download or cache the checkpoint in advance and specify the local directory through an environment variable.

### PowerShell

```powershell
$env:MSK_BENCH_MUSCLEMIMIC_CHECKPOINT = 'D:\checkpoints\mm-fullbody-base'
```

### Linux/macOS

```bash
export MSK_BENCH_MUSCLEMIMIC_CHECKPOINT=/data/checkpoints/mm-fullbody-base
```

You can also pass the `base_model_dir` argument directly when creating the environment. The base-policy path is resolved in the following order:

1. The `base_model_dir` argument;
2. The `MSK_BENCH_MUSCLEMIMIC_CHECKPOINT` environment variable;
3. The official Hugging Face URI.

Model checkpoints are not distributed with this repository.

## ResidualRL Environments

The following environments are currently available:

- `MSKBenchResidualWalk-v0`
- `MSKBenchResidualRun-v0`
- `MSKBenchResidualStair-v0`

Minimal example:

```python
import gymnasium as gym
import msk_bench  # Importing the package automatically registers the environments

env = gym.make("MSKBenchResidualWalk-v0")
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```

Run the script from the `musclemimic-main` directory:

```bash
uv run --no-sync python path/to/your_script.py
```

The base policy is loaded the first time `step()` is called. If the default Hugging Face URI is used, the model files may be downloaded automatically at that point.

## Custom Model Path

Normally, you do not need to pass `model_path` manually because `MyoFullBody` resolves the default XML file through the installed `musclemimic-models` package.

To use a custom model, pass the path to an existing XML file. If the path is invalid, the program raises a `FileNotFoundError` containing the absolute path before MuJoCo attempts to load the model.

The three reference trajectory files remain stored under:

```text
MSK-Bench/msk_bench/envs/msk/benchmark/residualrl/
```

## Testing

### Workspace Tests Without a Checkpoint

```bash
cd D:/MSK-Bench
python -m unittest discover -s tests -v
```

### MuscleMimic Integration Tests

Run the adapter tests inside a complete MuscleMimic environment:

```bash
cd musclemimic-main
uv run --no-sync pytest -p no:cacheprovider tests/unit/test_msk_bench_integration.py -v
```

A complete smoke test also requires a valid model checkpoint. It should verify that all three ResidualRL environments can complete `reset()` and at least one `step()` call.

## Troubleshooting

### Gymnasium Is Missing

If either of the following errors appears:

```text
Either gym or gymnasium is required
```

or:

```text
No module named 'gymnasium'
```

run:

```bash
cd musclemimic-main
uv pip install gymnasium==0.29.1
```

Then run the program with `uv run --no-sync`.

### `musclemimic` or `musclemimic_models` Is Missing

Confirm that the current directory is `musclemimic-main` and that you have already run:

```bash
uv sync
```

Do not add the removed `musclemimic_models-main` source directory back to `PYTHONPATH`.

### Model Checkpoint Not Found

Check whether `MSK_BENCH_MUSCLEMIMIC_CHECKPOINT` points to a complete Orbax checkpoint directory or to a parent directory containing checkpoints.

You can also clear the environment variable so that the loader falls back to the default Hugging Face URI.

### Observation or Action Dimension Mismatch

The current ResidualRL policy interface expects:

- Observation dimension: 2418;
- Action dimension: 354.

Use the official base policy that matches the MyoFullBody configuration with finger actions disabled.

### `uv lock` Takes a Long Time to Resolve Git Extras

The optional SMPL/GMR dependencies in MuscleMimic include Git-based dependencies. Resolving the lock file for the first time may require access to GitHub and can therefore take a long time.

For a standard installation, prefer the existing `uv.lock` file in the repository and run:

```bash
uv sync
```

## License and Resource Terms

The MuscleMimic source code is licensed under Apache-2.0. Model files, motion data, and Hugging Face checkpoints may use separate licenses.

Before using or redistributing these resources, review the licensing terms in the corresponding code repository, model repository, and dataset pages. This MSK-Bench workspace does not redistribute the model checkpoints.
