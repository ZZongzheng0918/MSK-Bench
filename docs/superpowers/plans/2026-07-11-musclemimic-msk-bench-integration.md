# MuscleMimic MSK-Bench Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean the local MuscleMimic checkout without removing training/evaluation/retargeting capabilities, expose a stable MSK-Bench integration API, align ResidualRL imports and resource resolution, and document setup for the complete benchmark workspace.

**Architecture:** Keep MuscleMimic as an editable external dependency and add one focused `musclemimic.integrations.msk_bench` adapter for checkpoint selection and PPO inference. ResidualRL retains trajectory/reward/environment code but delegates model defaults and policy loading to the adapter. The benchmark does not vendor model-package source or checkpoints.

**Tech Stack:** Python 3.11+, JAX, MuJoCo 3.4, Gymnasium 0.29.1, OmegaConf, unittest/pytest, uv, PowerShell.

---

## File map

- Create `musclemimic-main/musclemimic/integrations/__init__.py`: public integration namespace.
- Create `musclemimic-main/musclemimic/integrations/msk_bench.py`: checkpoint resolution, cached PPO inference, and `MyoFullBody` export.
- Create `musclemimic-main/tests/unit/test_msk_bench_integration.py`: adapter unit tests.
- Modify `musclemimic-main/pyproject.toml`: add the `benchmark` optional dependency.
- Modify `musclemimic-main/uv.lock`: lock the optional Gymnasium dependency.
- Create `tests/test_residualrl_musclemimic_contract.py`: lightweight workspace contract tests that do not download checkpoints.
- Modify `MSK-Bench/msk_bench/envs/msk/benchmark/residualrl/common.py`: use the stable adapter and correct model/checkpoint defaults.
- Create `README.md`: workspace installation, checkpoint, environment use, and troubleshooting guide.
- Delete only paths listed in the approved design cleanup section.

### Task 1: Establish cleanup and import contracts

**Files:**
- Create: `tests/test_residualrl_musclemimic_contract.py`

- [ ] **Step 1: Write failing static contract tests**

```python
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "MSK-Bench/msk_bench/envs/msk/benchmark/residualrl/common.py"
MUSCLEMIMIC_ROOT = ROOT / "musclemimic-main"
RESIDUAL_ROOT = COMMON.parent


class ResidualRLMuscleMimicContractTests(unittest.TestCase):
    def test_common_imports_only_public_msk_bench_adapter(self):
        tree = ast.parse(COMMON.read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertIn("musclemimic.integrations.msk_bench", imported)
        self.assertNotIn("musclemimic.algorithms", imported)
        self.assertNotIn("musclemimic.runner.eval_utils", imported)
        self.assertNotIn("musclemimic.environments.humanoids.myofullbody", imported)

    def test_common_has_no_missing_bundled_defaults(self):
        text = COMMON.read_text(encoding="utf-8")
        self.assertNotIn('RESIDUAL_DIR / "models" / "myofullbody.xml"', text)
        self.assertNotIn('RESIDUAL_DIR / "base_policy"', text)
        self.assertIn("MSK_BENCH_MUSCLEMIMIC_CHECKPOINT", text)

    def test_cleaned_checkout_retains_core_capabilities(self):
        for relative in (
            "musclemimic",
            "loco_mujoco",
            "bimanual",
            "fullbody",
            "scripts",
            "tests",
            "pyproject.toml",
            "uv.lock",
            "retarget.py",
            "run_my_retarget.py",
        ):
            self.assertTrue((MUSCLEMIMIC_ROOT / relative).exists(), relative)

    def test_peripheral_and_vendored_paths_are_removed(self):
        for relative in ("outputs", "examples", ".github", "musclemimic.egg-info", "tea_debug.log"):
            self.assertFalse((MUSCLEMIMIC_ROOT / relative).exists(), relative)
        self.assertFalse((RESIDUAL_ROOT / "musclemimic_models-main").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm the expected failures**

Run: `python -m unittest tests.test_residualrl_musclemimic_contract -v`

Expected: failures for the old internal imports, missing default paths, adapter module, and cleanup paths.

- [ ] **Step 3: Verify cleanup targets resolve inside approved directories**

Run this PowerShell check before deletion:

```powershell
$roots = @(
  (Resolve-Path 'D:\MSK-Bench\musclemimic-main').Path,
  (Resolve-Path 'D:\MSK-Bench\MSK-Bench\msk_bench\envs\msk\benchmark\residualrl').Path
)
$targets = @(
  'D:\MSK-Bench\musclemimic-main\outputs',
  'D:\MSK-Bench\musclemimic-main\examples',
  'D:\MSK-Bench\musclemimic-main\.github',
  'D:\MSK-Bench\musclemimic-main\musclemimic.egg-info',
  'D:\MSK-Bench\musclemimic-main\tea_debug.log',
  'D:\MSK-Bench\MSK-Bench\msk_bench\envs\msk\benchmark\residualrl\musclemimic_models-main'
)
$targets | ForEach-Object {
  $absolute = [IO.Path]::GetFullPath($_)
  if (-not ($roots | Where-Object { $absolute.StartsWith($_ + [IO.Path]::DirectorySeparatorChar) })) {
    throw "Unsafe cleanup target: $absolute"
  }
  $absolute
}
```

Expected: six absolute paths under the two approved roots and no exception.

- [ ] **Step 4: Remove approved peripheral paths and generated caches**

Run only after Step 3 succeeds:

```powershell
$muscleRoot = (Resolve-Path 'D:\MSK-Bench\musclemimic-main').Path
$residualRoot = (Resolve-Path 'D:\MSK-Bench\MSK-Bench\msk_bench\envs\msk\benchmark\residualrl').Path
$targets = @(
  "$muscleRoot\outputs",
  "$muscleRoot\examples",
  "$muscleRoot\.github",
  "$muscleRoot\musclemimic.egg-info",
  "$muscleRoot\tea_debug.log",
  "$residualRoot\musclemimic_models-main"
)
foreach ($target in $targets) {
  $absolute = [IO.Path]::GetFullPath($target)
  $safe = $absolute.StartsWith($muscleRoot + [IO.Path]::DirectorySeparatorChar) -or
          $absolute.StartsWith($residualRoot + [IO.Path]::DirectorySeparatorChar)
  if (-not $safe) { throw "Unsafe cleanup target: $absolute" }
  if (Test-Path -LiteralPath $absolute) {
    Remove-Item -LiteralPath $absolute -Recurse -Force
  }
}

$cacheDirs = Get-ChildItem -LiteralPath $muscleRoot -Directory -Recurse -Force |
  Where-Object { $_.Name -in @('__pycache__', '.pytest_cache', '.ruff_cache') }
foreach ($cacheDir in $cacheDirs) {
  $absolute = [IO.Path]::GetFullPath($cacheDir.FullName)
  if (-not $absolute.StartsWith($muscleRoot + [IO.Path]::DirectorySeparatorChar)) {
    throw "Unsafe cache target: $absolute"
  }
  Remove-Item -LiteralPath $absolute -Recurse -Force
}
```

- [ ] **Step 5: Re-run the cleanup-related tests**

Run: `python -m unittest tests.test_residualrl_musclemimic_contract.ResidualRLMuscleMimicContractTests.test_cleaned_checkout_retains_core_capabilities tests.test_residualrl_musclemimic_contract.ResidualRLMuscleMimicContractTests.test_peripheral_and_vendored_paths_are_removed -v`

Expected: 2 tests pass.

- [ ] **Step 6: Commit the cleaned retained baseline and contract test**

```bash
git add musclemimic-main MSK-Bench/msk_bench/envs/msk/benchmark/residualrl tests/test_residualrl_musclemimic_contract.py
git commit -m "chore: add cleaned MuscleMimic benchmark baseline"
```

Before committing, confirm `git status --short` contains no deleted path outside the approved list.

### Task 2: Add the stable MuscleMimic adapter

**Files:**
- Create: `musclemimic-main/musclemimic/integrations/__init__.py`
- Create: `musclemimic-main/musclemimic/integrations/msk_bench.py`
- Create: `musclemimic-main/tests/unit/test_msk_bench_integration.py`

- [ ] **Step 1: Write failing adapter tests**

```python
import numpy as np
import pytest

from musclemimic.integrations import msk_bench


def test_checkpoint_source_precedence(monkeypatch):
    monkeypatch.setenv(msk_bench.CHECKPOINT_ENV_VAR, "/env/checkpoint")
    assert msk_bench.resolve_checkpoint_source("/argument/checkpoint") == "/argument/checkpoint"
    assert msk_bench.resolve_checkpoint_source(None) == "/env/checkpoint"
    monkeypatch.delenv(msk_bench.CHECKPOINT_ENV_VAR)
    assert msk_bench.resolve_checkpoint_source(None) == msk_bench.DEFAULT_CHECKPOINT_SOURCE


def test_blank_checkpoint_values_fall_back(monkeypatch):
    monkeypatch.setenv(msk_bench.CHECKPOINT_ENV_VAR, "   ")
    assert msk_bench.resolve_checkpoint_source("  ") == msk_bench.DEFAULT_CHECKPOINT_SOURCE


def test_policy_input_dimension_is_validated():
    with pytest.raises(ValueError, match=r"expected observation dimension 4, got 3"):
        msk_bench.validate_policy_observation(np.zeros(3, dtype=np.float32), 4)


def test_policy_action_dimension_is_validated():
    with pytest.raises(ValueError, match=r"expected action dimension 2, got 3"):
        msk_bench.validate_policy_action(np.zeros(3, dtype=np.float32), 2)
```

- [ ] **Step 2: Run tests to verify import failure**

Run: `uv run pytest tests/unit/test_msk_bench_integration.py -v`

Expected: FAIL because `musclemimic.integrations` does not exist.

- [ ] **Step 3: Implement the public adapter**

Create `musclemimic/integrations/__init__.py`:

```python
"""Stable integration APIs for external projects."""
```

Create `musclemimic/integrations/msk_bench.py` with:

```python
from __future__ import annotations

import os
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

CHECKPOINT_ENV_VAR = "MSK_BENCH_MUSCLEMIMIC_CHECKPOINT"
DEFAULT_CHECKPOINT_SOURCE = "hf://amathislab/mm-fullbody-base"


def resolve_checkpoint_source(value: str | os.PathLike | None = None) -> str:
    explicit = "" if value is None else str(value).strip()
    if explicit:
        return explicit
    configured = os.environ.get(CHECKPOINT_ENV_VAR, "").strip()
    return configured or DEFAULT_CHECKPOINT_SOURCE


def validate_policy_observation(observation, expected_dim: int) -> np.ndarray:
    value = np.asarray(observation, dtype=np.float32).reshape(-1)
    if value.size != expected_dim:
        raise ValueError(f"MuscleMimic policy expected observation dimension {expected_dim}, got {value.size}")
    return value


def validate_policy_action(action, expected_dim: int) -> np.ndarray:
    value = np.asarray(action, dtype=np.float32).reshape(-1)
    if value.size != expected_dim:
        raise ValueError(f"MuscleMimic policy expected action dimension {expected_dim}, got {value.size}")
    return value


@lru_cache(maxsize=None)
def get_jax_policy(checkpoint_source: str | None, expected_obs_dim: int, act_dim: int):
    import jax
    import jax.numpy as jnp
    from gymnasium import spaces
    from omegaconf import OmegaConf

    from musclemimic.algorithms.ppo import PPOJax
    from musclemimic.runner.eval_utils import align_agent_state, load_checkpoint

    source = resolve_checkpoint_source(checkpoint_source)
    config, agent_state, _ = load_checkpoint(source)
    OmegaConf.set_struct(config, False)

    class DummyEnv:
        def __init__(self):
            self.observation_space = spaces.Box(-np.inf, np.inf, (expected_obs_dim,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (act_dim,), dtype=np.float32)
            self.mdp_info = SimpleNamespace(
                observation_space=self.observation_space,
                action_space=self.action_space,
            )
            self.info = self.mdp_info

    agent_conf = PPOJax.init_agent_conf(DummyEnv(), config)
    train_state = align_agent_state(agent_state, agent_conf).train_state

    @jax.jit
    def apply_policy(state, observation):
        variables = {"params": state.params, "run_stats": state.run_stats}
        output, _ = agent_conf.network.apply(
            variables, jnp.atleast_2d(observation), mutable=["run_stats"]
        )
        return jnp.squeeze(output[0].mean())

    def policy(state, observation):
        checked_observation = validate_policy_observation(observation, expected_obs_dim)
        action = apply_policy(state, checked_observation)
        return validate_policy_action(action, act_dim)

    return policy, train_state


from musclemimic.environments.humanoids import MyoFullBody  # noqa: E402

__all__ = [
    "CHECKPOINT_ENV_VAR",
    "DEFAULT_CHECKPOINT_SOURCE",
    "MyoFullBody",
    "get_jax_policy",
    "resolve_checkpoint_source",
    "validate_policy_action",
    "validate_policy_observation",
]
```

- [ ] **Step 4: Run adapter tests**

Run: `uv run pytest tests/unit/test_msk_bench_integration.py -v`

Expected: 4 tests pass without downloading a checkpoint.

- [ ] **Step 5: Commit the adapter**

```bash
git add musclemimic-main/musclemimic/integrations musclemimic-main/tests/unit/test_msk_bench_integration.py
git commit -m "feat: expose MSK-Bench MuscleMimic adapter"
```

### Task 3: Align ResidualRL resource resolution and imports

**Files:**
- Modify: `MSK-Bench/msk_bench/envs/msk/benchmark/residualrl/common.py`
- Modify: `tests/test_residualrl_musclemimic_contract.py`

- [ ] **Step 1: Add failing behavior tests using stub modules**

Add these imports and helper to the workspace test:

```python
import importlib.util
import os
import sys
import types
from unittest.mock import patch


def load_common_with_stubs():
    gymnasium = types.ModuleType("gymnasium")
    gymnasium.Env = type("Env", (), {})
    gymnasium.Wrapper = type("Wrapper", (), {})
    spaces = types.ModuleType("gymnasium.spaces")
    spaces.Box = type("Box", (), {})
    gymnasium.spaces = spaces

    mujoco = types.ModuleType("mujoco")
    mujoco.MjData = type("MjData", (), {})
    scipy = types.ModuleType("scipy")
    scipy_spatial = types.ModuleType("scipy.spatial")
    scipy_transform = types.ModuleType("scipy.spatial.transform")
    scipy_transform.Rotation = type("Rotation", (), {})

    loco_math = types.ModuleType("loco_mujoco.core.utils.math")
    loco_math.calculate_relative_site_quantities = lambda *args, **kwargs: None
    adapter = types.ModuleType("musclemimic.integrations.msk_bench")
    adapter.CHECKPOINT_ENV_VAR = "MSK_BENCH_MUSCLEMIMIC_CHECKPOINT"
    adapter.MyoFullBody = type("MyoFullBody", (), {})
    adapter.get_jax_policy = lambda *args, **kwargs: None
    adapter.resolve_checkpoint_source = lambda value=None: (
        str(value).strip() if value is not None and str(value).strip()
        else os.environ.get(adapter.CHECKPOINT_ENV_VAR, "").strip()
        or "hf://amathislab/mm-fullbody-base"
    )

    stubs = {
        "gymnasium": gymnasium,
        "gymnasium.spaces": spaces,
        "mujoco": mujoco,
        "scipy": scipy,
        "scipy.spatial": scipy_spatial,
        "scipy.spatial.transform": scipy_transform,
        "loco_mujoco": types.ModuleType("loco_mujoco"),
        "loco_mujoco.core": types.ModuleType("loco_mujoco.core"),
        "loco_mujoco.core.utils": types.ModuleType("loco_mujoco.core.utils"),
        "loco_mujoco.core.utils.math": loco_math,
        "musclemimic": types.ModuleType("musclemimic"),
        "musclemimic.integrations": types.ModuleType("musclemimic.integrations"),
        "musclemimic.integrations.msk_bench": adapter,
    }
    spec = importlib.util.spec_from_file_location("residualrl_common_under_test", COMMON)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module
```

Add this test method:

```python
def test_model_and_checkpoint_defaults_are_portable(self):
    module = load_common_with_stubs()
    self.assertIsNone(module.resolve_model_path(None))
    self.assertEqual(module.default_base_model_dir(None), "hf://amathislab/mm-fullbody-base")
    with self.assertRaisesRegex(FileNotFoundError, "Model XML does not exist"):
        module.resolve_model_path(ROOT / "missing-model.xml")
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `python -m unittest tests.test_residualrl_musclemimic_contract -v`

Expected: static import/default tests and the new path tests fail against the old implementation.

- [ ] **Step 3: Replace internal imports and defaults in `common.py`**

Use this public import:

```python
try:
    from musclemimic.integrations.msk_bench import (
        CHECKPOINT_ENV_VAR,
        MyoFullBody,
        get_jax_policy,
        resolve_checkpoint_source,
    )
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("musclemimic"):
        raise ModuleNotFoundError(
            "ResidualRL requires MuscleMimic. Install the local musclemimic project "
            "with the benchmark extra; see the MSK-Bench workspace README."
        ) from exc
    raise
```

Remove `DEFAULT_MODEL_PATH`, `DEFAULT_BASE_MODEL_DIR`, the local policy cache, and the local `get_jax_policy` implementation. Add:

```python
def resolve_model_path(value: str | os.PathLike | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model XML does not exist: {path}")
    return str(path)


def default_base_model_dir(value=None) -> str:
    return resolve_checkpoint_source(value)
```

In `FullBodyReferenceEnv.__init__`, change model resolution to `model_path = resolve_model_path(model_path)`. Keep trajectory path resolution unchanged. Pass `spec=model_path`; `None` intentionally activates MuscleMimic's installed `musclemimic_models` default.

- [ ] **Step 4: Run static and stubbed tests**

Run: `python -m unittest tests.test_residualrl_musclemimic_contract -v`

Expected: all contract tests pass and no checkpoint download occurs.

- [ ] **Step 5: Commit ResidualRL alignment**

```bash
git add MSK-Bench/msk_bench/envs/msk/benchmark/residualrl/common.py tests/test_residualrl_musclemimic_contract.py
git commit -m "fix: align ResidualRL with MuscleMimic adapter"
```

### Task 4: Add an explicit benchmark dependency extra

**Files:**
- Modify: `musclemimic-main/pyproject.toml`
- Modify: `musclemimic-main/uv.lock`
- Modify: `tests/test_residualrl_musclemimic_contract.py`

- [ ] **Step 1: Add a failing packaging assertion**

Use `tomllib` in the workspace test:

```python
def test_musclemimic_declares_benchmark_extra(self):
    import tomllib

    with (MUSCLEMIMIC_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    self.assertIn("gymnasium==0.29.1", project["optional-dependencies"]["benchmark"])
```

- [ ] **Step 2: Run the test and confirm failure**

Run: `python -m unittest tests.test_residualrl_musclemimic_contract.ResidualRLMuscleMimicContractTests.test_musclemimic_declares_benchmark_extra -v`

Expected: FAIL because `benchmark` is absent.

- [ ] **Step 3: Add the extra and refresh the lock**

Add to `[project.optional-dependencies]`:

```toml
benchmark = [
    "gymnasium==0.29.1",
]
```

Run: `uv lock`

Expected: `uv.lock` updates successfully without changing unrelated direct dependency constraints.

- [ ] **Step 4: Run packaging and adapter tests**

Run: `python -m unittest tests.test_residualrl_musclemimic_contract -v`

Run: `uv run --extra benchmark pytest tests/unit/test_msk_bench_integration.py -v`

Expected: both suites pass.

- [ ] **Step 5: Commit dependency metadata**

```bash
git add musclemimic-main/pyproject.toml musclemimic-main/uv.lock tests/test_residualrl_musclemimic_contract.py
git commit -m "build: add MuscleMimic benchmark extra"
```

### Task 5: Write the workspace README

**Files:**
- Create: `README.md`
- Modify: `tests/test_residualrl_musclemimic_contract.py`

- [ ] **Step 1: Add failing README contract tests**

```python
def test_workspace_readme_documents_residualrl_setup(self):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for required in (
        "uv sync --extra benchmark",
        "hf://amathislab/mm-fullbody-base",
        "MSK_BENCH_MUSCLEMIMIC_CHECKPOINT",
        "MSKBenchResidualWalk-v0",
        "MSKBenchResidualRun-v0",
        "MSKBenchResidualStair-v0",
        "musclemimic-models",
        "checkpoint",
        "license",
    ):
        self.assertIn(required, readme)
```

- [ ] **Step 2: Run the README test and confirm failure**

Run: `python -m unittest tests.test_residualrl_musclemimic_contract.ResidualRLMuscleMimicContractTests.test_workspace_readme_documents_residualrl_setup -v`

Expected: FAIL because the root README does not exist.

- [ ] **Step 3: Write `D:\MSK-Bench\README.md`**

The README must contain executable sections for:

```bash
cd musclemimic-main
uv sync --extra benchmark
```

```powershell
$env:PYTHONPATH = "D:\MSK-Bench\MSK-Bench"
$env:MSK_BENCH_MUSCLEMIMIC_CHECKPOINT = "D:\checkpoints\mm-fullbody-base"
uv run --extra benchmark python -c "import msk_bench; print('MSK-Bench registered')"
```

and a Python smoke example:

```python
import gymnasium as gym
import msk_bench  # registers environments

env = gym.make("MSKBenchResidualWalk-v0")
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

Also document the three environment IDs, automatic `musclemimic-models` dependency, default Hugging Face checkpoint, local override, upstream platform limits, first-use download behavior, independent data/checkpoint licenses, and error remedies.

- [ ] **Step 4: Run README and full contract tests**

Run: `python -m unittest tests.test_residualrl_musclemimic_contract -v`

Expected: all tests pass.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md tests/test_residualrl_musclemimic_contract.py
git commit -m "docs: add MSK-Bench workspace setup guide"
```

### Task 6: Verify without network or checkpoint

**Files:**
- No source changes expected.

- [ ] **Step 1: Check syntax for changed Python files**

Run:

```bash
python -m py_compile musclemimic-main/musclemimic/integrations/__init__.py musclemimic-main/musclemimic/integrations/msk_bench.py MSK-Bench/msk_bench/envs/msk/benchmark/residualrl/common.py tests/test_residualrl_musclemimic_contract.py
```

Expected: exit code 0.

- [ ] **Step 2: Run workspace contract tests**

Run: `python -m unittest tests.test_residualrl_musclemimic_contract -v`

Expected: all tests pass without network access or checkpoint files.

- [ ] **Step 3: Run MuscleMimic adapter tests in its uv environment**

Run: `uv run --extra benchmark pytest tests/unit/test_msk_bench_integration.py -v`

Expected: all adapter tests pass without downloading a checkpoint.

- [ ] **Step 4: Run lint on changed Python files**

Run:

```bash
uv run ruff check musclemimic/integrations tests/unit/test_msk_bench_integration.py ../MSK-Bench/msk_bench/envs/msk/benchmark/residualrl/common.py ../tests/test_residualrl_musclemimic_contract.py
```

Expected: no lint errors.

- [ ] **Step 5: Audit repository state**

Run: `git status --short`

Expected: only intended changes, with no deleted path outside the approved cleanup list.

### Task 7: Optional full-environment smoke verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Configure a real checkpoint**

Set `MSK_BENCH_MUSCLEMIMIC_CHECKPOINT` to a local complete Orbax checkpoint directory, or leave it unset to allow the official Hugging Face download.

- [ ] **Step 2: Import and register all environments**

Run from `musclemimic-main` with `D:\MSK-Bench\MSK-Bench` on `PYTHONPATH`:

```bash
uv run --extra benchmark python -c "import gymnasium as gym, msk_bench; print([k for k in gym.envs.registry if k.startswith('MSKBenchResidual')])"
```

Expected: the Walk, Run, and Stair residual IDs are listed.

- [ ] **Step 3: Reset and step each environment once**

Run:

```powershell
uv run --extra benchmark python -c "import gymnasium as gym, msk_bench; ids=('MSKBenchResidualWalk-v0','MSKBenchResidualRun-v0','MSKBenchResidualStair-v0'); [(lambda env: (env.reset(seed=0), (lambda result: result if len(result)==5 else (_ for _ in ()).throw(AssertionError('expected Gymnasium five-tuple')))(env.step(env.action_space.sample())), env.close()))(gym.make(env_id)) for env_id in ids]; print('three residual environments passed reset/step')"
```

Expected: all three environments complete one step. If the machine lacks compatible JAX/MuJoCo or checkpoint access, record this as an environment limitation rather than claiming the smoke test passed.

- [ ] **Step 4: Record final verification evidence**

Capture exact command outputs, passing test counts, and any skipped full-environment check in the final handoff. Do not report completion until Task 6 has passed.
