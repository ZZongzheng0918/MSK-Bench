import ast
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "MSK-Bench/msk_bench/envs/msk/benchmark/residualrl/common.py"
MUSCLEMIMIC_ROOT = ROOT / "musclemimic-main"
RESIDUAL_ROOT = COMMON.parent

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
        str(value).strip()
        if value is not None and str(value).strip()
        else os.environ.get(adapter.CHECKPOINT_ENV_VAR, "").strip()
        or "hf://amathislab/mm-fullbody-base"
    )

    legacy_myofullbody = types.ModuleType("musclemimic.environments.humanoids.myofullbody")
    legacy_myofullbody.MyoFullBody = adapter.MyoFullBody
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
        "musclemimic.environments": types.ModuleType("musclemimic.environments"),
        "musclemimic.environments.humanoids": types.ModuleType("musclemimic.environments.humanoids"),
        "musclemimic.environments.humanoids.myofullbody": legacy_myofullbody,
        "musclemimic.integrations": types.ModuleType("musclemimic.integrations"),
        "musclemimic.integrations.msk_bench": adapter,
    }
    spec = importlib.util.spec_from_file_location("residualrl_common_under_test", COMMON)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


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
        self.assertIn("CHECKPOINT_ENV_VAR", text)

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
            with self.subTest(relative=relative):
                self.assertTrue((MUSCLEMIMIC_ROOT / relative).exists(), relative)

    def test_peripheral_and_vendored_paths_are_removed(self):
        for relative in (
            "outputs",
            "examples",
            ".github",
            "musclemimic.egg-info",
            "tea_debug.log",
        ):
            with self.subTest(relative=relative):
                self.assertFalse((MUSCLEMIMIC_ROOT / relative).exists(), relative)
        self.assertFalse((RESIDUAL_ROOT / "musclemimic_models-main").exists())

    def test_model_and_checkpoint_defaults_are_portable(self):
        module = load_common_with_stubs()
        self.assertTrue(hasattr(module, "resolve_model_path"))
        self.assertIsNone(module.resolve_model_path(None))
        with patch.dict(os.environ, {"MSK_BENCH_MUSCLEMIMIC_CHECKPOINT": ""}):
            self.assertEqual(
                module.default_base_model_dir(None),
                "hf://amathislab/mm-fullbody-base",
            )
        with self.assertRaisesRegex(FileNotFoundError, "Model XML does not exist"):
            module.resolve_model_path(ROOT / "missing-model.xml")

    def test_workspace_readme_documents_residualrl_setup(self):
        readme_path = ROOT / "README.md"
        self.assertTrue(readme_path.is_file(), "root README.md is missing")
        readme = readme_path.read_text(encoding="utf-8")
        for required in (
            "uv sync",
            "uv pip install gymnasium==0.29.1",
            "hf://amathislab/mm-fullbody-base",
            "MSK_BENCH_MUSCLEMIMIC_CHECKPOINT",
            "MSKBenchResidualWalk-v0",
            "MSKBenchResidualRun-v0",
            "MSKBenchResidualStair-v0",
            "musclemimic-models",
            "checkpoint",
            "license",
        ):
            with self.subTest(required=required):
                self.assertIn(required, readme)


if __name__ == "__main__":
    unittest.main()
