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


if __name__ == "__main__":
    unittest.main()
