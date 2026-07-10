import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TASKS = [
    "MSKBenchStand-v0",
    "MSKBenchPowerlift-v0",
    "MSKBenchSingleLegStand-v0",
    "MSKBenchSit-v0",
    "MSKBenchBalance-v0",
    "MSKBenchSquat-v0",
    "MSKBenchWalk-v0",
    "MSKBenchCrawl-v0",
    "MSKBenchRun-v0",
    "MSKBenchJump-v0",
    "MSKBenchWalkTurn-v0",
    "MSKBenchSidestep-v0",
    "MSKBenchStairs-v0",
    "MSKBenchHurdle-v0",
    "MSKBenchStepStones-v0",
    "MSKBenchSlide-v0",
    "MSKBenchDoorOpen-v0",
    "MSKBenchReach-v0",
    "MSKBenchWalkAndSit-v0",
    "MSKBenchChinUp-v0",
    "MSKBenchCatch-v0",
    "MSKBenchPoleWalk-v0",
]
RESIDUAL_TASKS = [
    "MSKBenchResidualRun-v0",
    "MSKBenchResidualStair-v0",
    "MSKBenchResidualWalk-v0",
]
TRAINING_CONFIG_TASKS = EXPECTED_TASKS + RESIDUAL_TASKS


def snake_case_task(env_id):
    task = env_id.removeprefix("MSKBench").removesuffix("-v0")
    return re.sub(r"(?<!^)([A-Z])", r"_\1", task).lower()


class DepRLIntegrationTests(unittest.TestCase):
    def test_deprl_package_is_merged_without_generated_artifacts(self):
        self.assertTrue((ROOT / "deprl" / "__init__.py").exists())
        self.assertFalse((ROOT / "train_logs_powerlift").exists())
        generated = [
            path
            for path in ROOT.rglob("*")
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}
        ]
        self.assertEqual(generated, [])

    def test_pyproject_packages_and_training_dependencies_include_deprl(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("deprl*", pyproject)
        for dep in ["pyyaml", "pandas", "gdown", "wandb", "torch"]:
            with self.subTest(dep=dep):
                self.assertRegex(pyproject.lower(), rf'"{dep}([<>=!~].*)?"')
        self.assertIn("deprl-train", pyproject)

    def test_pyproject_discovers_external_msk_bench_package(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertRegex(pyproject, r'where\s*=\s*\[[^\]]*"\."[^\]]*"\.\./MSK-Bench"[^\]]*\]')
        self.assertIn('"msk_bench" = "../MSK-Bench/msk_bench"', pyproject)

    def test_deprl_gym_builder_does_not_require_myosuite_for_msk_bench(self):
        builder = ROOT / "deprl" / "vendor" / "tonic" / "environments" / "builders.py"
        self.assertTrue(builder.exists())
        text = builder.read_text(encoding="utf-8")
        self.assertNotRegex(text, r"(?m)^import myosuite\b")
        self.assertNotRegex(text, r"(?m)^from myosuite\b")
        self.assertIn("import msk_bench", text)

    def test_msk_bench_training_configs_cover_registered_tasks(self):
        config_dir = ROOT / "experiments" / "msk_bench_training_files"
        self.assertTrue(config_dir.exists())
        seen = {}
        for config in sorted(config_dir.glob("*.yaml")):
            text = config.read_text(encoding="utf-8")
            env_ids = re.findall(r"MSKBench[A-Za-z]+-v0", text)
            self.assertEqual(len(set(env_ids)), 1, config.name)
            env_id = env_ids[0]
            seen[env_id] = config.name
            self.assertIn("import msk_bench", text)
            self.assertNotIn("myosuite", text.lower())
            self.assertIn("deprl.environments.Gym", text)
            self.assertRegex(text, r"working_dir:\s*\.\/baselines_MSKBench")
        self.assertEqual(set(seen), set(TRAINING_CONFIG_TASKS))

    def test_msk_bench_training_config_names_are_canonical(self):
        config_dir = ROOT / "experiments" / "msk_bench_training_files"
        expected_names = {
            f"msk_bench_{snake_case_task(env_id)}.yaml"
            for env_id in TRAINING_CONFIG_TASKS
        }
        actual_names = {config.name for config in config_dir.glob("*.yaml")}
        self.assertEqual(actual_names, expected_names)

        readme = (config_dir / "README.md").read_text(encoding="utf-8")
        self.assertIn("msk_bench_powerlift.yaml", readme)
        self.assertTrue((config_dir / "msk_bench_powerlift.yaml").exists())


if __name__ == "__main__":
    unittest.main()