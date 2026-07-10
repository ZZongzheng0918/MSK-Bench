import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT.parent / "MSK-Bench" / "msk_bench" / "envs" / "msk" / "benchmark"
RESIDUAL_DIR = BENCHMARK_DIR / "residualrl"
CONFIG_DIR = ROOT / "experiments" / "msk_bench_training_files"


class ResidualRLBenchmarkTests(unittest.TestCase):
    def test_residual_package_has_shared_helpers_and_canonical_classes(self):
        self.assertTrue((RESIDUAL_DIR / "__init__.py").exists())
        self.assertTrue((RESIDUAL_DIR / "common.py").exists())

        run_text = (RESIDUAL_DIR / "run.py").read_text(encoding="utf-8")
        stair_text = (RESIDUAL_DIR / "stair.py").read_text(encoding="utf-8")
        walk_text = (RESIDUAL_DIR / "walk.py").read_text(encoding="utf-8")

        self.assertIn("class MSKBenchResidualRunEnvV0", run_text)
        self.assertIn("class MSKBenchResidualStairEnvV0", stair_text)
        self.assertIn("class MSKBenchResidualWalkEnvV0", walk_text)
        self.assertIn("alpha_stance = 0.2", walk_text)
        self.assertIn("alpha_swing = 0.6", walk_text)
        self.assertIn("metabolic_weight = 0.05", walk_text)
        self.assertIn("metabolic_smoothing = 0.1", walk_text)
        self.assertIn("metabolic_penalty", walk_text)
        for text in (run_text, stair_text, walk_text):
            self.assertNotIn("/home/", text)
            self.assertNotIn("register_env", text)
            self.assertNotIn("gymnasium.register", text)

    def test_parent_init_registers_residual_envs_without_changing_canonical_count(self):
        init_text = (BENCHMARK_DIR / "__init__.py").read_text(encoding="utf-8")
        canonical = re.findall(
            r'register_msk_bench_task\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"',
            init_text,
        )
        self.assertEqual(len(canonical), 22)
        self.assertIn("RESIDUAL_RL_DIR", init_text)
        self.assertIn("register_residual_msk_bench_task", init_text)
        self.assertIn('"ResidualRun"', init_text)
        self.assertIn('"ResidualStair"', init_text)
        self.assertIn('"ResidualWalk"', init_text)
        self.assertIn("msk_bench.envs.msk.benchmark.residualrl.run:make_env", init_text)
        self.assertIn("msk_bench.envs.msk.benchmark.residualrl.stair:make_env", init_text)
        self.assertIn("msk_bench.envs.msk.benchmark.residualrl.walk:make_env", init_text)

    def test_deprl_configs_exist_for_residual_envs(self):
        expected = {
            "msk_bench_residual_run.yaml": "MSKBenchResidualRun-v0",
            "msk_bench_residual_stair.yaml": "MSKBenchResidualStair-v0",
            "msk_bench_residual_walk.yaml": "MSKBenchResidualWalk-v0",
        }
        for config_name, env_id in expected.items():
            with self.subTest(config=config_name):
                text = (CONFIG_DIR / config_name).read_text(encoding="utf-8")
                self.assertIn(env_id, text)
                self.assertIn("import msk_bench", text)
                self.assertIn("deprl.environments.Gym", text)
                self.assertIn("base_model_dir", text)
                self.assertRegex(text, r"working_dir:\s*\.\/baselines_MSKBench")


if __name__ == "__main__":
    unittest.main()
