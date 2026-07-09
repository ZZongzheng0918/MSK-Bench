import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_ENVS = [
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

ALGORITHMS = [
    ("ppo", "train_ppo_msk_bench.py", "PPO", "MSKBenchWalk-v0"),
    ("sac", "train_sac_msk_bench.py", "SAC", "MSKBenchStand-v0"),
]


def literal_assignment(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"Missing assignment: {name}")


class SB3BenchmarkAlgorithmTests(unittest.TestCase):
    def test_algorithm_directories_are_siblings_of_deprl(self):
        self.assertTrue((ROOT / "depRL").is_dir())
        for dirname, script_name, _, _ in ALGORITHMS:
            with self.subTest(dirname=dirname):
                self.assertTrue((ROOT / dirname).is_dir())
                self.assertTrue((ROOT / dirname / script_name).is_file())

    def test_scripts_declare_the_22_msk_bench_environments(self):
        for dirname, script_name, algorithm_name, default_env in ALGORITHMS:
            script = ROOT / dirname / script_name
            tree = ast.parse(script.read_text(encoding="utf-8"))
            envs = list(literal_assignment(tree, "MSK_BENCH_ENVS"))
            with self.subTest(script=script_name):
                self.assertEqual(envs, EXPECTED_ENVS)
                self.assertEqual(literal_assignment(tree, "ALGORITHM_NAME"), algorithm_name)
                self.assertEqual(literal_assignment(tree, "DEFAULT_ENV_ID"), default_env)

    def test_scripts_are_msk_bench_only_and_path_portable(self):
        forbidden = ["import myosuite", "from myosuite", "/home/", "MyoCustomWalk", "myo_pole_walk.xml"]
        for dirname, script_name, _, _ in ALGORITHMS:
            script = ROOT / dirname / script_name
            text = script.read_text(encoding="utf-8")
            with self.subTest(script=script_name):
                self.assertIn("import msk_bench", text)
                self.assertIn("--env", text)
                self.assertIn("--dry-run", text)
                for item in forbidden:
                    self.assertNotIn(item, text)

    def test_list_envs_cli_prints_all_environments_without_training_dependencies(self):
        for dirname, script_name, _, _ in ALGORITHMS:
            script = ROOT / dirname / script_name
            result = subprocess.run(
                [sys.executable, str(script), "--list-envs"],
                cwd=script.parent,
                text=True,
                capture_output=True,
                timeout=15,
            )
            with self.subTest(script=script_name):
                self.assertEqual(result.returncode, 0, result.stderr)
                for env_id in EXPECTED_ENVS:
                    self.assertIn(env_id, result.stdout)


if __name__ == "__main__":
    unittest.main()