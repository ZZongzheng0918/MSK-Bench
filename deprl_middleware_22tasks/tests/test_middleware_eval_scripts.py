import ast
import re
import subprocess
import sys
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
EXPECTED_SCRIPTS = [
    "eval_middleware_success.py",
    "eval_middleware_robustness.py",
    "eval_middleware_smooth.py",
    "eval_middleware_energy.py",
    "render_middleware_all.py",
]


def env_slug(env_id):
    task = env_id.removeprefix("MSKBench").removesuffix("-v0")
    return re.sub(r"(?<!^)([A-Z])", r"_\1", task).lower()


class MiddlewareEvalScriptTests(unittest.TestCase):
    def test_eval_and_batch_render_scripts_exist_and_parse(self):
        for script_name in EXPECTED_SCRIPTS:
            with self.subTest(script=script_name):
                script = ROOT / script_name
                self.assertTrue(script.exists(), script_name)
                ast.parse(script.read_text(encoding="utf-8"), filename=str(script))

    def test_success_script_lists_all_canonical_envs(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "eval_middleware_success.py"), "--list-envs"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip().splitlines(), EXPECTED_TASKS)

    def test_middleware_configs_cover_all_tasks(self):
        config_dir = ROOT / "configs"
        actual_names = {config.name for config in config_dir.glob("*.yaml")}
        expected_names = {
            f"msk_bench_{env_slug(env_id)}_middleware.yaml"
            for env_id in EXPECTED_TASKS
        }
        self.assertEqual(actual_names, expected_names)
        for env_id in EXPECTED_TASKS:
            with self.subTest(env=env_id):
                text = (config_dir / f"msk_bench_{env_slug(env_id)}_middleware.yaml").read_text(encoding="utf-8")
                self.assertIn(env_id.replace("-v0", "-Middleware-v0"), text)
                self.assertIn("baselines_MSKBench_Middleware", text)
                self.assertIn("deprl_middleware_22tasks.registry", text)

    def test_derived_metric_scripts_delegate_to_common_helpers(self):
        expected_calls = {
            "eval_middleware_smooth.py": "common.main_deprl_smooth",
            "eval_middleware_energy.py": "common.main_deprl_energy",
            "render_middleware_all.py": "common.main_deprl_render",
        }
        for script_name, helper_call in expected_calls.items():
            with self.subTest(script=script_name):
                text = (ROOT / script_name).read_text(encoding="utf-8")
                self.assertIn("import eval_middleware_success as base", text)
                self.assertIn(helper_call, text)


if __name__ == "__main__":
    unittest.main()
