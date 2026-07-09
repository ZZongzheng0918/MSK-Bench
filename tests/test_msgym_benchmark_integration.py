import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MSGYM_ROOT = ROOT / "msgym"
MS_HUMAN_ROOT = ROOT / "MSK-Bench" / "msk_bench" / "simhive" / "ms_human_700"
EXPECTED_MSGYM_ENV_IDS = {
    "msgym/LocomotionFullEnv-v1",
    "msgym/LocomotionLegsEnv-v1",
    "msgym/ManipulationEnv-v1",
}

EXPECTED_BENCHMARK_CONFIGS = [
    ("msk_bench_stand.json", "MSKBenchStand-v0"),
    ("msk_bench_powerlift.json", "MSKBenchPowerlift-v0"),
    ("msk_bench_single_leg_stand.json", "MSKBenchSingleLegStand-v0"),
    ("msk_bench_sit.json", "MSKBenchSit-v0"),
    ("msk_bench_balance.json", "MSKBenchBalance-v0"),
    ("msk_bench_squat.json", "MSKBenchSquat-v0"),
    ("msk_bench_walk.json", "MSKBenchWalk-v0"),
    ("msk_bench_crawl.json", "MSKBenchCrawl-v0"),
    ("msk_bench_run.json", "MSKBenchRun-v0"),
    ("msk_bench_jump.json", "MSKBenchJump-v0"),
    ("msk_bench_walk_turn.json", "MSKBenchWalkTurn-v0"),
    ("msk_bench_sidestep.json", "MSKBenchSidestep-v0"),
    ("msk_bench_stairs.json", "MSKBenchStairs-v0"),
    ("msk_bench_hurdle.json", "MSKBenchHurdle-v0"),
    ("msk_bench_step_stones.json", "MSKBenchStepStones-v0"),
    ("msk_bench_slide.json", "MSKBenchSlide-v0"),
    ("msk_bench_door_open.json", "MSKBenchDoorOpen-v0"),
    ("msk_bench_reach.json", "MSKBenchReach-v0"),
    ("msk_bench_walk_and_sit.json", "MSKBenchWalkAndSit-v0"),
    ("msk_bench_chin_up.json", "MSKBenchChinUp-v0"),
    ("msk_bench_catch.json", "MSKBenchCatch-v0"),
    ("msk_bench_pole_walk.json", "MSKBenchPoleWalk-v0"),
]


class MsgymBenchmarkIntegrationTests(unittest.TestCase):
    def test_msgym_algorithm_directory_is_sibling_of_existing_algorithms(self):
        for dirname in ["depRL", "ppo", "sac", "msgym"]:
            with self.subTest(dirname=dirname):
                self.assertTrue((ROOT / dirname).is_dir(), dirname)
        for relpath in [
            "SB3-Scripts/train.py",
            "SB3-Scripts/eval.py",
            "DynSyn/SAC_DynSyn.py",
            "configs/msk_bench_walk.json",
            "configs/msk_bench_stand.json",
            "configs/msk_bench_pole_walk.json",
            "msgym/__init__.py",
            "msgym/envs/utils.py",
            "README.md",
        ]:
            with self.subTest(relpath=relpath):
                self.assertTrue((MSGYM_ROOT / relpath).is_file(), relpath)

    def test_ms_human_700_assets_live_inside_msk_bench_package(self):
        for relpath in [
            "MS-Human-700.xml",
            "MS-Human-700-Locomotion.xml",
            "MS-Human-700-Manipulation.xml",
            "Asset/Asset_Lowerbody.xml",
            "Geometry/r_pelvis.stl",
            "Muscle/Muscle_Leg_r.xml",
            "Tendon/Tendon_Leg_r.xml",
            "LICENSE",
        ]:
            with self.subTest(relpath=relpath):
                self.assertTrue((MS_HUMAN_ROOT / relpath).is_file(), relpath)

    def test_msgym_registers_three_environment_ids(self):
        init_text = (MSGYM_ROOT / "msgym" / "__init__.py").read_text(encoding="utf-8")
        for env_id in EXPECTED_MSGYM_ENV_IDS:
            with self.subTest(env_id=env_id):
                self.assertIn(f'id="{env_id}"', init_text)

    def test_model_path_helper_prefers_msk_bench_ms_human_assets(self):
        utils_text = (MSGYM_ROOT / "msgym" / "envs" / "utils.py").read_text(encoding="utf-8")
        self.assertIn("ms_human_700", utils_text)
        self.assertIn("import msk_bench", utils_text)
        self.assertIn("resources.files(\"msk_bench\")", utils_text)
        self.assertIn("resources.files(\"msgym\")", utils_text)

    def test_configs_are_project_relative_and_cover_all_msk_bench_environments(self):
        config_dir = MSGYM_ROOT / "configs"
        expected = dict(EXPECTED_BENCHMARK_CONFIGS)
        config_paths = sorted(config_dir.glob("*.json"))
        self.assertEqual([path.name for path in config_paths], sorted(expected))

        for config in config_paths:
            data = json.loads(config.read_text(encoding="utf-8"))
            text = config.read_text(encoding="utf-8")
            with self.subTest(config=config.name):
                self.assertEqual(data.get("env_name"), expected[config.name])
                self.assertTrue(data["env_name"].startswith("MSKBench"))
                self.assertNotIn("msgym/", data["env_name"])
                self.assertNotIn("myo", data["env_name"])
                self.assertEqual(data.get("single_env_kwargs"), {})
                self.assertNotIn("/DATA_EDS/", text)
                self.assertNotIn("/home/", text)
                self.assertNotIn("D:\\", text)
                self.assertTrue(str(data.get("log_root_dir", "")).startswith("./runs"))

    def test_training_utils_register_msk_bench_envs_in_subprocesses(self):
        utils_text = (MSGYM_ROOT / "SB3-Scripts" / "utils.py").read_text(encoding="utf-8")
        self.assertIn('startswith("MSKBench")', utils_text)
        self.assertIn("import msk_bench", utils_text)

    def test_train_cli_lists_configs_without_heavy_training_imports(self):
        result = subprocess.run(
            [sys.executable, str(MSGYM_ROOT / "SB3-Scripts" / "train.py"), "--list-configs"],
            cwd=MSGYM_ROOT,
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        listed = set(result.stdout.splitlines())
        self.assertEqual(listed, {name for name, _ in EXPECTED_BENCHMARK_CONFIGS})
        self.assertIn("msk_bench_walk.json", listed)
        self.assertIn("msk_bench_stand.json", listed)
        self.assertIn("msk_bench_pole_walk.json", listed)
        self.assertNotIn("locomotionFull.json", listed)


if __name__ == "__main__":
    unittest.main()