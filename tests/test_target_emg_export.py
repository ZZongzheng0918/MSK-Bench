import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON_PATH = ROOT / "emg_export_common.py"


def load_common():
    spec = importlib.util.spec_from_file_location("emg_export_common", COMMON_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeModel:
    actuator_names = ["other", "soleus_r", "gasmed_r", "tibant_r"]


class FakeSim:
    model = FakeModel()


class FakeEnv:
    sim = FakeSim()


class TargetEmgExportTests(unittest.TestCase):
    def test_target_muscles_match_draw_script_subset(self):
        common = load_common()
        self.assertEqual(
            common.TARGET_EMG_MUSCLES,
            (
                "soleus_r",
                "soleus_l",
                "gasmed_r",
                "gaslat_r",
                "tibant_r",
                "perlong_r",
                "perbrev_r",
                "recfem_r",
                "vaslat_r",
                "vasmed_r",
                "bflh_r",
                "semiten_r",
            ),
        )

    def test_target_emg_row_exports_only_target_muscle_columns(self):
        common = load_common()
        row = common.target_emg_row(
            algorithm="PPO",
            env_id="MSKBenchWalk-v0",
            episode=1,
            step=2,
            time_s=0.04,
            reward=1.5,
            done=False,
            env=FakeEnv(),
            activation=[0.1, 0.2, 0.3, 0.4],
        )

        self.assertEqual(row["algorithm"], "PPO")
        self.assertEqual(row["env_id"], "MSKBenchWalk-v0")
        self.assertEqual(row["step"], 2)
        self.assertEqual(row["soleus_r"], 0.2)
        self.assertEqual(row["gasmed_r"], 0.3)
        self.assertEqual(row["tibant_r"], 0.4)
        self.assertIn("semiten_r", row)
        self.assertNotIn("other", row)

        muscle_columns = [key for key in row if key in common.TARGET_EMG_MUSCLES]
        self.assertEqual(muscle_columns, list(common.TARGET_EMG_MUSCLES))

    def test_write_emg_csv_preserves_target_column_order(self):
        common = load_common()
        row = common.target_emg_row(
            algorithm="SAC",
            env_id="MSKBenchWalk-v0",
            episode=1,
            step=0,
            time_s=0.0,
            reward=0.0,
            done=False,
            env=FakeEnv(),
            activation=[0.1, 0.2, 0.3, 0.4],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "emg.csv"
            common.write_emg_csv([row], path)
            with path.open(newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle))
        self.assertEqual(header[:6], ["algorithm", "env_id", "episode", "step", "time_s", "reward"])
        self.assertEqual(header[7:], list(common.TARGET_EMG_MUSCLES))

    def test_algorithm_emg_export_wrappers_exist(self):
        expected = [
            ROOT / "ppo" / "export_ppo_emg.py",
            ROOT / "sac" / "export_sac_emg.py",
            ROOT / "depRL" / "export_deprl_emg.py",
            ROOT / "deprl_middleware_22tasks" / "export_middleware_emg.py",
            ROOT / "msgym" / "export_msgym_emg.py",
            ROOT / "draw_emg_12_muscles.py",
        ]
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), path)

    def test_draw_cli_has_portable_help(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "draw_emg_12_muscles.py"), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--series", result.stdout)
        self.assertIn("--human-mat", result.stdout)


if __name__ == "__main__":
    unittest.main()
