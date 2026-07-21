from __future__ import annotations

import math
from pathlib import Path
import unittest


class BenchmarkMetadataAndAnalysisTest(unittest.TestCase):
    def test_registry_exposes_canonical_task_suites(self):
        from msk_bench.registry import TASK_FAMILIES

        self.assertEqual(list(TASK_FAMILIES), ["stabilization", "locomotion", "interaction"])
        self.assertEqual(
            [task.env_id for task in TASK_FAMILIES["stabilization"]],
            [
                "MSKBenchStand-v0",
                "MSKBenchPowerlift-v0",
                "MSKBenchSingleLegStand-v0",
                "MSKBenchSit-v0",
                "MSKBenchBalance-v0",
                "MSKBenchSquat-v0",
            ],
        )
        self.assertEqual(len(TASK_FAMILIES["locomotion"]), 6)
        self.assertEqual(len(TASK_FAMILIES["interaction"]), 10)

    def test_suites_imports_and_matches_registry(self):
        from msk_bench.benchmarking.suites import SUITES, task_ids

        self.assertEqual(
            task_ids("stabilization"),
            [
                "MSKBenchStand-v0",
                "MSKBenchPowerlift-v0",
                "MSKBenchSingleLegStand-v0",
                "MSKBenchSit-v0",
                "MSKBenchBalance-v0",
                "MSKBenchSquat-v0",
            ],
        )
        self.assertEqual(len(SUITES["all"]), 22)

    def test_analysis_helpers_compute_benchmark_metrics(self):
        import numpy as np

        from msk_bench.analysis.emg import pearson_similarity
        from msk_bench.analysis.muscle_energy import activation_energy
        from msk_bench.analysis.recruitment import family_activation_mass
        from msk_bench.analysis.smoothness import finite_difference_jerk, log_mean_squared_jerk

        activations = np.array([[0.0, 0.5], [1.0, 0.5]])
        self.assertAlmostEqual(activation_energy(activations), 0.375)

        joint_velocity = np.array([[0.0], [1.0], [3.0], [6.0]])
        self.assertAlmostEqual(finite_difference_jerk(joint_velocity, dt=1.0), 1.0)
        self.assertAlmostEqual(log_mean_squared_jerk(joint_velocity, dt=1.0), 0.0)

        self.assertAlmostEqual(pearson_similarity([0, 1, 2], [0, 2, 4]), 1.0)
        self.assertTrue(math.isnan(pearson_similarity([1, 1, 1], [0, 1, 2])))

        masses = family_activation_mass(["vaslat_l", "soleus_r", "unknown"], [0.2, 0.3, 0.4])
        self.assertAlmostEqual(masses["quadriceps"], 0.2)
        self.assertAlmostEqual(masses["calves"], 0.3)
        self.assertAlmostEqual(masses["unclassified"], 0.4)

    def test_runner_builds_real_commands_and_requires_assets(self):
        from tempfile import TemporaryDirectory

        from msk_bench.benchmarking.runner import BenchmarkRunRequest, build_command, validate_request_assets

        with TemporaryDirectory() as temp_dir:
            output_json = Path(temp_dir) / "success.json"
            request = BenchmarkRunRequest(
                suite="stabilization",
                algorithm="ppo",
                metric="success",
                env_id="MSKBenchStand-v0",
                output_json=output_json,
            )

            command = build_command(request, repo_root=Path("D:/MSK-Bench"))
            self.assertEqual(command[:2], ["python", "benchmark_eval/evaluate.py"])
            self.assertIn("--algorithm", command)
            self.assertIn("ppo", command)
            self.assertIn("--metric", command)
            self.assertIn("success", command)
            self.assertIn("--env", command)
            self.assertIn("MSKBenchStand-v0", command)
            self.assertIn(str(output_json), command)

            missing = validate_request_assets(request, repo_root=Path(temp_dir))
            self.assertEqual(missing, ["script: benchmark_eval/evaluate.py"])


if __name__ == "__main__":
    unittest.main()
