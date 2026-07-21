from __future__ import annotations

import unittest
from pathlib import Path


class UnifiedRunnerEntrypointTest(unittest.TestCase):
    def test_package_runner_builds_unified_entrypoint_command(self) -> None:
        from msk_bench.benchmarking.runner import BenchmarkRunRequest, build_command

        command = build_command(
            BenchmarkRunRequest(
                suite="all",
                algorithm="ppo",
                metric="energy",
                env_id="MSKBenchWalk-v0",
                episodes=2,
            ),
            repo_root=Path("D:/MSK-Bench"),
            python="python",
        )

        self.assertEqual(command[:2], ["python", "benchmark_eval/evaluate.py"])
        self.assertIn("--algorithm", command)
        self.assertIn("ppo", command)
        self.assertIn("--metric", command)
        self.assertIn("energy", command)


    def test_package_runner_passes_weight_inputs_to_unified_evaluator(self) -> None:
        from msk_bench.benchmarking.runner import BenchmarkRunRequest, build_command

        expected_fields = ("model_dir", "log_path", "checkpoint")
        missing = [name for name in expected_fields if name not in BenchmarkRunRequest.__dataclass_fields__]
        self.assertEqual([], missing)

        command = build_command(
            BenchmarkRunRequest(
                suite="all",
                algorithm="deprl",
                metric="success",
                env_id="MSKBenchWalk-v0",
                model_root=Path("weights/deprl"),
                model_dir=Path("weights/deprl/walk"),
                run_path=Path("weights/deprl/walk_run"),
                log_path=Path("weights/msgym/walk_log"),
                checkpoint="last",
                checkpoint_file=Path("weights/deprl/walk_run/checkpoints/step_5000000.pt"),
            ),
            repo_root=Path("D:/MSK-Bench"),
            python="python",
        )

        for expected in (
            "--model-root",
            "weights\\deprl",
            "--model-dir",
            "weights\\deprl\\walk",
            "--run-path",
            "weights\\deprl\\walk_run",
            "--log-path",
            "weights\\msgym\\walk_log",
            "--checkpoint",
            "last",
            "--checkpoint-file",
            "weights\\deprl\\walk_run\\checkpoints\\step_5000000.pt",
        ):
            self.assertIn(expected, command)
if __name__ == "__main__":
    unittest.main()
