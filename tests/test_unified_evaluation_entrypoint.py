from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


def path_text(value: str) -> str:
    return str(Path(value))


class UnifiedEvaluationEntrypointTest(unittest.TestCase):
    def test_builds_one_metric_across_multiple_algorithms(self) -> None:
        from benchmark_eval.evaluate import EvaluationRequest, build_commands

        benchmark_root = Path("D:/MSK-Bench")
        request = EvaluationRequest(
            metric="success",
            algorithms=("ppo", "sac", "deprl"),
            env_id="MSKBenchWalk-v0",
            episodes=3,
            benchmark_root=benchmark_root,
            output_json=Path("results/success.json"),
        )

        commands = build_commands(request, python="python")

        self.assertEqual([command[1] for command in commands], [
            "ppo/eval_ppo_success.py",
            "sac/eval_sac_success.py",
            "depRL/eval_deprl_success.py",
        ])
        for algorithm, command in zip(("ppo", "sac", "deprl"), commands):
            self.assertIn("--env", command)
            self.assertIn("MSKBenchWalk-v0", command)
            self.assertIn("--episodes", command)
            self.assertIn("3", command)
            self.assertIn("--benchmark-root", command)
            self.assertIn(str(benchmark_root), command)
            self.assertTrue(any(f"success_{algorithm}.json" in item for item in command), command)

    def test_weight_inputs_are_first_class_evaluation_request_fields(self) -> None:
        from benchmark_eval.evaluate import EvaluationRequest, build_command

        expected_fields = (
            "model_root",
            "model_dir",
            "model_path",
            "norm_path",
            "run_path",
            "log_path",
            "checkpoint",
            "checkpoint_file",
        )
        missing = [name for name in expected_fields if name not in EvaluationRequest.__dataclass_fields__]
        self.assertEqual([], missing)

        ppo_request = EvaluationRequest(
            metric="success",
            algorithms=("ppo",),
            env_id="MSKBenchWalk-v0",
            model_root=Path("weights/ppo"),
            model_dir=Path("weights/ppo/walk"),
            model_path=Path("weights/ppo/walk/checkpoints/best_model.zip"),
            norm_path=Path("weights/ppo/walk/checkpoints/vec_normalize.pkl"),
            checkpoint="last",
        )
        ppo_command = build_command(ppo_request, "ppo")
        self.assertIn("--model-root", ppo_command)
        self.assertIn(path_text("weights/ppo"), ppo_command)
        self.assertIn("--model-dir", ppo_command)
        self.assertIn(path_text("weights/ppo/walk"), ppo_command)
        self.assertIn("--model-path", ppo_command)
        self.assertIn("--norm-path", ppo_command)
        self.assertNotIn("--checkpoint", ppo_command)

        deprl_request = EvaluationRequest(
            metric="success",
            algorithms=("deprl",),
            env_id="MSKBenchWalk-v0",
            run_path=Path("weights/deprl/walk_run"),
            checkpoint="5000000",
        )
        deprl_command = build_command(deprl_request, "deprl")
        self.assertIn("--run-path", deprl_command)
        self.assertIn(path_text("weights/deprl/walk_run"), deprl_command)
        self.assertIn("--checkpoint", deprl_command)
        self.assertIn("5000000", deprl_command)

    def test_cli_parser_accepts_explicit_weight_inputs(self) -> None:
        from benchmark_eval.evaluate import build_parser, request_from_args

        parser = build_parser()
        option_dests = {action.dest for action in parser._actions}
        missing = [name for name in ("model_dir", "checkpoint") if name not in option_dests]
        self.assertEqual([], missing)

        args = parser.parse_args([
            "--metric",
            "success",
            "--algorithm",
            "deprl",
            "--env",
            "MSKBenchWalk-v0",
            "--model-root",
            "weights/deprl",
            "--model-dir",
            "weights/deprl/walk",
            "--model-path",
            "weights/ppo/walk/checkpoints/best_model.zip",
            "--norm-path",
            "weights/ppo/walk/checkpoints/vec_normalize.pkl",
            "--run-path",
            "weights/deprl/walk_run",
            "--log-path",
            "weights/msgym/walk_log",
            "--checkpoint",
            "last",
            "--checkpoint-file",
            "weights/deprl/walk_run/checkpoints/step_5000000.pt",
        ])
        request = request_from_args(args)
        self.assertEqual(Path("weights/deprl"), request.model_root)
        self.assertEqual(Path("weights/deprl/walk"), request.model_dir)
        self.assertEqual(Path("weights/ppo/walk/checkpoints/best_model.zip"), request.model_path)
        self.assertEqual(Path("weights/ppo/walk/checkpoints/vec_normalize.pkl"), request.norm_path)
        self.assertEqual(Path("weights/deprl/walk_run"), request.run_path)
        self.assertEqual(Path("weights/msgym/walk_log"), request.log_path)
        self.assertEqual("last", request.checkpoint)
        self.assertEqual(Path("weights/deprl/walk_run/checkpoints/step_5000000.pt"), request.checkpoint_file)

    def test_msgym_weight_inputs_use_log_path_and_optional_model_files(self) -> None:
        from benchmark_eval.evaluate import EvaluationRequest, build_command

        request = EvaluationRequest(
            metric="success",
            algorithms=("msgym",),
            env_id="MSKBenchWalk-v0",
            log_path=Path("weights/msgym/MSKBenchWalk-v0/0721-120000_0"),
            model_path=Path("weights/msgym/MSKBenchWalk-v0/0721-120000_0/checkpoint/best_model.zip"),
            norm_path=Path("weights/msgym/MSKBenchWalk-v0/0721-120000_0/checkpoint/best_env.zip"),
        )

        command = build_command(request, "msgym")

        self.assertIn("msgym/eval_msgym_success.py", command)
        self.assertIn("--log-path", command)
        self.assertIn(path_text("weights/msgym/MSKBenchWalk-v0/0721-120000_0"), command)
        self.assertIn("--model-path", command)
        self.assertIn(path_text("weights/msgym/MSKBenchWalk-v0/0721-120000_0/checkpoint/best_model.zip"), command)
        self.assertIn("--norm-path", command)
        self.assertIn(path_text("weights/msgym/MSKBenchWalk-v0/0721-120000_0/checkpoint/best_env.zip"), command)

    def test_readme_documents_training_entrypoints_for_each_baseline(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        text = (repo_root / "README.md").read_text(encoding="utf-8")

        expected_phrases = (
            "### Training Overview",
            "### PPO Training",
            "python ppo\\train_ppo_msk_bench.py",
            "### SAC Training",
            "python sac\\train_sac_msk_bench.py",
            "### depRL Training",
            "python -m deprl.main",
            "depRL\\baselines_MSKBench",
            "### DynSyn/msgym Training",
            "python msgym\\SB3-Scripts\\train.py -f configs\\msk_bench_walk.json",
            "msgym\\runs\\msgym_logs",
            "best_model.zip",
            "best_env.zip",
            "### Latent-Action Middleware Training",
            "python deprl_middleware_22tasks\\generate_configs.py",
            "baselines_MSKBench_Middleware",
        )
        for phrase in expected_phrases:
            self.assertIn(phrase, text)

    def test_metric_aliases_route_to_existing_legacy_scripts(self) -> None:
        from benchmark_eval.evaluate import EvaluationRequest, build_commands

        request = EvaluationRequest(
            metric="emg_similarity",
            algorithms=("ppo", "sac", "dynsyn", "latent-action"),
            env_id="MSKBenchWalk-v0",
        )

        commands = build_commands(request)

        self.assertEqual([command[1] for command in commands], [
            "ppo/export_ppo_emg.py",
            "sac/export_sac_emg.py",
            "msgym/export_msgym_emg.py",
            "deprl_middleware_22tasks/export_middleware_emg.py",
        ])

    def test_cli_dry_run_prints_commands_without_running_simulation(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        completed = subprocess.run(
            [
                sys.executable,
                str(repo_root / "benchmark_eval" / "evaluate.py"),
                "--metric",
                "success",
                "--algorithms",
                "ppo,sac",
                "--env",
                "MSKBenchWalk-v0",
                "--episodes",
                "1",
            ],
            cwd=repo_root,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        self.assertIn("ppo/eval_ppo_success.py", completed.stdout)
        self.assertIn("sac/eval_sac_success.py", completed.stdout)
        self.assertIn("--metric success", completed.stdout)
        self.assertEqual("", completed.stderr)


if __name__ == "__main__":
    unittest.main()