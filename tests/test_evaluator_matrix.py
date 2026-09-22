"""All public metric/render entries, using genuinely trained model artifacts."""

import json

import pytest

import test_deprl_training_runtime as deprl_tests
import test_msgym_training_runtime as msgym_tests
import test_training_runtime as sb3_tests
from msk_bench.validation import validate_table, validate_video


@pytest.fixture(scope="session", params=["ppo", "sac", "deprl", "msgym", "middleware"])
def trained_artifact(request, tmp_path_factory):
    algorithm = request.param
    directory = tmp_path_factory.mktemp("trained_" + algorithm)
    if algorithm in ("ppo", "sac"):
        extra = (["--n-steps", "4", "--batch-size", "4", "--n-epochs", "1"] if algorithm == "ppo" else
                 ["--learning-starts", "1", "--buffer-size", "32", "--batch-size", "2",
                  "--train-freq", "1", "--gradient-steps", "1"])
        sb3_tests.test_sb3_short_training_saves_reloadable_artifacts(
            algorithm, f"rl_paradigms.{algorithm}.train_{algorithm}_msk_bench", extra, directory,
        )
    elif algorithm == "msgym":
        msgym_tests.test_msgym_short_training_and_reload(directory)
    else:
        deprl_tests.test_short_train_updates_saves_and_reloads(directory, algorithm, 1)
    manifests = list(directory.rglob("training_manifest.json"))
    assert len(manifests) == 1
    return algorithm, manifests[0]


@pytest.mark.parametrize("metric", ["success", "robustness", "smooth", "energy", "emg", "render"])
def test_every_evaluator_and_renderer(trained_artifact, metric, tmp_path):
    algorithm, manifest = trained_artifact
    env_id = json.loads(manifest.read_text())["env_id"].replace("-Middleware-", "-")
    extra = []
    if metric == "robustness":
        extra = ["--", "--noise-type", "all", "--action-scales", "0,0.02",
                 "--obs-scales", "0,0.01", "--dynamics-scales", "0,0.05"]
    if metric == "render":
        extra = ["--", "--width", "320", "--height", "240"]
    deprl_tests.run_isolated([
        "-m", "benchmark_eval.evaluate", "--algorithm", algorithm, "--metric", metric,
        "--env", env_id, "--episodes", "1", "--max-steps", "4",
        "--artifact-manifest", str(manifest), "--output-dir", str(tmp_path), "--execute", *extra,
    ], tmp_path)
    if metric == "render":
        videos = list(tmp_path.rglob("*.mp4"))
        assert videos
        for path in videos:
            assert validate_video(path) == (240, 320, 3)
    elif metric == "emg":
        tables = list(tmp_path.rglob("*.csv"))
        assert tables
        assert all(validate_table(path) > 0 for path in tables)
    else:
        for suffix in ("json", "csv"):
            assert validate_table(tmp_path / f"{metric}_{algorithm}.{suffix}") > 0


def test_manifest_drives_exact_weight_paths(tmp_path):
    from benchmark_eval.evaluate import build_parser, request_from_args, build_command

    model, norm = tmp_path / "final_model.zip", tmp_path / "vec_normalize.pkl"
    model.touch()
    norm.touch()
    manifest = tmp_path / "training_manifest.json"
    manifest.write_text(json.dumps({
        "algorithm": "ppo", "env_id": "MSKBenchSquat-v0", "seed": 17, "timesteps": 8,
        "model_path": str(model), "normalization_path": str(norm),
    }))
    args = build_parser().parse_args(["--artifact-manifest", str(manifest)])
    request = request_from_args(args)
    assert request.algorithms == ("ppo",)
    assert request.env_id == "MSKBenchSquat-v0"
    command = build_command(request, "ppo")
    assert command[command.index("--model-path") + 1] == str(model)
    assert command[command.index("--norm-path") + 1] == str(norm)
