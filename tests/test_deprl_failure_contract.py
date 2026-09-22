"""Do not report failed depRL runs or failed policy loads as successes."""

from pathlib import Path

import pytest
import torch
import yaml

from test_deprl_training_runtime import CONFIG, run_isolated


def test_invalid_policy_load_raises(tmp_path):
    from deprl.custom_mpo_torch import TunedMPO

    agent = TunedMPO()
    agent.model = torch.nn.Linear(2, 1)
    with pytest.raises(FileNotFoundError):
        agent.load(str(tmp_path / "missing"), only_checkpoint=True)


def test_worker_error_reaches_parent_and_closes(tmp_path):
    run_isolated(["-c", f"""
import multiprocessing
import yaml
from deprl.custom_distributed import distribute
with open({str(CONFIG)!r}) as stream:
    config = yaml.safe_load(stream)
tonic = config['tonic']
tonic['header'] += '\\nimport multiprocessing\\nif multiprocessing.current_process().name != "MainProcess": raise RuntimeError("worker startup failed")'
env = distribute(tonic['environment'], tonic, dict(), parallel=2, sequential=1)
try:
    env.initialize(0)
    try:
        env.start()
    except RuntimeError as error:
        assert 'worker' in str(error)
    else:
        raise AssertionError('Worker failure was ignored')
finally:
    env.close()
    env.close()
assert not multiprocessing.active_children()
"""], tmp_path)


def test_training_exception_is_not_swallowed(tmp_path):
    config = yaml.safe_load(CONFIG.read_text())
    config["tonic"]["before_training"] = "raise RuntimeError('intentional training failure')"
    config_path = tmp_path / "fail.yaml"
    config_path.write_text(yaml.safe_dump(config))
    run_isolated(["-c", f"""
from deprl.main import parse_config, train
config = parse_config([{str(config_path)!r}, '--parallel', '1', '--sequential', '1',
                       '--hidden-size', '32', '--output-dir', {str(tmp_path)!r}])
try:
    train(config)
except RuntimeError as error:
    assert 'intentional training failure' in str(error)
else:
    raise AssertionError('Training error was swallowed')
"""], tmp_path)
    assert not list(tmp_path.rglob("training_manifest.json"))


def test_requested_one_episode_is_evaluated(tmp_path):
    config = yaml.safe_load(CONFIG.read_text())
    config["tonic"]["environment"] = "deprl.environments.Gym('MSKBenchSquat-v0', scaled_actions=False, max_episode_steps=2)"
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(yaml.safe_dump(config))
    run_isolated([
        "-m", "deprl.main", str(config_path), "--steps", "4", "--epoch-steps", "4",
        "--save-steps", "4", "--parallel", "1", "--sequential", "1",
        "--test-episodes", "1", "--hidden-size", "32", "--batch-size", "2",
        "--buffer-size", "32", "--steps-before-batches", "1", "--steps-between-batches", "1",
        "--batch-iterations", "1", "--cpu", "--output-dir", str(tmp_path / "runs"),
    ], tmp_path)
    import pandas as pd

    log = pd.read_csv(next((tmp_path / "runs").rglob("log.csv")))
    assert log["test/episode_length/mean"].iloc[-1] == 2
    assert log["test/episode_length/size"].iloc[-1] == 1
    assert len(list((tmp_path / "runs").rglob("training_manifest.json"))) == 1


def test_deprl_extras_declare_direct_dependencies():
    import tomllib

    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    for extra in ("deprl", "middleware", "all"):
        declared = project["project"]["optional-dependencies"][extra]
        assert any(item.startswith("gdown") for item in declared)
        assert any(item.startswith("pandas") for item in declared)
