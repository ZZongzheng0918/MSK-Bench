import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_reward_projection_preserves_l1_and_relative_bound():
    from rl_paradigms.agentic_walk.constraints import constrain_weights
    old = dict(a=10.0, b=-30.0, inactive=0.0)
    new = constrain_weights(old, dict(a=1e6, b=0, inactive=100), l1_norm=40)
    assert sum(abs(v) for v in new.values()) == pytest.approx(40)
    assert new["inactive"] == 0
    for key, value in old.items():
        assert abs(new[key] - value) <= abs(value) * .2 + 1e-9
    with pytest.raises(ValueError):
        constrain_weights(old, dict(a=float("nan")), l1_norm=40)


def test_agentic_controller_initializes_matched_norm_and_requires_credentials(tmp_path, monkeypatch):
    from rl_paradigms.agentic_walk.agentic_walk_v0 import AgenticRewardController
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MSK_BENCH_AGENTIC_API_KEY", raising=False)
    controller = AgenticRewardController(tmp_path / "weights.json", tmp_path / "best.json",
                                         dict(a=1, b=-3), 1, l1_norm=40)
    weights = controller.read_weights(dict(a=1, b=-3))
    assert sum(abs(v) for v in weights.values()) == pytest.approx(40)
    controller.record(.2, .1)
    with pytest.raises(RuntimeError, match="API key"):
        controller.maybe_update(1, weights)


def test_agentic_rejects_nonfinite_saved_weights(tmp_path):
    from rl_paradigms.agentic_walk.agentic_walk_v0 import AgenticRewardController
    path = tmp_path / "weights.json"
    path.write_text('{"a": NaN}')
    controller = AgenticRewardController(path, tmp_path / "best.json", dict(a=1), 1, enabled=False)
    with pytest.raises(ValueError):
        controller.read_weights(dict(a=1))


def test_agentic_does_not_silently_accept_corrupt_state(tmp_path):
    from rl_paradigms.agentic_walk.agentic_walk_v0 import AgenticRewardController
    path = tmp_path / "weights.json"
    controller = AgenticRewardController(path, tmp_path / "best.json", dict(a=1), 1, l1_norm=40)
    path.write_text("{incomplete")
    with pytest.raises(ValueError, match="state"):
        controller.read_weights(dict(a=1))


def test_agentic_respects_existing_lock(tmp_path, monkeypatch):
    import os
    from rl_paradigms.agentic_walk.agentic_walk_v0 import AgenticRewardController
    path = tmp_path / "weights.json"
    controller = AgenticRewardController(path, tmp_path / "best.json", dict(a=1), 1)
    lock = path.with_suffix(".json.lock")
    lock.write_text("another writer")
    os.utime(lock, (1, 1))
    def forbidden(*args):
        pytest.fail("Existing lock must prevent an API call even after a long request")
    monkeypatch.setattr(controller, "_update_from_llm", forbidden)
    controller.record(.1, .2)
    assert controller.maybe_update(1, dict(a=1)) == dict(a=1)
    assert lock.read_text() == "another writer"


def test_agentic_mocked_response_is_constrained_and_recorded(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from rl_paradigms.agentic_walk.agentic_walk_v0 import AgenticRewardController
    closed = []
    response = SimpleNamespace(model="test-model", choices=[
        SimpleNamespace(message=SimpleNamespace(content='{"a": 100, "b": -1}'))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)),
                             close=lambda: closed.append(True))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: client))
    monkeypatch.setenv("MSK_BENCH_AGENTIC_API_KEY", "test-not-a-secret")
    monkeypatch.setenv("MSK_BENCH_AGENTIC_MODEL", "test-model")
    path = tmp_path / "weights.json"
    controller = AgenticRewardController(path, tmp_path / "best.json", dict(a=1, b=-3), 1, l1_norm=40)
    old = controller.read_weights(dict(a=1, b=-3))
    controller.record(.4, .1)
    updated = controller.maybe_update(1, old)
    assert sum(abs(v) for v in updated.values()) == pytest.approx(40)
    assert abs(updated["a"] - old["a"]) <= 2 + 1e-9
    log = path.with_suffix(".updates.jsonl").read_text()
    assert json.loads(log)["applied"] == updated
    assert "test-not-a-secret" not in log
    assert closed == [True]


def test_analysis_cli_outputs_auditable_files(tmp_path):
    from msk_bench.analysis.robustness import PAPER_GRIDS
    rows = [dict(env_id="A", algorithm="ppo", noise_type=noise, noise_scale=scale, success_rate=100)
            for noise, scales in PAPER_GRIDS.items() for scale in scales]
    inputs = tmp_path / "robustness.json"
    output = tmp_path / "summary.json"
    inputs.write_text(json.dumps(rows))
    completed = subprocess.run([sys.executable, "-B", "-m", "benchmark_eval.analyze", "robustness",
                                "--input", str(inputs), "--output", str(output), "--success-unit", "percent"],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    data = json.loads(output.read_text())
    assert data["result"]["average"] == 100
    assert len(data["input_sha256"][str(inputs)]) == 64
    failed = subprocess.run([sys.executable, "-B", "-m", "benchmark_eval.analyze", "robustness",
                             "--input", str(inputs), "--output", str(inputs), "--success-unit", "percent"],
                            capture_output=True, text=True)
    assert failed.returncode != 0
    assert json.loads(inputs.read_text()) == rows


def test_latent_config_dimension_and_strict_weights():
    from rl_paradigms.deprl_middleware_22tasks.generate_configs import TASKS, config_text
    for dim in (16, 32, 64, 128):
        config = config_text(TASKS[0], "encoder.pt", "decoder.pt", latent_dim=dim, strict_weights=True)
        assert f"latent_dim={dim}" in config
        assert "strict_weights=True" in config
    with pytest.raises(ValueError):
        config_text(TASKS[0], None, None, strict_weights=True)


def test_recruitment_sums_all_frames_without_truncation():
    from msk_bench.analysis.recruitment import family_activation_mass
    result = family_activation_mass(["vaslat_r", "soleus_r"], [[1, 2], [3, 4]])
    assert result["quadriceps"] == 4
    assert result["calves"] == 6
    with pytest.raises(ValueError):
        family_activation_mass(["soleus_r"], [[1, 2]])


def test_emg_csv_cycle_comparison(tmp_path):
    from benchmark_eval.analyze import compare_emg_csv
    phase = np.linspace(0, 2 * np.pi, 101)
    values = np.sin(phase)
    simulated = tmp_path / "sim.csv"
    reference = tmp_path / "ref.csv"
    simulated.write_text("episode,cycle,soleus_r\n" + "\n".join(
        f"1,{cycle},{value}" for cycle in (0, 1) for value in values))
    reference.write_text("human_soleus\n" + "\n".join(map(str, 2 * values + 3)))
    result = compare_emg_csv(simulated, reference, {"soleus_r": "human_soleus"})
    assert result["mean_pearson"] == pytest.approx(1)
    assert result["muscles"]["soleus_r"]["cycle_count"] == 2
    from benchmark_eval.analyze import main
    mapping_path, output = tmp_path / "columns.json", tmp_path / "emg.json"
    mapping_path.write_text(json.dumps({"soleus_r": "human_soleus"}))
    assert main(["emg", "--input", str(simulated), "--reference", str(reference),
                 "--mapping", str(mapping_path), "--source", "Synthetic test fixture, not human data",
                 "--output", str(output)]) == 0
    assert json.loads(output.read_text())["result"]["mean_pearson"] == pytest.approx(1)
    with pytest.raises(ValueError):
        compare_emg_csv(simulated, reference, {"missing": "human_soleus"})


@pytest.mark.parametrize("metric", ["peak", "recruitment", "reconstruction"])
def test_remaining_analysis_commands_write_valid_json(tmp_path, metric):
    from benchmark_eval.analyze import main
    output = tmp_path / "result.json"
    extra = []
    if metric == "peak":
        inputs = tmp_path / "evaluations.json"
        inputs.write_text(json.dumps([dict(algorithm="ppo", env_id="A", step=step, mean_return=reward)
                                      for step, reward in ((100, 2), (200, 1))]))
    elif metric == "recruitment":
        inputs, mapping = tmp_path / "activations.npz", tmp_path / "mapping.json"
        np.savez(inputs, activations=np.array([[1., 2.], [3., 4.]]),
                 actuator_names=np.array(["m1", "m2"]))
        mapping.write_text(json.dumps(dict(m1="quadriceps", m2="calves")))
        extra = ["--mapping", str(mapping)]
    else:
        inputs = tmp_path / "reconstruction.npz"
        target = np.array([[0., 1.], [1., 0.], [2., 2.]])
        np.savez(inputs, target=target, reconstruction=target)
    assert main([metric, "--input", str(inputs), "--output", str(output), *extra]) == 0
    result = json.loads(output.read_text())["result"]
    if metric == "peak":
        assert result[0]["peak_efficiency_steps"] == 100
    elif metric == "recruitment":
        assert result["family_share"] == dict(quadriceps=.4, calves=.6)
    else:
        assert result["mse"] == 0
        assert result["explained_variance"] == 1


@pytest.mark.parametrize("mass", [.05, .5, 5, 10])
def test_powerlift_load_is_applied_to_physics(mass):
    import gymnasium as gym
    import msk_bench  # noqa: F401 - register benchmark environments
    env = gym.make("MSKBenchPowerlift-v0", object_mass_kg=mass)
    try:
        env.reset(seed=0)
        base = env.unwrapped
        index = base.sim.model.body_name2id("dumbbell")
        assert base.sim.model.body_mass[index] == pytest.approx(mass)
        assert base.sim_obsd.model.body_mass[index] == pytest.approx(mass)
        assert np.isfinite(env.step(np.zeros(env.action_space.shape))[1])
    finally:
        env.close()


def test_evaluator_scale_defaults_match_protocol():
    root = Path(__file__).resolve().parents[1]
    files = list((root / "rl_paradigms").glob("*/eval_*_success.py"))
    files += list((root / "rl_paradigms").glob("*/eval_*_robustness.py"))
    for path in files:
        source = path.read_text(encoding="utf-8")
        assert 'default="0,0.02,0.05,0.08,0.12"' not in source, path
        assert 'default="0,0.01,0.02,0.04,0.06"' not in source, path
