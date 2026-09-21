from test_deprl_training_runtime import run_isolated


def test_middleware_generator_accepts_training_overrides(tmp_path):
    run_isolated([
        "-m", "rl_paradigms.deprl_middleware_22tasks.generate_configs",
        "--output-dir", str(tmp_path / "configs"), "--steps", "8",
        "--epoch-steps", "8", "--save-steps", "8", "--parallel", "1",
        "--sequential", "1", "--working-dir", str(tmp_path / "runs"), "--cpu",
    ], tmp_path)
    import yaml

    files = list((tmp_path / "configs").glob("*.yaml"))
    assert len(files) == 22
    for path in files:
        config = yaml.safe_load(path.read_text())
        assert config["tonic"]["parallel"] == config["tonic"]["sequential"] == 1
        assert config["tonic"]["cpu_override"]
        assert config["working_dir"] == str(tmp_path / "runs")
        assert config["trainer_args"] == {"steps": 8, "epoch_steps": 8, "save_steps": 8}
        assert "sys.path" not in config["tonic"]["header"]
