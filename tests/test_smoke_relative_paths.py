import json
from pathlib import Path

from test_smoke_runner import runner


def test_evaluation_manifest_is_absolute_from_external_working_directory(tmp_path, monkeypatch):
    module = runner()
    directory = tmp_path / 'training' / 'ppo'
    directory.mkdir(parents=True)
    manifest = directory / 'training_manifest.json'
    manifest.write_text(json.dumps({'algorithm': 'ppo', 'env_id': 'MSKBenchSquat-v0'}))
    monkeypatch.chdir(tmp_path)
    args = module.build_parser().parse_args(['--phase', 'evaluation', '--algorithm', 'ppo',
                                           '--training-dir', 'training', '--output-dir', 'output'])
    command = module.job_command(module.build_plan(args)[0], args)
    value = command[command.index('--artifact-manifest') + 1]
    assert Path(value).is_absolute()
    assert Path(value) == manifest
