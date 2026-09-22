import importlib.util
from collections import Counter
from pathlib import Path

import pytest


def runner():
    path = Path(__file__).resolve().parents[1] / 'tools/smoke_all.py'
    spec = importlib.util.spec_from_file_location('smoke_all', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_plan_covers_every_entry(tmp_path):
    module = runner()
    args = module.build_parser().parse_args(['--phase', 'all', '--output-dir', str(tmp_path)])
    jobs = module.build_plan(args)
    assert Counter(job['phase'] for job in jobs) == {
        'environments': 26, 'training': 5, 'evaluation': 25, 'rendering': 5}
    assert len({job['name'] for job in jobs}) == 61


def test_filters_and_missing_manifest(tmp_path):
    module = runner()
    args = module.build_parser().parse_args(['--phase', 'evaluation', '--algorithm', 'ppo',
                                           '--output-dir', str(tmp_path), '--keep-going'])
    assert len(module.build_plan(args)) == 5
    with pytest.raises(FileNotFoundError, match='manifest'):
        module.find_manifest(tmp_path, 'ppo', 'MSKBenchSquat-v0')


def test_report_is_saved_when_job_fails(tmp_path, monkeypatch):
    import json
    import subprocess
    module = runner()
    monkeypatch.setattr(module, 'build_plan', lambda args: [
        {'name': 'broken', 'phase': 'environments', 'env': 'MSKBenchSquat-v0', 'algorithm': None,
         'metric': None}])
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw:
                        subprocess.CompletedProcess(a[0], 2, 'out', 'intentional failure'))
    assert module.main(['--output-dir', str(tmp_path)]) == 1
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['failed'] == 1
    assert report['results'][0]['returncode'] == 2
