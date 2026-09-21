import subprocess

from benchmark_eval import evaluate


def test_child_preserves_no_bytecode_flag(monkeypatch):
    commands = []
    monkeypatch.setattr(evaluate.sys, 'dont_write_bytecode', True)
    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(evaluate.subprocess, 'run', run)
    request = evaluate.EvaluationRequest(algorithms=('ppo',), execute=True)
    evaluate.run_request(request)
    assert commands[0][1] == '-B'
