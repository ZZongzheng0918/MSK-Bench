"""Real, isolated MSK-Bench acceptance jobs; no pytest internals or external checkouts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from msk_bench.registry import CANONICAL_ENV_IDS
from msk_bench.validation import validate_table, validate_video

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ('ppo', 'sac', 'deprl', 'msgym', 'middleware')
EXTENSIONS = ('MSKBenchResidualRun-v0', 'MSKBenchResidualWalk-v0',
              'MSKBenchResidualStair-v0', 'MSKBenchAgenticWalk-v0')
METRICS = ('success', 'robustness', 'smooth', 'energy', 'emg')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('environments', 'training', 'evaluation', 'rendering', 'all'), default='all')
    parser.add_argument('--env', choices=CANONICAL_ENV_IDS, default='MSKBenchSquat-v0',
                        help='Representative training/evaluation task; environment phase always checks all 26 IDs.')
    parser.add_argument('--algorithm', choices=('all', *ALGORITHMS), default='all')
    parser.add_argument('--output-dir', type=Path, default=Path('.smoke/all'))
    parser.add_argument('--training-dir', type=Path, help='Existing training root for separate evaluation/render phases.')
    parser.add_argument('--json-report', type=Path)
    parser.add_argument('--keep-going', action='store_true')
    parser.add_argument('--timeout', type=int, default=900, help='Seconds per job, including first checkpoint download.')
    parser.add_argument('--skip-pretrained', action='store_true',
                        help='Explicit offline dynamics-only mode; not full pretrained acceptance.')
    return parser


def build_plan(args):
    jobs = []
    algorithms = ALGORITHMS if args.algorithm == 'all' else (args.algorithm,)
    def add(phase, env, algorithm=None, metric=None):
        name = '/'.join(str(x) for x in (phase, algorithm, env, metric) if x)
        jobs.append(dict(name=name, phase=phase, env=env, algorithm=algorithm, metric=metric))
    if args.phase in ('all', 'environments'):
        for env in (*CANONICAL_ENV_IDS, *EXTENSIONS):
            add('environments', env)
    if args.phase in ('all', 'training'):
        for algorithm in algorithms:
            add('training', args.env, algorithm)
    if args.phase in ('all', 'evaluation'):
        for algorithm in algorithms:
            for metric in METRICS:
                add('evaluation', args.env, algorithm, metric)
    if args.phase in ('all', 'rendering'):
        for algorithm in algorithms:
            add('rendering', args.env, algorithm, 'render')
    return jobs


def find_manifest(root, algorithm, env_id):
    matches = []
    for path in Path(root).rglob('training_manifest.json'):
        value = json.loads(path.read_text(encoding='utf-8'))
        if value['algorithm'] == algorithm and value['env_id'].replace('-Middleware-', '-') == env_id:
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one training manifest for {algorithm}/{env_id} under {root}; found {len(matches)}')
    return matches[0].resolve()


def environment_code(env_id, pretrained):
    return f"""
import gymnasium as gym
import numpy as np
import msk_bench
kwargs = {{}}
if 'Residual' in {env_id!r}:
    if {pretrained!r}:
        from musclemimic.integrations.msk_bench import resolve_checkpoint_source
        from musclemimic.runner.checkpointing import _canonicalize_resume_path
        kwargs['base_model_dir'] = _canonicalize_resume_path(resolve_checkpoint_source(None))
    else:
        kwargs['base_model_dir'] = None
env = gym.make({env_id!r}, **kwargs)
try:
    obs, info = env.reset(seed=17)
    assert env.observation_space.contains(obs), 'reset observation outside declared space'
    if 'Residual' in {env_id!r} and {pretrained!r}:
        assert env.get_wrapper_attr('_base_policy_fn') is not None
    env.action_space.seed(17)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs), 'step observation outside declared space'
    assert np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict)
    print('ENVIRONMENT_OK', {env_id!r})
finally:
    env.close()
"""


def training_command(algorithm, env_id, output):
    from rl_paradigms.deprl_middleware_22tasks.generate_configs import TASKS, config_text
    task = next(task for task in TASKS if task.env_id == env_id)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.rglob('training_manifest.json')):
        raise FileExistsError(f'Training output already contains artifacts: {output}. Use a fresh output directory.')
    prefix = [sys.executable, '-B', '-I']
    if algorithm in ('ppo', 'sac'):
        extra = (['--n-steps', '4', '--batch-size', '4', '--n-epochs', '1'] if algorithm == 'ppo' else
                 ['--learning-starts', '1', '--buffer-size', '32', '--batch-size', '2',
                  '--train-freq', '1', '--gradient-steps', '1'])
        return prefix + ['-m', f'rl_paradigms.{algorithm}.train_{algorithm}_msk_bench',
                         '--env', env_id, '--total-timesteps', '8', '--num-envs', '1', '--eval-envs', '1',
                         '--eval-freq', '0', '--output-root', str(output), '--hidden-size', '32',
                         '--verbose', '0', '--no-resume', *extra]
    if algorithm == 'msgym':
        config = ROOT / f'rl_paradigms/msgym/configs/msk_bench_{task.slug}.json'
        return prefix + [str(ROOT / 'rl_paradigms/msgym/SB3-Scripts/train.py'), '-f', str(config),
                         '--total-timesteps', '8', '--env-nums', '1', '--check-freq', '0', '--record-freq', '0',
                         '--dump-freq', '0', '--log-root-dir', str(output), '--no-progress-bar', '--seed', '17',
                         '--learning-starts', '1', '--batch-size', '2', '--buffer-size', '32',
                         '--hidden-size', '32', '--device', 'cpu']
    config = ROOT / f'rl_paradigms/depRL/experiments/msk_bench_training_files/msk_bench_{task.slug}.yaml'
    if algorithm == 'middleware':
        config = output / 'requested_config.yaml'
        config.write_text(config_text(task, None, None), encoding='utf-8')
    return prefix + ['-m', 'deprl.main', str(config), '--steps', '8', '--epoch-steps', '8', '--save-steps', '8',
                     '--parallel', '1', '--sequential', '1', '--seed', '17', '--test-episodes', '0',
                     '--hidden-size', '32', '--batch-size', '2', '--buffer-size', '32',
                     '--steps-before-batches', '1', '--steps-between-batches', '1', '--batch-iterations', '1',
                     '--cpu', '--output-dir', str(output)]


def job_command(job, args):
    root = args.output_dir.resolve()
    phase, algorithm = job['phase'], job['algorithm']
    if phase == 'environments':
        return [sys.executable, '-B', '-I', '-c', environment_code(job['env'], not args.skip_pretrained)]
    if phase == 'training':
        return training_command(algorithm, job['env'], root / 'training' / algorithm)
    training_root = args.training_dir or root / 'training'
    manifest = find_manifest(training_root, algorithm, job['env'])
    command = [sys.executable, '-B', '-I', '-m', 'benchmark_eval.evaluate', '--artifact-manifest', str(manifest),
               '--algorithm', algorithm, '--env', job['env'], '--metric', job['metric'],
               '--episodes', '1', '--max-steps', '4', '--output-dir', str(root / phase / algorithm / job['metric']),
               '--execute']
    if job['metric'] == 'robustness':
        command += ['--', '--noise-type', 'all', '--action-scales', '0,0.02', '--obs-scales', '0,0.01',
                    '--dynamics-scales', '0,0.05']
    elif job['metric'] == 'render':
        command += ['--', '--width', '320', '--height', '240']
    return command


def validate_artifacts(job, root):
    phase, algorithm, metric = job['phase'], job['algorithm'], job['metric']
    if phase == 'environments':
        return []
    if phase == 'training':
        manifest = find_manifest(root / 'training' / algorithm, algorithm, job['env'])
        data = json.loads(manifest.read_text(encoding='utf-8'))
        assert data['timesteps'] >= 8
        paths = [manifest, Path(data['model_path'])]
        if data.get('normalization_path'):
            paths.append(Path(data['normalization_path']))
        assert all(p.is_file() and p.stat().st_size > 0 for p in paths)
        return [str(p) for p in paths]
    directory = root / phase / algorithm / metric
    if metric == 'render':
        paths = list(directory.rglob('*.mp4'))
        assert paths, 'No MP4 generated'
        for path in paths:
            assert validate_video(path) == (240, 320, 3)
    elif metric == 'emg':
        paths = list(directory.rglob('*.csv'))
        assert paths, 'No EMG CSV generated'
        for path in paths:
            validate_table(path)
    else:
        paths = [directory / f'{metric}_{algorithm}.{suffix}' for suffix in ('json', 'csv')]
        for path in paths:
            validate_table(path)
    return [str(p) for p in paths]


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        raise ValueError('--timeout must be positive')
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    report_path = (args.json_report or root / 'report.json').resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = {key: value for key, value in os.environ.items() if key.upper() != 'PYTHONPATH'}
    child_env.update(OMP_NUM_THREADS='1', JAX_PLATFORMS='cpu', PYTHONIOENCODING='utf-8',
                     MSK_BENCH_AGENTIC_STATE_DIR=str(root / 'agentic_state'))
    jobs = build_plan(args)
    report = dict(python=sys.executable, phase=args.phase, pretrained=not args.skip_pretrained,
                  planned=len(jobs), passed=0, failed=0, results=[])
    for index, job in enumerate(jobs, 1):
        print(f'[{index}/{len(jobs)}] {job["name"]}', flush=True)
        start = time.monotonic()
        result = dict(job, command=[], returncode=1, stdout_tail='', stderr_tail='', artifacts=[])
        try:
            command = job_command(job, args)
            result['command'] = command
            completed = subprocess.run(command, cwd=ROOT, env=child_env, capture_output=True,
                                       text=True, encoding='utf-8', errors='replace', timeout=args.timeout)
            result.update(returncode=completed.returncode, stdout_tail=completed.stdout[-6000:],
                          stderr_tail=completed.stderr[-6000:])
            logs = root / 'logs'
            logs.mkdir(exist_ok=True)
            log = logs / (job['name'].replace('/', '_') + '.log')
            log.write_text(completed.stdout + '\n' + completed.stderr, encoding='utf-8')
            result['log_path'] = str(log)
            if completed.returncode == 0:
                result['artifacts'] = validate_artifacts(job, root)
        except Exception as error:
            result['returncode'] = 1
            result['stderr_tail'] += '\n' + repr(error)
        result['duration_seconds'] = round(time.monotonic() - start, 3)
        report['results'].append(result)
        report['passed' if result['returncode'] == 0 else 'failed'] += 1
        report['complete'] = len(report['results']) == len(jobs)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        print(f'  {"PASS" if result["returncode"] == 0 else "FAIL"} ({result["duration_seconds"]}s)', flush=True)
        if result['returncode'] != 0 and not args.keep_going:
            break
    print(f'{report["passed"]} passed, {report["failed"]} failed; report: {report_path}', flush=True)
    return int(report['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
