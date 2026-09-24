# MSK-Bench

<p align="center">
  <a href="https://zzongzheng0918.github.io/MSK-Bench/"><img alt="Project Page" src="https://img.shields.io/badge/PROJECT%20PAGE-WEBSITE-2F80ED?style=for-the-badge&amp;labelColor=3B3B3B"></a>
  <a href="https://arxiv.org/abs/2609.26872v1"><img alt="arXiv paper" src="https://img.shields.io/badge/ARXIV-2609.26872-B31B1B?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white&amp;labelColor=3B3B3B"></a>
  <a href="https://huggingface.co/Zzz0918/MSK-Bench"><img alt="Model" src="https://img.shields.io/badge/MODEL-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=black"></a>
</p>

MSK-Bench is a full-body musculoskeletal motor-control benchmark with 22 tasks across stabilization, locomotion, and physical interaction. This repository provides the environments, baseline training/evaluation/rendering entry points, physiology-oriented metrics, and source code for focused control studies described in the project.

- Project page: https://zzongzheng0918.github.io/MSK-Bench/
- Paper: [arXiv](https://arxiv.org/abs/2609.26872v1) | [PDF](https://arxiv.org/pdf/2609.26872v1)
- Model: https://huggingface.co/Zzz0918/MSK-Bench
- Source code: this `main` branch
- User guide: [USER_GUIDE.md](USER_GUIDE.md)
- Data and license boundaries: [docs/data-and-licenses.md](docs/data-and-licenses.md)
- Functional validation record: [docs/validation.md](docs/validation.md)

## Installation

Use Python 3.11. The `main` branch contains the source release and project website. Clone the repository, then install from its root:

```bash
git clone https://github.com/ZZongzheng0918/MSK-Bench.git
cd MSK-Bench
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all,dev]"
python -m pip check
```

No external source checkout is required. Rendering requires a working OpenGL driver (GLFW on Windows/macOS or EGL on headless Linux); operating-system graphics libraries are not Python dependencies.

The official `amathislab/mm-fullbody-base` checkpoint is downloaded when a real pretrained residual-control run is requested. It is not trained locally or redistributed here. The resolved checkpoint path is passed explicitly to the residual environment; a bare `gym.make` call does not silently enable pretrained inference.

## Model weights: obtain separately

This is a source-code release, not a checkpoint or experiment-results archive.
Weights do not need to be committed to use or contribute to the project.

- MuscleMimic base policy: download from the upstream
  [amathislab/mm-fullbody-base model repository](https://huggingface.co/amathislab/mm-fullbody-base).
  Use the existing loader after installation; it downloads/caches the checkpoint
  and resolves the concrete checkpoint directory:

  ```bash
  python -B -c "from musclemimic.runner.checkpointing import _canonicalize_resume_path; print(_canonicalize_resume_path('hf://amathislab/mm-fullbody-base'))"
  ```

  To reuse a local checkpoint, set MSK_BENCH_MUSCLEMIMIC_CHECKPOINT to its directory,
  or pass the directory explicitly. For example:

  ```python
  import gymnasium as gym
  import msk_bench
  from musclemimic.integrations.msk_bench import resolve_checkpoint_source
  from musclemimic.runner.checkpointing import _canonicalize_resume_path

  checkpoint = _canonicalize_resume_path(resolve_checkpoint_source(None))
  env = gym.make("MSKBenchResidualStair-v0", base_model_dir=checkpoint)
  try:
      observation, info = env.reset(seed=0)
  finally:
      env.close()
  ```

  The first uncached request requires network access. Observe the upstream model's
  terms; no copy of the checkpoint is included here. This does not change or
  retrain the upstream MuscleMimic integration.
- PPO, SAC, depRL, DynSyn and learned middleware weights: generate them using the
  repository's [training instructions](USER_GUIDE.md#training), or supply
  your own compatible checkpoints. The [evaluation guide](benchmark_eval/README.md#loading-trained-weights)
  documents model, normalization and run-directory arguments. No public download
  location for the authors' task-specific trained checkpoints is asserted.

## Included functionality

The repository includes:

- 22 canonical MSK-Bench environments using the 416-muscle full-body model.
- PPO, SAC, depRL, DynSyn/msgym, latent-action middleware, MuscleMimic integration, and residual-control entry points.
- Unified success, robustness, activation-cost, joint-smoothness, peak-efficiency, and EMG-envelope evaluation utilities.
- Short-training, checkpoint reload, evaluation, and video-rendering smoke workflows.
- AgenticWalk and the repository-provided ResidualStair prior and dedicated stair MJCF model.
- Source code for the direct GPT-6 muscle-control diagnostic.

### Included EMG code and MS-Human-700 configurations

EMG functionality is included in [benchmark_eval/emg_export.py](benchmark_eval/emg_export.py)
and [msk_bench/analysis/emg.py](msk_bench/analysis/emg.py), with a file-based comparison
entry point in [benchmark_eval/analyze.py](benchmark_eval/analyze.py).
These export simulated activations and process/compare EMG envelopes; external
human recordings are distinct from this code. The historical plotting script
accepted CleanEMG_Data.mat via --human-mat; that data file was not found in the
current source tree. It is not a prerequisite for installing or training the benchmark.

The [MS-Human-700 asset directory](msk_bench/simhive/ms_human_700/) is included:

- MS-Human-700.xml: the full 700-actuator model.
- MS-Human-700-Locomotion.xml and MS-Human-700-Manipulation.xml: reduced model variants.
- Corresponding environment implementations under
  [rl_paradigms/msgym/msgym/envs](rl_paradigms/msgym/msgym/envs/):
  locomotionFull_v1.py, locomotionLegs_v1.py and manipulation_v1.py.

These models/configurations and their source code are not missing from the release.
They are separate from the canonical 22-task, 416-muscle benchmark. Stored trained
policies, human recording files and historical figure inputs are optional external
experiment artifacts, not required contents of the code repository.

Most canonical training, evaluation, and rendering workflows run without gated motion data. `MSKBenchResidualWalk-v0` and `MSKBenchResidualRun-v0` additionally require two separately obtained reference motions.

## Restricted motion references

The following AMASS-derived motions originate from [amathislab/musclemimic-retargeted](https://huggingface.co/datasets/amathislab/musclemimic-retargeted) and are intentionally not redistributed:

| Task | Required destination | Reference SHA-256 |
| --- | --- | --- |
| Residual Walk | `rl_paradigms/residualrl/walking_medium09_poses.npz` | `8f320295504ffc0a7759ba76ee454dd2b1e3f1feb7244d476b2373100235cc9b` |
| Residual Run | `rl_paradigms/residualrl/walking_run04_poses.npz` | `a887fe08cf92a1d12ec6b470eda997f912be2e651451fcff93ea7c1db0feca63` |

Each user must obtain permitted access from the provider and comply with both the dataset and [AMASS](https://amass.is.tue.mpg.de/license.html) terms. Authentication tokens must never be committed.

The verified Run reference can be downloaded after access is granted:

```bash
hf auth login
hf download amathislab/musclemimic-retargeted MyoFullBody/gmr/KIT/314/walking_run04_poses.npz --repo-type dataset --revision 0c1c8f9ead144b2d783e900f8fb640d2f7a815ce --local-dir authorized-motions
```

The exact Walk reference used by this release is not the same as the current same-name upstream candidate. The current candidate has SHA-256 `6b2c8b96a5f1cafec9dd695bc8ee9aa5814a4e7d6ba361d27167f002dc39f1fc`; equivalence has not been established. Exact Walk reproduction therefore requires an authorized historical copy matching the reference hash above.

After obtaining authorized files, verify and import them:

```bash
python -B tools/prepare_authorized_motions.py --source-dir authorized-motions
python -B tools/prepare_authorized_motions.py
```

The helper is offline, validates all required hashes before copying, refuses to overwrite different files, and never accepts licenses or dataset terms on a user's behalf. Metadata is stored in [docs/restricted-motions.json](docs/restricted-motions.json).

The included `clean_walk.npy` and `stair_prior_89d.npz` were generated by the MSK-Bench authors. Other retained assets remain subject to the terms listed in [docs/data-and-licenses.md](docs/data-and-licenses.md).

## GPT-6 direct muscle-control diagnostic

The source implementation is in [experiments/gpt6_direct_control](experiments/gpt6_direct_control/README.md). It exposes a paused, file-mediated interface in which a controller reads the structured policy observation and returns one validated 354-dimensional excitation array. Each accepted action is held for 0.2 simulated seconds in a 100-Hz control loop.

The repository contains experiment code and execution instructions only. It does not include an LLM client, hidden reasoning, generated actions, observations, trajectories, videos, figures, or reports. Reproducing a new LLM-driven rollout requires recording the exact model identifier, model settings, prompts, and submitted action files. This source release reproduces the protocol, not previously generated model decisions.

## Quick functional checks

Metric definitions, robustness AUC aggregation, EMG comparison, latent-dimension
configuration, agentic constraints, and load/recruitment analysis are documented in
[Quantitative analysis workflows](docs/analysis-workflows.md). It also lists the
external artifacts and protocol details still needed for exact experiment replication.

Run the regression suite without the gated Walk and Run motions:

```bash
python -B -m pytest -q -p no:cacheprovider
```

Run representative short training, evaluation, and rendering:

```bash
python -B tools/smoke_all.py --phase training --env MSKBenchSquat-v0 --output-dir .smoke/public --keep-going
python -B tools/smoke_all.py --phase evaluation --env MSKBenchSquat-v0 --training-dir .smoke/public/training --output-dir .smoke/evaluation --keep-going
python -B tools/smoke_all.py --phase rendering --env MSKBenchSquat-v0 --training-dir .smoke/public/training --output-dir .smoke/rendering --keep-going
```

These smoke tests establish that entry points run, models are saved/reloaded, metrics are exported, and videos are decodable. They do not establish convergence or reproduce paper performance.

After both exact restricted references are available, run the complete environment matrix:

```bash
python -B tools/prepare_authorized_motions.py
python -B tools/smoke_all.py --phase environments --output-dir .smoke/environments --keep-going
```

Real pretrained checks are opt-in:

```bash
# PowerShell
$env:MSK_BENCH_TEST_PRETRAINED = "1"
python -B -m pytest tests/test_pretrained_runtime.py -q -rs -p no:cacheprovider
```

## Citation

If you use MSK-Bench in research, please cite the [arXiv paper](https://arxiv.org/abs/2609.26872v1):

```bibtex
@misc{ou2026mskbenchbenchmarkingfullbodymusculoskeletal,
      title={MSK-Bench: Benchmarking Full-Body Musculoskeletal Motor Control Across Tasks, Control Paradigms, and Physiological Metrics},
      author={Mengtao Ou and Zongzheng Zhang and Zhenghao Xiao and Yixuan Pan and Ziwen Zhuang and Hang Zhao and Hongyang Li and Yanan Sui and Libin Liu and Hao Zhao},
      year={2026},
      eprint={2609.26872},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2609.26872},
}
```

The repository also provides machine-readable citation metadata in [CITATION.cff](CITATION.cff).

## License and attribution

Original MSK-Bench contributions are released under the [Apache License 2.0](LICENSE).

Third-party source code and assets retain their original licenses and copyright notices. See [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [PATCHES.md](PATCHES.md), and the nested license files. Gated datasets, downloaded checkpoints, SMPL-family assets, and other external artifacts are not automatically covered by the top-level code license.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please keep generated checkpoints, logs, videos, local datasets, and access credentials out of Git, preserve upstream notices, and add tests for behavior changes.
