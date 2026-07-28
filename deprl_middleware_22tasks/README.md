# depRL Middleware for 22 MSK-Bench Tasks

This directory contains the latent-action middleware wrapper, config generation utilities, expert-data collection script, transformer training script, and evaluator wrappers for all canonical MSK-Bench tasks.

## Attribution

- Upstream: this package is a local MSK-Bench integration built around depRL-style latent-action workflows. Related depRL upstream code is tracked at https://github.com/martius-lab/depRL.
- License: original MSK-Bench wrapper code is released under the repository's top-level MIT License; depRL-derived components retain the license files in `../depRL/`.
- Local MSK-Bench changes: 22-task middleware registration, benchmark config generation, expert-data collection, transformer training, encoder/decoder artifact handling, evaluator entry points, JSON/CSV result export, EMG export, and rendering helpers.

Do not remove this attribution section. This README describes the MSK-Bench middleware integration and should not be replaced by an upstream README from a related dependency.

## Training

Generate middleware configs, optionally embedding encoder and decoder weight paths:

```powershell
python deprl_middleware_22tasks\generate_configs.py --output-dir deprl_middleware_22tasks\configs --encoder-path artifacts\spinal_encoder_weights.pth --decoder-path artifacts\spinal_decoder_weights.pth
```

Collect expert data and train encoder/decoder weights when needed:

```powershell
python deprl_middleware_22tasks\collect_expert_data.py --env MSKBenchReach-v0 --checkpoint-dir checkpoints\deprl\reach --samples 50000 --output artifacts\expert_synergy.pt
python deprl_middleware_22tasks\train_expert_transformer.py --data artifacts\expert_synergy.pt --output-dir artifacts --latent-dim 64
```

Train a middleware policy through depRL/Tonic:

```powershell
cd D:\MSK-Bench\depRL
python -m deprl.main ..\deprl_middleware_22tasks\configs\msk_bench_walk_middleware.yaml
```

Generated configs use `working_dir: ./baselines_MSKBench_Middleware`, so policy checkpoints are written under `depRL\baselines_MSKBench_Middleware\<tonic-name>\<timestamp>\checkpoints\`.

## Evaluation

```powershell
python deprl_middleware_22tasks\eval_middleware_success.py --env MSKBenchWalk-v0 --episodes 10 --benchmark-root D:\MSK-Bench --run-path depRL\baselines_MSKBench_Middleware\<tonic-name>\<timestamp> --checkpoint last
```

Use `--run-path`, `--checkpoint`, or `--checkpoint-file` for trained middleware policies. Use `--encoder-path`, `--decoder-path`, and `--strict-weights` in middleware data/render workflows that require explicit middleware weights.

See `../THIRD_PARTY_NOTICES.md` and `../PATCHES.md` for the release-facing third-party notice and local patch summary.
