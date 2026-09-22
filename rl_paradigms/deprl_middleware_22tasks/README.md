# depRL Middleware for 22 MSK-Bench Tasks

This directory contains the latent-action middleware wrapper, config generation utilities, expert-data collection script, transformer training script, and evaluator wrappers for all canonical MSK-Bench tasks.

## Attribution

- Upstream: this package is a local MSK-Bench integration built around depRL-style latent-action workflows. Related depRL upstream code is tracked at https://github.com/martius-lab/depRL.
- License: depRL-derived components retain the license files in `../depRL/`; original MSK-Bench wrapper contributions use Apache-2.0.
- Local MSK-Bench changes: 22-task middleware registration, benchmark config generation, expert-data collection, transformer training, encoder/decoder artifact handling, evaluator entry points, JSON/CSV result export, EMG export, and rendering helpers.

Do not remove this attribution section. This README describes the MSK-Bench middleware integration and should not be replaced by an upstream README from a related dependency.

## Training

From the repository root, collect Walk expert data from your trained checkpoint and train matching encoder/decoder weights:

```powershell
python rl_paradigms\deprl_middleware_22tasks\collect_expert_data.py --env MSKBenchWalk-v0 --checkpoint-dir checkpoints\deprl\walk --samples 50000 --output artifacts\expert_synergy.pt
python rl_paradigms\deprl_middleware_22tasks\train_expert_transformer.py --data artifacts\expert_synergy.pt --output-dir artifacts --latent-dim 64
```

Resolve existing weights to absolute paths before changing directories, then generate strict-loading configs:

```powershell
$encoderPath = (Resolve-Path artifacts\spinal_encoder_weights.pth -ErrorAction Stop).Path
$decoderPath = (Resolve-Path artifacts\spinal_decoder_weights.pth -ErrorAction Stop).Path
python rl_paradigms\deprl_middleware_22tasks\generate_configs.py `
  --output-dir rl_paradigms\deprl_middleware_22tasks\configs `
  --latent-dim 64 --strict-weights `
  --encoder-path "$encoderPath" `
  --decoder-path "$decoderPath"
```

Keep generated machine-specific configs out of Git. Use compatible expert data and weight architectures for each target environment. Non-strict configs can silently fall back to pass-through mode without weights; this is a pipeline check, not a learned middleware baseline.

Train the generated Walk policy through depRL/Tonic:

```powershell
Push-Location rl_paradigms\depRL
python -m deprl.main ..\deprl_middleware_22tasks\configs\msk_bench_walk_middleware.yaml
Pop-Location
```

Generated configs use `working_dir: ./baselines_MSKBench_Middleware`, so policy checkpoints are written under `rl_paradigms\depRL\baselines_MSKBench_Middleware\<tonic-name>\<timestamp>\checkpoints\`.

## Evaluation

```powershell
python rl_paradigms\deprl_middleware_22tasks\eval_middleware_success.py --env MSKBenchWalk-v0 --episodes 10 --benchmark-root . --run-path rl_paradigms\depRL\baselines_MSKBench_Middleware\<tonic-name>\<timestamp> --checkpoint last
```

Use `--run-path`, `--checkpoint`, or `--checkpoint-file` for trained middleware policies. Use `--encoder-path`, `--decoder-path`, and `--strict-weights` in middleware data/render workflows that require explicit middleware weights.

See `../../THIRD_PARTY_NOTICES.md` and `../../PATCHES.md` for the release-facing third-party notice and local patch summary.
