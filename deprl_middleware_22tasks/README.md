# depRL Middleware for 22 MSK-Bench Tasks

This folder organizes the middleware code into one importable package and adds depRL configs for all 22 MSK-Bench tasks.

## Files

- `deprl_middleware_22tasks/`: package with the Transformer networks, wrapper, registry, and expert-data helpers.
- `configs/`: 22 depRL YAML files, one per MSK-Bench task.
- `collect_expert_data.py`: collect expert actions and muscle physics tensors from a depRL checkpoint.
- `train_expert_transformer.py`: train encoder/decoder weights from collected expert data.
- `render_middleware.py`: render a depRL checkpoint through the middleware environment.
- `generate_configs.py`: regenerate the 22 YAML files, optionally with absolute encoder/decoder paths.

## Install / expose to depRL

From this folder:

```powershell
pip install -e .
```

Or set `PYTHONPATH` before launching depRL:

```powershell
$env:PYTHONPATH="D:\MSK-Bench\deprl_middleware_22tasks;D:\MSK-Bench\MSK-Bench;$env:PYTHONPATH"
```

## Train depRL with middleware

```powershell
cd D:\MSK-Bench\depRL
python -m deprl.main D:\MSK-Bench\deprl_middleware_22tasks\configs\msk_bench_walk_middleware.yaml
```

Each YAML imports `deprl_middleware_22tasks.registry`, which registers `MSKBench...-Middleware-v0` environments.

## Regenerate configs with trained teacher weights

```powershell
python D:\MSK-Bench\deprl_middleware_22tasks\generate_configs.py `
  --encoder-path D:\MSK-Bench\deprl_middleware_22tasks\artifacts\spinal_encoder_weights.pth `
  --decoder-path D:\MSK-Bench\deprl_middleware_22tasks\artifacts\spinal_decoder_weights.pth
```

When weights are not provided, the wrapper safely falls back to pass-through behavior unless `strict_weights=True`.
