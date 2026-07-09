# msgym / DynSyn-SAC Benchmark

This directory integrates msgym and DynSyn-SAC as a benchmark algorithm alongside `depRL`, `ppo`, and `sac`.

## Contents

- `msgym/`: Gymnasium environments for MS-Human-700.
- `DynSyn/`: DynSyn SAC implementation.
- `SB3-Scripts/`: training and evaluation entry points.
- `configs/`: 22 normalized MSK-Bench training configs with project-relative output paths.

The canonical MS-Human-700 model assets live in:

```text
D:\MSK-Bench\MSK-Bench\msk_bench\simhive\ms_human_700
```

`msgym.envs.utils.get_ms_human_model_path()` prefers that MSK-Bench asset location and falls back to local msgym assets if present.

## Usage

```powershell
cd D:\MSK-Bench\msgym
python SB3-Scripts\train.py --list-configs
python SB3-Scripts\train.py -f configs\msk_bench_walk.json
python SB3-Scripts\eval.py -f runs\msgym_logs\MSKBenchWalk-v0\<run-id>
```

Training outputs are written under `runs/msgym_logs` by default.