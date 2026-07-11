# MSK-Bench 工作区

本目录整合了 MSK-Bench 环境、训练算法以及 MuscleMimic residual-control 扩展。ResidualRL 使用 MuscleMimic 的 MyoFullBody 模型和基础 PPO 策略，再将 benchmark 智能体输出作为 residual action 叠加到基础动作上。

## 目录

- `MSK-Bench/`：`msk_bench` Python 包和标准/ResidualRL 环境。
- `musclemimic-main/`：保留完整训练、评估和动作重定向能力的 MuscleMimic 源码。
- `depRL/`、`ppo/`、`sac/`、`msgym/`：benchmark 算法与相关入口。
- `tests/`：不依赖 checkpoint 的工作区合同测试。

ResidualRL 不包含 MuscleMimic checkpoint，也不再内嵌 `musclemimic-models` 源码副本。

## 系统要求

MuscleMimic 要求 Python 3.11 或更高版本，并使用 uv 管理依赖。根据上游 MuscleMimic README：

- 训练需要 Linux、NVIDIA GPU 和兼容的 CUDA/JAX 环境。
- 推理与评估由上游正式支持 Linux 和 macOS。
- Windows 不在上游正式支持范围内；可进行静态测试，但完整 JAX/MuJoCo 运行可能需要 Linux 或 WSL2。

## 安装

先安装 [uv](https://docs.astral.sh/uv/)，然后在 MuscleMimic 项目中同步官方依赖：

```bash
cd musclemimic-main
uv sync
```

ResidualRL 还使用 Gymnasium。由于它属于 MSK-Bench 集成层而不是 MuscleMimic 核心依赖，请安装到同一个 uv 环境：

```bash
uv pip install gymnasium==0.29.1
```

如果之后再次执行 `uv sync`，请重新执行上面的 `uv pip install gymnasium==0.29.1`。运行 benchmark 命令时使用 `uv run --no-sync`，避免 uv 在启动前移除这个集成层依赖。

MuscleMimic 的 `pyproject.toml` 已声明 `musclemimic-models>=1.0.2`，因此 `uv sync` 会安装正式的 `musclemimic-models` 包和 `myofullbody.xml`；不需要把模型源码复制到 `residualrl`。

### CUDA 训练环境

Linux x86_64 + NVIDIA GPU 可按上游说明安装 CUDA extra：

```bash
uv sync --extra cuda
uv pip install gymnasium==0.29.1
```

## 配置 Python 路径

从 `musclemimic-main` 目录运行时，把 MSK-Bench 包目录加入 `PYTHONPATH`。

PowerShell：

```powershell
$env:PYTHONPATH = (Resolve-Path '..\MSK-Bench').Path
uv run --no-sync python -c "import msk_bench; print('MSK-Bench registered')"
```

Linux/macOS：

```bash
export PYTHONPATH="$(cd ../MSK-Bench && pwd)"
uv run --no-sync python -c "import msk_bench; print('MSK-Bench registered')"
```

## MuscleMimic checkpoint

默认基础策略来源是：

```text
hf://amathislab/mm-fullbody-base
```

MuscleMimic 的 checkpoint 加载器支持 Hugging Face URI、本地 `checkpoint_<step>` 目录和包含多个 checkpoint 的父目录。首次使用默认 URI 时会联网下载，可能需要 Hugging Face 登录或资源访问权限。

推荐提前下载或缓存 checkpoint，并通过环境变量指定本地目录：

PowerShell：

```powershell
$env:MSK_BENCH_MUSCLEMIMIC_CHECKPOINT = 'D:\checkpoints\mm-fullbody-base'
```

Linux/macOS：

```bash
export MSK_BENCH_MUSCLEMIMIC_CHECKPOINT=/data/checkpoints/mm-fullbody-base
```

创建环境时也可以传入 `base_model_dir`；优先级为：

1. `base_model_dir` 参数；
2. `MSK_BENCH_MUSCLEMIMIC_CHECKPOINT` 环境变量；
3. 官方 Hugging Face URI。

checkpoint 不随本仓库分发。

## ResidualRL 环境

可用环境：

- `MSKBenchResidualWalk-v0`
- `MSKBenchResidualRun-v0`
- `MSKBenchResidualStair-v0`

最小示例：

```python
import gymnasium as gym
import msk_bench  # 导入时注册环境

env = gym.make("MSKBenchResidualWalk-v0")
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```

从 `musclemimic-main` 目录运行：

```bash
uv run --no-sync python path/to/your_script.py
```

第一次执行 `step()` 时会加载基础策略；如果使用默认 Hugging Face URI，此时可能发生下载。

## 路径覆盖

通常不需要传 `model_path`：`MyoFullBody` 会通过已安装的 `musclemimic-models` 解析默认 XML。如果需要自定义模型，可以传入存在的 XML 文件；无效路径会在 MuJoCo 加载前抛出包含绝对路径的 `FileNotFoundError`。

三个参考轨迹仍随 benchmark 保存在：

```text
MSK-Bench/msk_bench/envs/msk/benchmark/residualrl/
```

## 测试

不需要 checkpoint 的工作区测试：

```bash
cd D:/MSK-Bench
python -m unittest discover -s tests -v
```

在完整 MuscleMimic 环境中运行适配器测试：

```bash
cd musclemimic-main
uv run --no-sync pytest -p no:cacheprovider tests/unit/test_msk_bench_integration.py -v
```

完整 smoke test 还需要有效 checkpoint，并应验证三个 residual 环境均可完成 `reset()` 和至少一次 `step()`。

## 常见问题

### 缺少 gymnasium

出现 `Either gym or gymnasium is required` 或 `No module named 'gymnasium'`：

```bash
cd musclemimic-main
uv pip install gymnasium==0.29.1
```

随后使用 `uv run --no-sync`。

### 缺少 musclemimic 或 musclemimic_models

确认当前目录是 `musclemimic-main` 并已运行 `uv sync`。不要把已删除的 `musclemimic_models-main` 源码目录重新加入 `PYTHONPATH`。

### checkpoint 不存在

检查 `MSK_BENCH_MUSCLEMIMIC_CHECKPOINT` 是否指向完整 Orbax checkpoint 或其父目录。也可清除该变量，让加载器使用默认 Hugging Face URI。

### 观测或动作维度不匹配

当前 ResidualRL 策略合同为 2418 维观测和 354 维动作。应使用与 MyoFullBody 关闭手指动作配置匹配的官方基础策略。

### `uv lock` 在 Git extra 上耗时

MuscleMimic 的 SMPL/GMR extras 含 Git 依赖。首次重新解析锁文件可能需要访问 GitHub；普通安装优先使用仓库已有 `uv.lock` 和 `uv sync`。

## License 与资源许可

MuscleMimic 源码使用 Apache-2.0 license。模型、动作数据和 Hugging Face checkpoint 可能采用独立 license；使用或重新分发前请分别查看对应仓库和数据页面，本 benchmark 不重新分发这些 checkpoint。
