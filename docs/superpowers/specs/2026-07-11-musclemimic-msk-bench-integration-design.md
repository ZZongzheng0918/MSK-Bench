# MuscleMimic 与 MSK-Bench ResidualRL 集成设计

## 背景与目标

MSK-Bench 的 `residualrl` 环境目前直接导入 MuscleMimic 内部模块，并默认引用不存在的 `residualrl/models/myofullbody.xml` 与 `residualrl/base_policy`。项目还包含一个嵌套的 `musclemimic_models-main` 源码副本，容易与 MuscleMimic 声明的正式依赖发生版本漂移。

本次改造的目标是：

- 保留 MuscleMimic 的训练、评估和动作重定向能力。
- 清理确认无运行价值的示例、生成物、缓存和外围仓库配置。
- 为 MSK-Bench 提供稳定、明确的 MuscleMimic 集成边界。
- 不在 benchmark 中提交基础策略 checkpoint；按照 MuscleMimic 官方 README 获取。
- 让模型、checkpoint 和 Python 导入失败时给出可操作的错误信息。
- 为整个 `D:\MSK-Bench` 工作区提供统一 README。

## 非目标

- 不重写 MuscleMimic 的 PPO、环境或动作重定向算法。
- 不把 MuscleMimic 的内部实现复制到 MSK-Bench。
- 不提交 Hugging Face checkpoint、用户缓存或生成输出。
- 不承诺在未安装依赖和未取得 checkpoint 的机器上运行完整策略推理。
- 不删除来源或用途不明确的自定义文件。

## 选定方案

采用“外部可编辑依赖 + 稳定适配层”。MuscleMimic 保持独立项目，由其 `pyproject.toml` 管理 JAX、MuJoCo、`musclemimic_models` 等依赖；MSK-Bench 的 ResidualRL 只通过专用公共适配模块访问环境和策略加载能力。

不采用以下方案：

- 在 `residualrl` 内继续维护 `musclemimic_models-main` 副本，因为会重复打包资源并产生版本漂移。
- 把 MuscleMimic 所需模块复制进 benchmark，因为会破坏上游训练、评估和重定向代码的一致性。

## 架构与模块边界

### MuscleMimic 侧

新增轻量的 benchmark 集成模块，作为 MSK-Bench 使用的公共边界。该模块负责：

- 导出 `MyoFullBody` 环境类。
- 解析 checkpoint 来源。
- 加载 PPO checkpoint 并构造推理函数。
- 对外隐藏 `algorithms`、`runner.eval_utils` 等内部路径。

现有训练、评估和动作重定向入口保持不变。适配模块只组合已有能力，不改变算法行为。

### MSK-Bench ResidualRL 侧

`residualrl/common.py` 负责 benchmark 本地的轨迹适配、目标观测和 Gymnasium 环境包装，但不再拼接 MuscleMimic 内部模块路径。

模型路径规则：

1. 调用方显式传入 `model_path` 时使用该路径。
2. 未传入时向 `MyoFullBody` 传递 `None`，由 MuscleMimic 使用已安装的 `musclemimic_models.get_xml_path("myofullbody")`。

checkpoint 规则：

1. `base_model_dir` 显式参数优先。
2. 其次读取 `MSK_BENCH_MUSCLEMIMIC_CHECKPOINT` 环境变量。
3. 最后使用官方 URI `hf://amathislab/mm-fullbody-base`。

官方 URI 由 MuscleMimic 已有的 checkpoint 规范化逻辑处理。首次使用需要网络、Hugging Face 访问权限以及单独接受资源许可证。调用方也可以提前下载并提供本地 checkpoint 路径。

### 依赖边界

MuscleMimic 的可选依赖中新增 benchmark 运行所需的 Gymnasium 依赖，使安装命令能够显式表达用途。MSK-Bench 源码目录通过 README 中的 `PYTHONPATH` 或当前项目既有启动方式暴露，不在本次范围内把整个工作区重构成单一 Python distribution。

## 数据与控制流

1. 用户按根 README 安装本地 MuscleMimic 及 benchmark 可选依赖。
2. 导入 `msk_bench` 时注册 `ResidualWalk`、`ResidualRun` 和 `ResidualStair`。
3. Gymnasium 通过 `make_env` 创建 ResidualRL 环境。
4. `MyoFullBody` 从已安装的 `musclemimic_models` 解析默认 XML。
5. `NPZTrajectoryAdapter` 读取 benchmark 内随任务发布的参考轨迹。
6. 第一次执行 residual step 时，适配层解析并缓存 checkpoint 策略；后续 step 复用同一进程内的缓存。
7. wrapper 将基础策略动作与 residual action 合成，再计算任务奖励和终止条件。

## 错误处理

- 未安装 MuscleMimic：提示安装本地 `musclemimic-main`，不伪装成普通环境注册错误。
- 未安装 `musclemimic_models`：提示运行 MuscleMimic 依赖同步命令。
- checkpoint 下载或本地路径解析失败：错误中显示最终解析来源、环境变量名和 README 配置入口。
- 显式模型路径不存在：在 MuJoCo 加载前抛出包含绝对路径的 `FileNotFoundError`。
- 观测或动作维度与 checkpoint 不一致：在首次策略调用时报告期望值和实际值。
- 参考轨迹缺少 `states`/`qpos`：保留当前明确的 `KeyError`，并包含轨迹路径。

## 清理范围

### 删除

- `musclemimic-main/outputs/`
- `musclemimic-main/musclemimic.egg-info/`
- `musclemimic-main/tea_debug.log`
- `musclemimic-main/examples/`
- `musclemimic-main/.github/`
- 项目内的 `__pycache__`、`.pytest_cache`、`.ruff_cache` 等生成缓存
- `residualrl/musclemimic_models-main/`，由正式 `musclemimic-models` 依赖替代

### 保留

- `musclemimic/` 与 `loco_mujoco/`
- `bimanual/`、`fullbody/` 和 `scripts/`
- `tests/`、`pyproject.toml`、`uv.lock`、许可证和贡献说明
- `assets/`，因为原始 README 仍引用其中图片和动画
- `retarget.py` 与 `run_my_retarget.py`，其时间和用途表明可能是用户自定义入口
- `residualrl` 的三个参考轨迹 NPZ 文件

删除前再次列出目标并验证所有解析后的绝对路径都位于上述两个明确目录内。不得删除未列入本设计的文件。

## README 设计

在 `D:\MSK-Bench\README.md` 说明：

- 工作区目录关系和各子项目职责。
- MuscleMimic 官方系统要求；训练需要 Linux/NVIDIA GPU，推理按上游支持范围说明。
- 使用 `uv sync` 安装 MuscleMimic，以及安装 benchmark 可选依赖。
- `musclemimic_models` 随 MuscleMimic 依赖安装，无需在 residualrl 内保留源码副本。
- checkpoint 的官方 Hugging Face 地址、本地路径覆盖和环境变量配置。
- 设置 MSK-Bench 源码路径并导入环境的示例。
- ResidualWalk、ResidualRun、ResidualStair 的最小创建/reset/step 示例。
- 常见问题：缺少 Gymnasium、模型包、checkpoint、Hugging Face 权限、Windows 平台差异和维度不匹配。
- checkpoint 与数据采用独立许可证，不能随 benchmark 重新分发。

## 测试策略

### 无 checkpoint 的确定性测试

- 适配模块可以导入，且不会在导入阶段下载 checkpoint。
- checkpoint 来源优先级正确：参数高于环境变量，高于官方 URI。
- 未显式传模型路径时使用 MuscleMimic 默认模型解析。
- 显式不存在的模型路径产生清晰错误。
- `residualrl` 三个模块不再导入 MuscleMimic 内部实现路径。
- README 中引用的本地目录和环境 ID 存在。

### 安装完整依赖后的集成测试

- 导入并注册三个 ResidualRL 环境。
- 使用本地或已缓存 checkpoint 创建环境并执行 `reset()`。
- 对每个环境执行至少一个 `step()`，验证观测维度、动作维度、奖励类型和 Gymnasium 五元组。
- 运行 MuscleMimic 与本次适配相关的现有测试，确保训练、评估和重定向入口未被裁剪破坏。

### 受限环境下的验收说明

当前机器若缺少 Gymnasium、JAX/MuJoCo 完整版本或 checkpoint，只能完成静态导入边界、路径解析和不触网单元测试。最终交付必须明确区分“当前已运行验证”和“需要完整 MuscleMimic 环境执行的验证命令”。

## 完成标准

- 清理仅覆盖明确批准的外围文件。
- MuscleMimic 的训练、评估、重定向代码与配置仍在。
- ResidualRL 不再引用不存在的默认 XML 或本地 `base_policy` 目录。
- ResidualRL 通过稳定适配层访问 MuscleMimic。
- 根 README 可以指导新用户安装模型依赖、获取 checkpoint 并启动三个 residual 环境。
- 无 checkpoint 测试通过；完整依赖环境的验证命令和限制被如实记录。
