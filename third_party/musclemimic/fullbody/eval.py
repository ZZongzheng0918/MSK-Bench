"""
Fullbody policy evaluation and visualization entrypoint.

Common workflows:

1. Basic MJX playback
   uv run python fullbody/eval.py --path outputs/.../checkpoint_123

2. MuJoCo playback or viewer
   uv run python fullbody/eval.py --path outputs/.../checkpoint_123 --use_mujoco
   uv run python fullbody/eval.py --path outputs/.../checkpoint_123 --use_mujoco --mujoco_viewer

3. Record a rollout
   uv run python fullbody/eval.py --path outputs/.../checkpoint_123 --record

4. Inspect available trajectories
   uv run python fullbody/eval.py --path outputs/.../checkpoint_123 --list_trajs

5. Replay a specific trajectory
   uv run python fullbody/eval.py --path outputs/.../checkpoint_123 --traj_index 3 --traj_start_step 0

6. Run validation metrics for the KIT testing group
   uv run python fullbody/eval.py \
     --path hf://amathislab/mm-base-ll \
     --metrics --metrics_only \
     --motion_group KIT_KINESIS_TESTING_MOTIONS \
     --terminal_state_type MeanRelativeSiteDeviationWithRootTerminalStateHandler \
     --root_deviation_threshold 0.5 \
     --eval_seed 0

7. Evaluate all motions with MJX/GPU metrics
   uv run python fullbody/eval.py \
     --path hf://amathislab/mm-base-ll \
     --evaluate_all --metrics --metrics_only \
     --motion_group KIT_KINESIS_TESTING_MOTIONS \
     --terminal_state_type MeanRelativeSiteDeviationWithRootTerminalStateHandler \
     --root_deviation_threshold 0.5 \
     --metrics_envs 8 \
     --eval_seed 0

8. Evaluate all motions with MuJoCo/CPU metrics
   uv run python fullbody/eval.py \
     --path hf://amathislab/mm-base-ll \
     --use_mujoco \
     --evaluate_all --metrics --metrics_only \
     --motion_group KIT_KINESIS_TESTING_MOTIONS \
     --terminal_state_type MeanRelativeSiteDeviationWithRootTerminalStateHandler \
     --root_deviation_threshold 0.5 \
     --eval_seed 0

Useful notes:
- `--traj_index` is 0-based.
- `--motion_group` loads an entire dataset split; use `--motion_path` for a single motion.
- `--metrics_envs` only affects MJX metrics, not MuJoCo metrics.
- Evaluation defaults to `validation.terminal_state_type` / `validation.terminal_state_params`; use `--strict_termination` to keep training-time termination settings.
- Fullbody metrics default to stochastic evaluation; use `--metrics_deterministic` to override.
- `--eval_seed` controls the RNG seed for stochastic action sampling (default: 0). Use different values to test variance across random seeds.
- `--train_state_seed` selects which set of network weights to evaluate when a checkpoint was trained with multiple random seeds (n_seeds > 1). For standard single-seed training this flag has no effect and can be omitted.
- `--terminal_state_type MeanRelativeSiteDeviationWithRootTerminalStateHandler` terminates when relative site tracking drifts too far, with extra root-position and optional root-orientation guards.
- `--mean_site_deviation_threshold`, `--root_deviation_threshold`, `--root_orientation_threshold`, and `--root_site` apply to compatible fullbody terminal handlers.
- `--root_deviation_threshold 0.5` sets that root-position tolerance to 0.5 meters.
"""

import argparse
import os
import sys

import pandas as pd  # 🌟 新增：用于导出CSV
import mujoco        # 🌟 新增：用于获取肌肉名称

from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults, apply_terminal_cli_overrides
from musclemimic.algorithms import PPOJax
from musclemimic.runner.eval_utils import (
    add_common_eval_args,
    normalize_eval_args,
    validate_viewer_args,
    setup_headless,
    load_checkpoint,
    apply_temporal_params,
    apply_trajectory_selection,
    configure_goal_visualization,
    configure_recording,
    align_agent_state,
    validate_traj_index,
    format_trajectory_listing,
    verify_env_dt,
    check_trajectory_sync,
    run_validation_metrics,
    run_validation_metrics_mjx_all,
    run_validation_metrics_mujoco,
    run_with_mujoco_viewer,
    run_with_trajectory_export,
)
from loco_mujoco.task_factories import TaskFactory

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "
from jax import config as jax_config
jax_config.update("jax_default_matmul_precision", "high")

import jax
import jax.numpy as jnp
import numpy as np


# ================= 1. 柔顺度评估函数 (全维度版) =================
def evaluate_compliance(trajectory, dt=0.01):
    """
    计算轨迹的多阶导数能量，返回每一个维度的柔顺度指标
    """
    if len(trajectory) < 4:
        return None, None, None, None, None
        
    # 一阶差分：变化率 (Velocity)
    vel = np.diff(trajectory, n=1, axis=0) / dt
    # 二阶差分：加速度 (Acceleration)
    accel = np.diff(trajectory, n=2, axis=0) / (dt**2)
    # 三阶差分：急动度 (Jerk)
    jerk = np.diff(trajectory, n=3, axis=0) / (dt**3)

    metrics = {
        "rate_energy": np.mean(np.square(vel), axis=0),
        "accel_energy": np.mean(np.square(accel), axis=0),
        "jerk_energy": np.mean(np.square(jerk), axis=0),
        "max_jerk_peak": np.max(np.abs(jerk), axis=0),
        "mean_absolute_jerk": np.mean(np.abs(jerk), axis=0) 
    }
    
    global_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
    return metrics, global_metrics, vel, accel, jerk
# ================================================================


def run_comprehensive_eval(env, agent_conf, agent_state, args, dt, eval_seed=0):
    """
    运行综合评测套件：
    1. 测试成功率、平均肌肉激活度、抗干扰鲁棒性、平滑度/Jerk
    2. 针对 Run 和 Stairs 任务记录：最大攀爬高度、峰值地反力、运动学跟踪误差
    3. 支持导出完整跟踪误差 CSV：
       Time_Step, err_root_xyz, err_root_yaw, err_joint_pos, err_joint_vel, err_site_abs, err_rpos
    """
    print(f"\n{'='*60}")
    print(f"🚀 启动综合评测套件 (Comprehensive Evaluation Suite)")
    print(f"   测试回合数 (Episodes) : {args.test_episodes}")
    print(f"   仿真控制步长 (dt)     : {dt:.4f}s")
    print(f"   抗干扰噪声 (Noise)    : 动力学缩放 {args.noise_scale*100}%")
    print(f"{'='*60}\n")

    # 1. 准备策略函数
    train_state = agent_state.train_state
    if agent_conf.config.experiment.n_seeds > 1:
        seed_idx = args.train_state_seed if args.train_state_seed is not None else 0
        train_state = jax.tree.map(lambda x: x[seed_idx], train_state)

    def sample_actions(ts, obs, _rng):
        obs_b = jnp.atleast_2d(obs) if hasattr(obs, "ndim") and obs.ndim == 1 else obs
        vars_in = {"params": ts.params, "run_stats": ts.run_stats}
        y, updates = agent_conf.network.apply(vars_in, obs_b, mutable=["run_stats"])
        pi, _ = y
        ts_out = ts.replace(run_stats=updates["run_stats"])
        a = pi.mode() if not args.stochastic else pi.sample(seed=_rng)

        if hasattr(a, "ndim") and a.ndim > 1 and a.shape[0] == 1:
            a = a[0]

        return a, ts_out

    policy_fn = jax.jit(sample_actions)
    rng = jax.random.key(eval_seed)

    # 2. 获取底层 MuJoCo model 和 data
    model = None
    data = None
    curr_env = env

    while True:
        if hasattr(curr_env, "model") and hasattr(curr_env, "data"):
            model = curr_env.model
            data = curr_env.data
            break
        elif hasattr(curr_env, "env"):
            curr_env = curr_env.env
        elif hasattr(curr_env, "unwrapped"):
            curr_env = curr_env.unwrapped
        else:
            break

    if model is None:
        raise RuntimeError(
            "无法从环境中获取 MuJoCo Model，这可能是由于环境被深度封装。"
            "请检查环境类中是否包含 .model 属性。"
        )

    # 计算全身总重，用于计算 GRF / body weight
    body_weight = np.sum(model.body_mass) * 9.81

    # 获取所有肌肉 / actuator 名称
    actuator_names = []
    for i in range(model.na):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        actuator_names.append(name if name else f"actuator_{i}")

    # 缓存原始动力学参数
    original_gains = np.array(model.actuator_gainprm).copy()

    # 3. 总体指标记录器
    success_flags = []
    activation_efforts = []
    survival_steps = []
    episode_rewards = []

    act_mean_jerk_list = []
    jnt_mean_jerk_list = []

    # 任务特化指标
    max_z_list = []
    peak_grf_list = []
    tracking_err_list = []

    emg_csv_saved = False

    # 统一定义需要导出的 tracking error 字段
    tracking_error_keys = [
        "err_root_xyz",
        "err_root_yaw",
        "err_joint_pos",
        "err_joint_vel",
        "err_site_abs",
        "err_rpos",
    ]

    def scalar_mean_abs(x):
        """
        将 info 里的误差项安全转成一个标量：
        - 如果是数组，就取 mean(abs(x))
        - 如果是标量，就取 abs(x)
        - 如果转换失败，返回 nan
        """
        try:
            arr = np.asarray(x)
            return float(np.mean(np.abs(arr)))
        except Exception:
            return float("nan")

    for ep in range(args.test_episodes):
        obs = env.reset()

        ep_actions = []
        ep_qvels = []
        ep_activations = []

        # 每个 episode 的逐步跟踪误差记录
        ep_tracking_error_records = []

        # 回合级任务指标
        ep_max_z = -float("inf")
        ep_peak_grf = 0.0
        ep_step_tracking_errs = []

        # 注入动力学噪声
        if args.noise_scale > 0.0:
            variance = np.random.uniform(
                low=1.0 - args.noise_scale,
                high=1.0 + args.noise_scale,
                size=model.na,
            )
            for i in range(model.na):
                model.actuator_gainprm[i, 0] = original_gains[i, 0] * variance[i]

        ep_reward = 0.0
        ep_steps = 0
        ep_activation_sum = 0.0

        while True:
            rng, _rng = jax.random.split(rng)
            action, train_state = policy_fn(train_state, obs, _rng)

            action_tensor = jnp.atleast_2d(action)
            obs, reward, absorbing, done, info = env.step(action_tensor)

            ep_actions.append(np.array(action).flatten())

            if data is not None and hasattr(data, "qvel"):
                ep_qvels.append(np.array(data.qvel).copy())

            # 记录肌肉激活度
            if data is not None and hasattr(data, "act") and data.act is not None:
                current_act = np.array(data.act).copy()
            elif data is not None and hasattr(data, "ctrl"):
                current_act = np.array(data.ctrl).copy()
            else:
                current_act = np.array(action).flatten()

            if len(current_act) == len(actuator_names):
                ep_activations.append(current_act)
            else:
                ep_activations.append(np.array(action).flatten())

            ep_activation_sum += np.mean(np.abs(current_act))
            ep_reward += float(reward)
            ep_steps += 1

            # =========================
            # 任务特化指标：高度、GRF
            # =========================
            if data is not None:
                # 1. 最大 root z 高度
                if data.qpos is not None and len(data.qpos) >= 3:
                    ep_max_z = max(ep_max_z, float(data.qpos[2]))

                # 2. 峰值地反力 GRF，单位为 x body weight
                if body_weight > 0:
                    total_fz = 0.0

                    for i in range(data.ncon):
                        contact = data.contact[i]
                        force = np.zeros(6, dtype=np.float64)
                        mujoco.mj_contactForce(model, data, i, force)

                        normal_z = contact.frame[2]
                        if normal_z > 0.5:
                            total_fz += force[0] * normal_z

                    grf_bw = total_fz / body_weight
                    ep_peak_grf = max(ep_peak_grf, float(grf_bw))

            # ==========================================================
            # 逐步记录完整运动学跟踪误差，用于导出 CSV
            # CSV 列：
            # Time_Step, err_root_xyz, err_root_yaw, err_joint_pos,
            # err_joint_vel, err_site_abs, err_rpos
            # ==========================================================
            step_error_record = {
                "Time_Step": ep_steps - 1,
            }

            for key in tracking_error_keys:
                if key in info:
                    step_error_record[key] = scalar_mean_abs(info[key])
                else:
                    step_error_record[key] = np.nan

            ep_tracking_error_records.append(step_error_record)

            # 保留原来的平均 joint position tracking error 统计逻辑
            if "err_joint_pos" in info:
                ep_step_tracking_errs.append(scalar_mean_abs(info["err_joint_pos"]))

            # ==========================================================
            # 终止逻辑
            # 如果你已经通过 --no_termination 在外部成功关闭了终止，
            # 这里 done 会自然等到轨迹结束才触发。
            # ==========================================================
            if done:
                traj_len = int(info.get("traj_len", 1))
                subtraj_step = int(info.get("subtraj_step_no", 0))

                is_success = subtraj_step >= traj_len - 1

                success_flags.append(is_success)
                activation_efforts.append(ep_activation_sum / max(1, ep_steps))
                survival_steps.append(ep_steps)
                episode_rewards.append(ep_reward)

                max_z_list.append(ep_max_z)
                peak_grf_list.append(ep_peak_grf)

                if ep_step_tracking_errs:
                    tracking_err_list.append(np.mean(ep_step_tracking_errs))

                status_str = "✅ 成功" if is_success else f"❌ 跌倒 ({subtraj_step}/{traj_len})"
                print(
                    f"回合 {ep+1:<3} | "
                    f"存活: {ep_steps:<4} | "
                    f"奖励: {ep_reward:<8.1f} | "
                    f"高度: {ep_max_z:<5.2f}m | "
                    f"峰值GRF: {ep_peak_grf:<4.1f}x | "
                    f"{status_str}"
                )

                # =========================
                # 导出肌肉激活度 CSV
                # =========================
                if args.export_emg_csv and is_success and not emg_csv_saved:
                    df = pd.DataFrame(ep_activations, columns=actuator_names)
                    df.insert(0, "Time_Step", range(len(df)))

                    csv_filename = "EMG_activation_trajectory.csv"
                    df.to_csv(csv_filename, index=False)

                    print(
                        f"💾 成功导出首个成功回合的肌肉激活度轨迹至 {csv_filename} "
                        f"(包含 {len(actuator_names)} 块肌肉)"
                    )
                    emg_csv_saved = True

                # =========================
                # 导出运动学跟踪误差 CSV
                # =========================
                if getattr(args, "export_tracking_csv", False) and ep_tracking_error_records:
                    df_err = pd.DataFrame(
                        ep_tracking_error_records,
                        columns=[
                            "Time_Step",
                            "err_root_xyz",
                            "err_root_yaw",
                            "err_joint_pos",
                            "err_joint_vel",
                            "err_site_abs",
                            "err_rpos",
                        ],
                    )

                    if args.test_episodes == 1:
                        csv_filename = "Kinematic_Tracking_Error.csv"
                    else:
                        csv_filename = f"Kinematic_Tracking_Error_episode_{ep+1}.csv"

                    df_err.to_csv(csv_filename, index=False)
                    print(f"💾 成功导出运动学跟踪误差 CSV: {csv_filename}")

                break

        # =========================
        # 计算 action jerk
        # =========================
        if len(ep_actions) >= 4:
            _, act_g, _, _, _ = evaluate_compliance(np.array(ep_actions), dt)
            if act_g:
                act_mean_jerk_list.append(act_g["mean_absolute_jerk"])

        # =========================
        # 计算 joint jerk
        # =========================
        if len(ep_qvels) >= 4:
            _, jnt_g, _, j_a, _ = evaluate_compliance(np.array(ep_qvels), dt)
            if jnt_g:
                true_joint_jerk_mean = np.mean(np.abs(j_a))
                jnt_mean_jerk_list.append(true_joint_jerk_mean)

    # 还原动力学参数
    model.actuator_gainprm[:] = original_gains

    # 4. 汇总统计
    total_success = sum(success_flags)
    success_rate = (total_success / args.test_episodes) * 100
    avg_steps = np.mean(survival_steps) if survival_steps else 0
    avg_activation = np.mean(activation_efforts) if activation_efforts else 0
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0

    avg_act_jerk = np.mean(act_mean_jerk_list) if act_mean_jerk_list else float("nan")
    avg_jnt_jerk = np.mean(jnt_mean_jerk_list) if jnt_mean_jerk_list else float("nan")

    avg_max_z = (
        np.mean(max_z_list)
        if max_z_list and max_z_list[0] != -float("inf")
        else float("nan")
    )
    avg_peak_grf = np.mean(peak_grf_list) if peak_grf_list else float("nan")
    avg_tracking_err = np.mean(tracking_err_list) if tracking_err_list else float("nan")

    print(f"\n{'='*60}")
    print(f"📊 综合评测最终报告 (Comprehensive Eval Report)")
    print(f"{'='*60}")
    print(f"🏆 成功率 (Success Rate)     : {success_rate:.1f}% ({total_success}/{args.test_episodes})")
    print(f"⏱️ 平均存活步数 (Avg Steps)  : {avg_steps:.1f} 步")
    print(f"🌟 平均奖励 (Avg Reward)     : {avg_reward:.2f}")
    print(f"⚡ 平均肌肉激活度 (Mean Act) : {avg_activation:.4f}")
    print(f"------------------------------------------------------------")
    print(f" 🏔️ 最大攀爬高度 (Stairs Z)   : {avg_max_z:.2f} m 🌟 (填入Table 2)")
    print(f" 💥 峰值地反力 (Run Peak GRF): {avg_peak_grf:.2f} x BW 🌟 (填入Table 3)")
    print(f" 🎯 运动学追踪误差 (Run Err)  : {avg_tracking_err:.4f} rad 🌟 (填入Table 3)")
    print(f"------------------------------------------------------------")
    print(f"[肌肉控制端 - Actions] (无量纲)")
    print(f" 🔥 平均绝对 Jerk            : {avg_act_jerk:>10.4f}")
    print(f"[物理关节端 - Joints Kinematics]")
    print(f" 🔥 真正平均绝对 Jerk        : {avg_jnt_jerk:>10.4f} rad/s^3")
    print(f"{'='*60}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate, visualize, or record a fullbody PPO policy.")
    add_common_eval_args(parser, default_n_envs=10)
    parser.add_argument(
        "--hfield_length",
        type=int,
        default=None,
        help="Override terrain heightfield resolution (default: use training config). Lower values reduce contacts.",
    )
    parser.add_argument(
        "--export_tracking_csv",
        action="store_true",
        help="导出运动学跟踪误差 CSV，包括 root、joint、site 等误差项",
    )
    parser.add_argument(
        "--motion_path",
        type=str,
        nargs="+",
        default=None,
        help="Override motion path for the dataset (default: use training config).",
    )
    parser.add_argument(
        "--motion_group",
        type=str,
        default=None,
        help="Override motion group for the dataset (default: use training config).",
    )
    parser.add_argument(
        "--start_from_beginning",
        default=False,
        action="store_true",
        help="Start evaluation from the beginning of the motion (default: False).",
    )
    parser.add_argument(
        "--evaluate_all",
        default=False,
        action="store_true",
        help="Evaluate all trajectories in the dataset (default: False).",
    )
    parser.add_argument(
        "--no_termination",
        default=False,
        action="store_true",
        help="Disable early termination (run for full n_steps regardless of falls).",
    )
    parser.add_argument(
        "--terminal_state_type",
        type=str,
        default=None,
        help="Override terminal state handler type (e.g., MeanRelativeSiteDeviationWithRootTerminalStateHandler)",
    )
    parser.add_argument(
        "--mean_site_deviation_threshold",
        type=float,
        default=None,
        help="Override mean site deviation threshold for compatible fullbody terminal handlers.",
    )
    parser.add_argument(
        "--root_deviation_threshold",
        type=float,
        default=None,
        help="Override root position deviation threshold for compatible fullbody terminal handlers.",
    )
    parser.add_argument(
        "--root_orientation_threshold",
        type=float,
        default=None,
        help="Override root orientation deviation threshold in radians for compatible fullbody terminal handlers.",
    )
    parser.add_argument(
        "--root_site",
        type=str,
        default=None,
        help="Override the root site name used by compatible fullbody terminal handlers.",
    )
    parser.add_argument(
        "--n_substeps",
        type=int,
        default=None,
        help="Override n_substeps (control frequency). Default 5 = 100Hz, 10 = 50Hz.",
    )

    # ========================== 新增综合评测参数 ==========================
    parser.add_argument(
        "--test_suite",
        action="store_true",
        help="启用综合评测套件 (测试成功率、能耗、抗干扰)",
    )
    parser.add_argument(
        "--test_episodes",
        type=int,
        default=20,
        help="综合评测运行的回合数 (默认: 20)",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.0,
        help="动力学噪声注入强度 (例如 0.2 表示 ±20% 的肌肉力量浮动，默认 0.0 无噪声)",
    )
    parser.add_argument(
        "--export_emg_csv",
        action="store_true",
        help="导出第一个成功回合的肌肉激活度(EMG)时间序列至 CSV 文件",
    )
    # =====================================================================

    args = parser.parse_args()

    try:
        normalize_eval_args(args)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    viewer_err = validate_viewer_args(args)
    if viewer_err:
        print(viewer_err)
        return 1

    is_headless, headless_err = setup_headless(args)
    if headless_err:
        print(headless_err)
        print("   Use --no_render for headless operation")
        return 1

    config, agent_state, _metadata = load_checkpoint(args.path)
    OmegaConf.set_struct(config, False)

    # Restore training configuration
    print("=== RESTORING TRAINING CONFIGURATION ===")
    training_env_params = config.experiment.env_params
    config.experiment.env_params = training_env_params.copy()

    env_name = config.experiment.env_params.get("env_name")
    goal_type = config.experiment.env_params.get("goal_type")
    goal_params = config.experiment.env_params.get("goal_params", {})

    print("Restored training environment configuration:")
    print(f"   Environment: {env_name}")
    print(f"   Goal type: {goal_type}")
    print(f"   Goal params: {goal_params}")

    # Evaluation-specific overrides
    config.experiment.env_params["headless"] = args.no_render
    apply_trajectory_selection(config, args.traj_index, args.traj_start_step)

    # Preserve training temporal parameters
    training_timestep = config.experiment.env_params.get("timestep", 0.002)
    training_n_substeps = config.experiment.env_params.get("n_substeps", 5)
    training_control_dt = apply_temporal_params(config)

    print("\n=== TRAINING TEMPORAL CONFIGURATION ===")
    print(f"   Training timestep: {training_timestep}")
    print(f"   Training n_substeps: {training_n_substeps}")
    print(f"   Training control_dt: {training_control_dt}")

    # Override n_substeps if specified (for testing different control frequencies)
    if args.n_substeps is not None:
        config.experiment.env_params["n_substeps"] = args.n_substeps
        new_control_dt = training_timestep * args.n_substeps
        print(f"\n=== OVERRIDE: n_substeps={args.n_substeps} ===")
        print(f"   New control_dt: {new_control_dt} ({1/new_control_dt:.0f}Hz)")

        # Scale n_step_stride to maintain same lookahead time window
        goal_params = config.experiment.env_params.get("goal_params", {})
        old_stride = goal_params.get("n_step_stride")
        if old_stride is not None:
            # lookahead_time = stride * control_dt (keep constant)
            new_stride = int(round(old_stride * training_control_dt / new_control_dt))
            new_stride = max(1, new_stride)
            config.experiment.env_params["goal_params"]["n_step_stride"] = new_stride
            print(f"   n_step_stride: {old_stride} -> {new_stride} (preserving {old_stride * training_control_dt:.2f}s lookahead)")

    # Fullbody-specific visualization and terminal overrides
    if "MyoFullBody" in env_name:
        print("\nConfiguring MyoFullBody evaluation:")
        configure_goal_visualization(config, args, "GoalTrajMimicv2", is_mjx_env="Mjx" in env_name)
        print(f"   Training goal_type: {goal_type}")
        print(f"   Evaluation goal_type: {config.experiment.env_params.get('goal_type')}")
        print(
            "   Training sites_for_mimic: "
            f"{config.experiment.env_params.get('goal_params', {}).get('sites_for_mimic', 'Not specified')}"
        )
        print(f"   Evaluation headless mode: {args.no_render}")
        print(
            "   Enhanced visualization geometries: "
            f"{config.experiment.env_params.get('goal_params', {}).get('n_visual_geoms', 'Default')}"
        )

    # Terrain override
    if args.hfield_length is not None:
        if "terrain_params" not in config.experiment.env_params:
            config.experiment.env_params["terrain_params"] = {}
        config.experiment.env_params["terrain_params"]["hfield_length"] = args.hfield_length
        print(f"Terrain override: hfield_length={args.hfield_length} (cell size: {8.0/args.hfield_length:.2f}m)")

    # Handle start_from_beginning option: start each episode from initial
    # timestep of a random motion
    if args.start_from_beginning:
        if "th_params" not in config.experiment.env_params:
            config.experiment.env_params.th_params = {}
        config.experiment.env_params.th_params.start_from_random_step = False

    if args.metrics and args.evaluate_all:
        print("\nEvaluating all trajectories in the dataset for validation metrics.")

        if "th_params" not in config.experiment.env_params:
            config.experiment.env_params.th_params = {}

        config.experiment.env_params.th_params.random_start = False
        config.experiment.env_params.th_params.fixed_start_conf = [0, 0]
        config.experiment.env_params.th_params.start_from_random_step = False


    # Override motion path if specified
    if args.motion_path is not None:
        motion_paths = args.motion_path if isinstance(args.motion_path, list) else [args.motion_path]
        config.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path = motion_paths
        config.experiment.task_factory.params.amass_dataset_conf.dataset_group = None
        print(f"Motion path override: {motion_paths}")
    elif args.motion_group is not None:
        config.experiment.task_factory.params.amass_dataset_conf.dataset_group = args.motion_group
        print(f"Motion group override: {args.motion_group}")

    play_env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    # Terminal state configuration priority: CLI > validation config > training config.
    apply_eval_terminal_defaults(play_env_params, config, args.strict_termination)
    try:
        apply_terminal_cli_overrides(play_env_params, args)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    # Final runtime env overrides.
    if args.use_mujoco and "Mjx" in play_env_params.get("env_name", ""):
        play_env_params["env_name"] = play_env_params["env_name"].replace("Mjx", "")
    if not args.use_mujoco:
        play_env_params["num_envs"] = int(args.num_envs)

    # Compute actual control_dt (with override if specified)
    actual_control_dt = training_timestep * (args.n_substeps if args.n_substeps else training_n_substeps)

    if args.record:
        # Extract motion name from dataset config for recording folder name
        motion_paths = config.experiment.task_factory.params.amass_dataset_conf.get("rel_dataset_path", [])
        motion_name = None
        if motion_paths:
            # Handle nested list/ListConfig - flatten to get first string path
            first_path = motion_paths
            while hasattr(first_path, "__iter__") and not isinstance(first_path, str):
                first_path = list(first_path)[0] if first_path else None
            if first_path:
                # Use full path with / replaced by _ (e.g., "KIT/9/walking_run07_poses" -> "KIT_9_walking_run07_poses")
                motion_name = str(first_path).replace("/", "_")
        record_tag = configure_recording(play_env_params, args, actual_control_dt, motion_name)
        print(f"Recording to: {args.record_dir}/{record_tag}/ @ {int(1/actual_control_dt)}fps")

    # Create environment.
    # task_factory.params contributes factory-specific inputs such as dataset selection.
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    merged_params = {**play_env_params, **task_params}
    try:
        env = factory.make(**merged_params)
    except Exception as exc:
        print(f"Error creating environment: {exc!s}")
        print("Exception traceback:")
        import traceback

        traceback.print_exc()
        return 1

    if args.list_trajs:
        for line in format_trajectory_listing(env, args.list_trajs_limit):
            print(line)
        return 0

    validate_traj_index(env, args.traj_index)

    # Build agent configuration and align state for inference
    agent_conf = PPOJax.init_agent_conf(env, config)
    agent_state = align_agent_state(agent_state, agent_conf)

    # Verify env timing and trajectory sync
    verify_env_dt(env, actual_control_dt)
    if "MyoFullBody" in env_name:
        check_trajectory_sync(env)


    # ========================== 综合评测套件拦截 ==========================
    if args.test_suite:
        if not args.use_mujoco:
            print("⚠️ 警告: 动力学噪声注入需要修改底层物理参数。建议添加 --use_mujoco 以获得准确的抗干扰测试结果。")
        
        run_comprehensive_eval(
            env,
            agent_conf,
            agent_state,
            args,
            actual_control_dt,
            eval_seed=args.eval_seed,
        )
        return 0
    # ======================================================================


    # Optional metrics (training-style validation)
    if args.metrics:
        metrics_deterministic = False  # Default to stochastic for fullbody eval
        if args.metrics_deterministic:
            metrics_deterministic = True
        elif args.metrics_stochastic:
            metrics_deterministic = False

        if args.use_mujoco:
            # Use MuJoCo CPU-compatible metrics collection (uses the already created env)
            print("Note: Using MuJoCo (CPU) environment for validation metrics")
            metrics_dict = run_validation_metrics_mujoco(
                env,
                agent_conf,
                agent_state,
                num_steps=args.metrics_steps or 500,
                deterministic=metrics_deterministic,
                train_state_seed=args.train_state_seed,
                evaluate_all=args.evaluate_all,
                eval_seed=args.eval_seed,
            )
        elif args.evaluate_all:
            # MJX GPU evaluate_all: per-trajectory evaluation without AutoResetWrapper
            metrics_dict = run_validation_metrics_mjx_all(
                env,
                agent_conf,
                agent_state,
                deterministic=metrics_deterministic,
                train_state_seed=args.train_state_seed,
                num_envs=args.metrics_envs,
                eval_seed=args.eval_seed,
            )
        else:
            # Use JAX-based validation (MJX environment)
            _validation_summary, metrics_dict = run_validation_metrics(
                config,
                agent_state,
                train_state_seed=args.train_state_seed,
                num_steps=args.metrics_steps,
                num_envs=args.metrics_envs,
                deterministic_override=metrics_deterministic,
                eval_seed=args.eval_seed,
            )

        print("\n=== VALIDATION METRICS ===")
        for key in sorted(metrics_dict.keys()):
            print(f"{key}: {metrics_dict[key]:.6f}")
        if args.metrics_only:
            return 0

    # Run evaluation
    print(f"\nStarting evaluation for {args.n_steps} steps...")
    if args.record:
        print("Recording enabled")
        if is_headless:
            print("  Recording will use headless EGL rendering (no display window)")
    if args.export_trajectory:
        print(f"Trajectory export enabled - saving to {args.trajectory_dir}")

    try:
        enable_render = not args.no_render or args.record

        if args.export_trajectory:
            print("Running evaluation with trajectory export...")
            if args.no_render:
                print("  Headless mode enabled for trajectory export")
            trajectory_filepath, trajectory_data = run_with_trajectory_export(
                env,
                agent_conf,
                agent_state,
                n_steps=args.n_steps,
                deterministic=not args.stochastic,
                use_mujoco=args.use_mujoco,
                train_state_seed=args.train_state_seed,
                trajectory_dir=args.trajectory_dir,
                file_prefix="myofullbody_episodes",
            )

        elif args.use_mujoco and args.mujoco_viewer:
            print("Running MuJoCo evaluation with default viewer...")
            run_with_mujoco_viewer(
                env,
                agent_conf,
                agent_state,
                n_steps=args.n_steps,
                deterministic=not args.stochastic,
                train_state_seed=args.train_state_seed,
            )
        elif args.viser_viewer:
            print("Launching Viser web viewer at http://localhost:8080")
            from musclemimic.viewer import ViserViewer

            viewer = ViserViewer(env, agent_conf, agent_state, deterministic=not args.stochastic)
            viewer.run(n_steps=args.n_steps)
        elif args.use_mujoco:
            print("Running MuJoCo evaluation...")
            PPOJax.play_policy_mujoco(
                env,
                agent_conf,
                agent_state,
                deterministic=not args.stochastic,
                n_steps=args.n_steps,
                render=enable_render,
                record=args.record,
                train_state_seed=args.train_state_seed,
            )
        else:
            print("Running MJX evaluation...")
            mjx_play_envs = 1 if args.traj_index is not None else int(args.num_envs)
            use_sequential_mjx = args.traj_index is not None and not args.evaluate_all
            PPOJax.play_policy(
                env,
                agent_conf,
                agent_state,
                deterministic=not args.stochastic,
                n_steps=args.n_steps,
                n_envs=mjx_play_envs,
                render=enable_render,
                record=args.record,
                train_state_seed=args.train_state_seed,
                sequential_mjx=use_sequential_mjx,
            )

        print("\nEvaluation completed successfully!")
        if args.record:
            print(f"Video recording saved to: {args.record_dir}/{record_tag}/")
            if hasattr(env, "video_file_path") and env.video_file_path:
                print(f"Video saved as: {env.video_file_path}")
        if args.export_trajectory:
            print(f"Trajectory data saved to: {trajectory_filepath}")

    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
    except Exception as exc:
        print(f"Error during evaluation: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())