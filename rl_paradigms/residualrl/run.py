import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium
import gymnasium.spaces as spaces
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from loco_mujoco.core.utils.math import calculate_relative_site_quantities
from musclemimic.environments.humanoids.myofullbody import MyoFullBody


# ==================================================
# 0. 默认配置
# ==================================================
DEFAULT_OBS_DIM = 2418
DEFAULT_ACT_DIM = 354

DEFAULT_BASE_MODEL_DIR = os.environ.get("MUSCLEMIMIC_BASE_MODEL_DIR")

DEFAULT_RUN_MOTION_PATH = os.environ.get(
    "MUSCLEMIMIC_RUN_MOTION_PATH",
    str(Path(__file__).resolve().parent / "walking_run04_poses.npz"),
)


def _default_model_path() -> str:
    """Locate the installed MyoFullBody XML model."""
    env_path = os.environ.get("MUSCLEMIMIC_MODEL_PATH")
    if env_path:
        return env_path

    candidates: list[Path] = [
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "musclemimic_models"
        / "model"
        / "body"
        / "myofullbody.xml",
    ]

    spec = importlib.util.find_spec("musclemimic_models")
    if spec is not None:
        if spec.submodule_search_locations:
            for location in spec.submodule_search_locations:
                candidates.append(
                    Path(location) / "model" / "body" / "myofullbody.xml"
                )
        elif spec.origin:
            candidates.append(
                Path(spec.origin).resolve().parent
                / "model"
                / "body"
                / "myofullbody.xml"
            )

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    # Return the most likely path so the following FileNotFoundError contains
    # a concrete path and tells the user what to configure.
    return str(candidates[0])


DEFAULT_MODEL_PATH = _default_model_path()


def _require_existing_path(path: str | os.PathLike[str], name: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"{name} does not exist: {resolved}\n"
            f"Pass an explicit {name.lower().replace(' ', '_')} or set the "
            f"corresponding MUSCLEMIMIC_* environment variable."
        )
    return str(resolved)


# ==================================================
# 1. 全局 JAX 策略加载器
# ==================================================
# 按进程、checkpoint、观测维度和动作维度分别缓存，避免不同环境串用权重。
_JAX_POLICY_CACHE: dict[tuple[int, str, int, int], tuple[Any, Any]] = {}


def get_jax_policy(
    model_dir: str,
    expected_obs_dim: int = DEFAULT_OBS_DIM,
    act_dim: int = DEFAULT_ACT_DIM,
):
    model_dir = str(Path(model_dir).expanduser().resolve())
    cache_key = (os.getpid(), model_dir, int(expected_obs_dim), int(act_dim))

    if cache_key not in _JAX_POLICY_CACHE:
        import jax
        import jax.numpy as jnp
        from omegaconf import OmegaConf

        from musclemimic.algorithms import PPOJax
        from musclemimic.runner.eval_utils import (
            align_agent_state,
            load_checkpoint,
        )

        config, agent_state, _ = load_checkpoint(model_dir)
        OmegaConf.set_struct(config, False)

        class DummyEnv:
            def __init__(self, obs_dim: int, action_dim: int):
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(obs_dim,),
                    dtype=np.float32,
                )
                self.action_space = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(action_dim,),
                    dtype=np.float32,
                )
                self.mdp_info = SimpleNamespace(
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                )
                self.info = self.mdp_info

        dummy_env = DummyEnv(expected_obs_dim, act_dim)
        agent_conf = PPOJax.init_agent_conf(dummy_env, config)
        train_state = align_agent_state(
            agent_state,
            agent_conf,
        ).train_state

        @jax.jit
        def get_action(ts, obs):
            variables = {
                "params": ts.params,
                "run_stats": ts.run_stats,
            }
            distribution, _ = agent_conf.network.apply(
                variables,
                jnp.atleast_2d(obs),
                mutable=["run_stats"],
            )
            # 使用冻结基础策略的确定性均值动作。
            return jnp.squeeze(distribution[0].mean())

        _JAX_POLICY_CACHE[cache_key] = (get_action, train_state)

    return _JAX_POLICY_CACHE[cache_key]


# ==================================================
# 2. 数学工具
# ==================================================
def root_local_velocity(data: mujoco.MjData) -> np.ndarray:
    """Return root linear and angular velocity in the root local frame."""
    qpos = np.asarray(data.qpos)
    qvel = np.asarray(data.qvel)

    if qpos.size < 7 or qvel.size < 6:
        return qvel[:6].astype(np.float32, copy=True)

    # MuJoCo quaternion order: w, x, y, z
    quat_wxyz = qpos[3:7]
    quat_xyzw = np.array(
        [
            quat_wxyz[1],
            quat_wxyz[2],
            quat_wxyz[3],
            quat_wxyz[0],
        ],
        dtype=np.float64,
    )

    norm = np.linalg.norm(quat_xyzw)
    if norm < 1e-8:
        return qvel[:6].astype(np.float32, copy=True)

    rotation = R.from_quat(quat_xyzw / norm)
    linear_local = rotation.inv().apply(qvel[:3])
    angular_local = rotation.inv().apply(qvel[3:6])

    return np.concatenate(
        [linear_local, angular_local],
        axis=0,
    ).astype(np.float32)


def _run_tracking_distance(distance: float, flight_phase: bool) -> float:
    return float(distance) * (0.5 if flight_phase else 1.0)


def _run_propulsive_reward(
    current_forward_velocity: float,
    reference_forward_velocity: float,
    flight_phase: bool,
    weight: float = 0.3,
) -> float:
    if not flight_phase:
        return 0.0
    return float(
        weight
        * max(
            0.0,
            current_forward_velocity - reference_forward_velocity,
        )
    )


def _run_impact_penalty(
    vertical_grf_body_weights: float,
    limit_body_weights: float = 2.5,
    weight: float = 0.1,
) -> float:
    return float(
        -weight
        * max(
            0.0,
            vertical_grf_body_weights - limit_body_weights,
        )
    )


# ==================================================
# 3. NPZ 轨迹适配器
# 结构直接仿照可运行的 stair 环境：
# - init_state 返回简单轨迹状态
# - reset_state 不修改 frozen carry
# - update_state 仅维护内部步数
# - 参考帧由 data.time / dt 直接索引
# ==================================================
class NPZTrajectoryAdapter:
    def __init__(
        self,
        npz_path: str,
        model: mujoco.MjModel,
        dt: float = 0.01,
    ):
        npz_path = _require_existing_path(npz_path, "Motion path")

        trajectory = np.load(npz_path, allow_pickle=True)

        if "qpos" not in trajectory:
            raise KeyError(
                f"{npz_path} does not contain 'qpos'. "
                f"Available keys: {list(trajectory.files)}"
            )
        if "qvel" not in trajectory:
            raise KeyError(
                f"{npz_path} does not contain 'qvel'. "
                f"Available keys: {list(trajectory.files)}"
            )

        self.qpos_data = np.asarray(
            trajectory["qpos"],
            dtype=np.float64,
        ).copy()
        self.qvel_data = np.asarray(
            trajectory["qvel"],
            dtype=np.float64,
        ).copy()

        if "frequency" in trajectory:
            frequency = float(
                np.asarray(trajectory["frequency"]).item()
            )
            if not np.isfinite(frequency) or frequency <= 0.0:
                raise ValueError(
                    f"Invalid trajectory frequency: {frequency}"
                )
            self.frequency = frequency
            self.dt = 1.0 / frequency
        else:
            self.dt = float(dt)
            self.frequency = 1.0 / self.dt

        self.n_frames = int(self.qpos_data.shape[0])
        self.model = model
        self.is_numpy = True
        self.internal_step = 0

        if self.qpos_data.shape != (
            self.n_frames,
            self.model.nq,
        ):
            raise ValueError(
                "Trajectory/model qpos mismatch: "
                f"trajectory={self.qpos_data.shape}, "
                f"expected=({self.n_frames}, {self.model.nq})."
            )

        if self.qvel_data.shape != (
            self.n_frames,
            self.model.nv,
        ):
            raise ValueError(
                "Trajectory/model qvel mismatch: "
                f"trajectory={self.qvel_data.shape}, "
                f"expected=({self.n_frames}, {self.model.nv})."
            )

        if not np.all(np.isfinite(self.qpos_data)):
            raise ValueError("qpos contains NaN or Inf.")
        if not np.all(np.isfinite(self.qvel_data)):
            raise ValueError("qvel contains NaN or Inf.")

        # 兼容旧代码中对这些字段的访问。
        self.root_lin_vel = self.qvel_data[:, 0:3]
        self.root_ang_vel = self.qvel_data[:, 3:6]
        self.joint_vel = self.qvel_data[:, 6:]

        # 分开保存当前参考和初始参考，避免同一个 MjData 被覆盖。
        self._ref_data = mujoco.MjData(model)
        self._init_ref_data = mujoco.MjData(model)

        print(
            "[NPZTrajectoryAdapter] Loaded run trajectory\n"
            f"  path: {npz_path}\n"
            f"  qpos: {self.qpos_data.shape}\n"
            f"  qvel: {self.qvel_data.shape}\n"
            f"  frequency: {self.frequency:.6f} Hz\n"
            f"  dt: {self.dt:.8f} s"
        )

    def init_state(
        self,
        env,
        key,
        model,
        data,
        backend,
    ):
        del env, key, model, data, backend

        self.internal_step = 0

        # 与可运行的 stair 代码保持相同思路。
        # 增加 subtraj_step_no_init 仅用于兼容 MuscleMimic 的接口。
        return SimpleNamespace(
            traj_no=0,
            subtraj_step_no=0,
            subtraj_step_no_init=0,
        )

    def reset_state(
        self,
        env,
        model,
        data,
        carry,
        backend,
    ):
        del env, model, backend

        # 不对 frozen LocoCarry 或 traj_state 原地赋值。
        self.internal_step = 0
        return data, carry

    def update_state(
        self,
        env,
        model,
        data,
        carry,
        backend,
    ):
        del env, model, data, backend

        # 与 stair 一样，仅维护本地计数。
        # 实际参考帧统一由仿真时间计算。
        self.internal_step += 1
        return carry

    def len_trajectory(self, traj_no: int = 0) -> int:
        if int(traj_no) != 0:
            raise IndexError(
                f"This adapter contains one trajectory; got traj_no={traj_no}."
            )
        return self.n_frames

    def _clip_step(self, step: int) -> int:
        return int(
            np.clip(
                int(step),
                0,
                self.n_frames - 1,
            )
        )

    def _fill_reference_data(
        self,
        target: mujoco.MjData,
        step: int,
    ) -> mujoco.MjData:
        step = self._clip_step(step)

        target.time = float(step) * self.dt
        target.qpos[:] = self.qpos_data[step]
        target.qvel[:] = self.qvel_data[step]

        if target.act.size > 0:
            target.act[:] = 0.0
        if target.ctrl.size > 0:
            target.ctrl[:] = 0.0

        mujoco.mj_forward(
            self.model,
            target,
        )
        return target

    def get_traj_data_at(
        self,
        traj_no,
        step,
        carry=None,
        backend=None,
    ):
        del traj_no, carry, backend
        return self._fill_reference_data(
            self._ref_data,
            step,
        )

    def get_current_traj_data(
        self,
        carry,
        backend,
    ):
        del backend

        traj_state = getattr(carry, "traj_state", None)
        step = (
            getattr(traj_state, "subtraj_step_no", self.internal_step)
            if traj_state is not None
            else self.internal_step
        )
        return self._fill_reference_data(
            self._ref_data,
            int(np.asarray(step).item()),
        )

    def get_init_traj_data(
        self,
        carry,
        backend,
    ):
        del carry, backend
        return self._fill_reference_data(
            self._init_ref_data,
            0,
        )


# ==================================================
# 4. Full-body goal wrapper
# ==================================================
class FullBodyGoalWrapper:
    SITES_FOR_MIMIC = [
        "pelvis_mimic",
        "upper_body_mimic",
        "head_mimic",
        "left_shoulder_mimic",
        "left_elbow_mimic",
        "left_hand_mimic",
        "right_shoulder_mimic",
        "right_elbow_mimic",
        "right_hand_mimic",
        "left_hip_mimic",
        "left_knee_mimic",
        "left_ankle_mimic",
        "left_toes_mimic",
        "right_hip_mimic",
        "right_knee_mimic",
        "right_ankle_mimic",
        "right_toes_mimic",
    ]

    N_STEP_LOOKAHEAD = 5
    N_STEP_STRIDE = 20
    GOAL_DIM = 469

    def __init__(self, env):
        self.env = env
        self._initialized = False

    def _lazy_init(self) -> None:
        if self._initialized:
            return

        model = self.env._model
        self._ref_data = mujoco.MjData(model)

        site_ids: list[int] = []
        missing_sites: list[str] = []

        for site_name in self.SITES_FOR_MIMIC:
            site_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                site_name,
            )
            if site_id == -1:
                missing_sites.append(site_name)
            else:
                site_ids.append(int(site_id))

        if missing_sites:
            raise ValueError(
                "The MuJoCo model is missing required mimic sites: "
                + ", ".join(missing_sites)
            )

        self._rel_site_ids = np.asarray(
            site_ids,
            dtype=np.int32,
        )
        self._body_rootid = model.body_rootid
        self._site_bodyid = model.site_bodyid
        self._rel_body_ids = self._site_bodyid[self._rel_site_ids]

        root_joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "root",
        )
        if root_joint_id == -1:
            raise ValueError("The MuJoCo model has no joint named 'root'.")

        qpos_adr = int(model.jnt_qposadr[root_joint_id])
        qvel_adr = int(model.jnt_dofadr[root_joint_id])

        self._root_qpos_ind = np.asarray(
            [qpos_adr, qpos_adr + 1, qpos_adr + 2],
            dtype=np.int32,
        )
        self._root_qvel_ind = np.arange(
            qvel_adr,
            qvel_adr + 6,
            dtype=np.int32,
        )

        self._initialized = True

    def get_goal_obs(self) -> np.ndarray:
        self._lazy_init()

        data = self.env._data
        current_step = self.env.ref_step
        traj_state = SimpleNamespace(
            traj_no=0,
            subtraj_step_no=current_step,
        )

        site_rpos, site_rangles, site_rvel = (
            calculate_relative_site_quantities(
                data,
                self._rel_site_ids,
                self._rel_body_ids,
                self._body_rootid,
                np,
            )
        )

        current_rpos = np.ravel(site_rpos).astype(np.float32)
        current_rangles = np.ravel(site_rangles).astype(np.float32)
        current_rvel = np.ravel(site_rvel).astype(np.float32)

        trajectory_goal_obs = self._build_concise_traj_goal(
            traj_state
        )

        trajectory_length = self.env.th.len_trajectory(
            traj_state.traj_no
        )
        motion_phase = np.asarray(
            [
                float(traj_state.subtraj_step_no)
                / max(float(trajectory_length - 1), 1.0)
            ],
            dtype=np.float32,
        )

        goal_obs = np.concatenate(
            [
                current_rpos,
                current_rangles,
                current_rvel,
                trajectory_goal_obs,
                motion_phase,
            ],
            axis=0,
        )

        if goal_obs.size != self.GOAL_DIM:
            raise ValueError(
                "Goal observation dimension mismatch: "
                f"expected {self.GOAL_DIM}, got {goal_obs.size}."
            )

        return goal_obs

    def _build_concise_traj_goal(
        self,
        traj_state: SimpleNamespace,
    ) -> np.ndarray:
        trajectory_length = self.env.th.len_trajectory(
            traj_state.traj_no
        )

        reference_data = self.env.th.get_traj_data_at(
            traj_state.traj_no,
            traj_state.subtraj_step_no,
        )

        reference_root_position = reference_data.qpos[
            self._root_qpos_ind
        ].copy()
        reference_root_velocity = reference_data.qvel[
            self._root_qvel_ind
        ].copy()

        site_rpos_0, _, _ = calculate_relative_site_quantities(
            reference_data,
            self._rel_site_ids,
            self._rel_body_ids,
            self._body_rootid,
            np,
        )

        all_site_rpos = [
            np.ravel(site_rpos_0).astype(np.float32)
        ]
        all_position_delta: list[np.ndarray] = []
        all_velocity_delta: list[np.ndarray] = []

        for lookahead_index in range(1, self.N_STEP_LOOKAHEAD):
            future_step = int(
                np.clip(
                    traj_state.subtraj_step_no
                    + lookahead_index * self.N_STEP_STRIDE,
                    0,
                    trajectory_length - 1,
                )
            )

            future_data = self.env.th.get_traj_data_at(
                traj_state.traj_no,
                future_step,
            )

            future_site_rpos, _, _ = (
                calculate_relative_site_quantities(
                    future_data,
                    self._rel_site_ids,
                    self._rel_body_ids,
                    self._body_rootid,
                    np,
                )
            )

            all_site_rpos.append(
                np.ravel(future_site_rpos).astype(np.float32)
            )
            all_position_delta.append(
                (
                    future_data.qpos[self._root_qpos_ind]
                    - reference_root_position
                ).astype(np.float32)
            )
            all_velocity_delta.append(
                (
                    future_data.qvel[self._root_qvel_ind]
                    - reference_root_velocity
                ).astype(np.float32)
            )

        components: list[np.ndarray] = [all_site_rpos[0]]

        for index in range(len(all_position_delta)):
            components.extend(
                [
                    all_position_delta[index],
                    all_velocity_delta[index],
                    all_site_rpos[index + 1],
                ]
            )

        return np.concatenate(components, axis=0)


# ==================================================
# 5. Walking core environment
# 完全沿用 stair 环境的简单执行结构：
# - reset 后直接写入参考轨迹第一帧
# - step 只执行 MyoFullBody.step
# - 参考轨迹结束时 truncated
# ==================================================
class MSKBenchResidualRunEnvV0(gymnasium.Env, MyoFullBody):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    class _SimCompat:
        def __init__(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
        ):
            self.model = model
            self.data = data

    @property
    def sim(self):
        return self._SimCompat(
            self._model,
            self._data,
        )

    @property
    def ref_step(self) -> int:
        return int(
            np.clip(
                int(float(self._data.time) / self.th.dt),
                0,
                self.th.n_frames - 1,
            )
        )

    def __init__(
        self,
        model_path: str | None = None,
        motion_path: str | None = None,
        motion_dt: float = 0.01,
        physics_substep_factor: int = 2,
        physics_integrator: str = "implicit",
        solver_iterations: int = 100,
        solver_ls_iterations: int = 50,
        **kwargs,
    ):
        model_path = model_path or DEFAULT_MODEL_PATH
        motion_path = motion_path or DEFAULT_RUN_MOTION_PATH

        model_path = _require_existing_path(
            model_path,
            "Model path",
        )
        motion_path = _require_existing_path(
            motion_path,
            "Motion path",
        )

        # 与 stair 环境相同：不使用官方轨迹 goal，
        # 由下面的 FullBodyGoalWrapper 独立构造 469 维 goal。
        MyoFullBody.__init__(
            self,
            spec=model_path,
            disable_fingers=True,
            enable_muscle_length_observations=True,
            enable_muscle_velocity_observations=True,
            enable_muscle_force_observations=True,
            enable_muscle_excitation_observations=True,
            enable_muscle_activation_observations=True,
            enable_touch_sensor_observations=True,
            **kwargs,
        )

        # ----------------------------------------------------------
        # MuJoCo 数值稳定配置
        #
        # 减小内部物理 timestep，同时按相同比例增大 n_substeps，
        # 因此 agent 的控制周期保持不变：
        #
        # old_dt * old_n_substeps
        #     == new_dt * new_n_substeps
        #
        # residual_scale 仍然可以保持 0.6。
        # ----------------------------------------------------------
        physics_substep_factor = max(
            int(physics_substep_factor),
            1,
        )

        original_physics_timestep = float(
            self._model.opt.timestep
        )
        original_n_substeps = int(
            self._n_substeps
        )
        original_control_dt = (
            original_physics_timestep
            * original_n_substeps
        )

        self._model.opt.timestep = (
            original_physics_timestep
            / physics_substep_factor
        )
        self._n_substeps = (
            original_n_substeps
            * physics_substep_factor
        )

        integrator_name = str(
            physics_integrator
        ).strip().lower()
        integrator_map = {
            "euler": mujoco.mjtIntegrator.mjINT_EULER,
            "rk4": mujoco.mjtIntegrator.mjINT_RK4,
            "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
            "implicitfast": (
                mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            ),
        }
        if integrator_name not in integrator_map:
            raise ValueError(
                "physics_integrator must be one of "
                f"{tuple(integrator_map)}, got "
                f"{physics_integrator!r}."
            )

        self._model.opt.integrator = integrator_map[
            integrator_name
        ]

        # Newton + elliptic cone generally gives a more accurate contact
        # solve than the cheaper alternatives.
        self._model.opt.solver = (
            mujoco.mjtSolver.mjSOL_NEWTON
        )
        self._model.opt.cone = (
            mujoco.mjtCone.mjCONE_ELLIPTIC
        )
        self._model.opt.iterations = max(
            int(solver_iterations),
            1,
        )
        self._model.opt.ls_iterations = max(
            int(solver_ls_iterations),
            1,
        )
        self._model.opt.tolerance = 1e-10
        self._model.opt.ls_tolerance = 1e-4

        new_control_dt = (
            float(self._model.opt.timestep)
            * int(self._n_substeps)
        )
        if not np.isclose(
            new_control_dt,
            original_control_dt,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Internal timestep refinement changed the "
                "agent control period: "
                f"old={original_control_dt}, "
                f"new={new_control_dt}."
            )

        self.physics_substep_factor = (
            physics_substep_factor
        )
        self.physics_control_dt = (
            new_control_dt
        )

        print(
            "[MuJoCo stability]\n"
            f"  integrator: {integrator_name}\n"
            f"  physics timestep: "
            f"{self._model.opt.timestep:.8f} s\n"
            f"  n_substeps: {self._n_substeps}\n"
            f"  control dt: {new_control_dt:.8f} s\n"
            f"  solver iterations: "
            f"{self._model.opt.iterations}\n"
            f"  line-search iterations: "
            f"{self._model.opt.ls_iterations}"
        )

        self.th = NPZTrajectoryAdapter(
            motion_path,
            self._model,
            dt=motion_dt,
        )
        self._goal_wrapper = FullBodyGoalWrapper(self)

        self.horizon = self.th.n_frames
        self._max_episode_steps = self.th.n_frames

        action_dim = int(self._model.nu)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(DEFAULT_OBS_DIM,),
            dtype=np.float32,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
        **kwargs,
    ):
        del options

        if seed is not None:
            np.random.seed(seed)

        kwargs.pop("seed", None)
        kwargs.pop("options", None)

        MyoFullBody.reset(
            self,
            **kwargs,
        )

        self._data.time = 0.0

        # 与 stair 一样，直接从轨迹首帧设置完整状态。
        reference_qpos = self.th.qpos_data[0]
        reference_qvel = self.th.qvel_data[0]

        self._data.qpos[:] = reference_qpos
        self._data.qvel[:] = reference_qvel

        if self._data.act.size > 0:
            self._data.act[:] = 0.0
        if self._data.ctrl.size > 0:
            self._data.ctrl[:] = 0.0

        mujoco.mj_forward(
            self._model,
            self._data,
        )

        return self._get_full_obs(), {}

    def step(self, action):
        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(self.action_space.shape)

        # 与可运行 stair 环境保持一致：直接调用，不接管其内部奖励。
        MyoFullBody.step(
            self,
            action,
        )

        truncated = bool(
            self._data.time
            >= (self.th.n_frames * self.th.dt) - 0.02
        )
        terminated = False

        return (
            self._get_full_obs(),
            0.0,
            terminated,
            truncated,
            {},
        )

    def _get_full_obs(self) -> np.ndarray:
        carry = getattr(
            self,
            "_additional_carry",
            getattr(self, "_carry", None),
        )

        observation = self._create_observation(
            self._model,
            self._data,
            carry,
        )

        if isinstance(observation, tuple):
            observation = observation[0]

        base_obs = np.asarray(
            observation,
            dtype=np.float32,
        ).ravel()
        goal_obs = self._goal_wrapper.get_goal_obs()

        full_obs = np.concatenate(
            [base_obs, goal_obs],
            axis=0,
        ).astype(np.float32)

        if full_obs.size != DEFAULT_OBS_DIM:
            raise ValueError(
                "Full observation dimension mismatch: "
                f"expected {DEFAULT_OBS_DIM}, got {full_obs.size}. "
                f"base={base_obs.size}, goal={goal_obs.size}."
            )

        return full_obs


# ==================================================
# 6. Walking residual wrapper
# 奖励函数与上一版 walk 代码保持一致
# ==================================================
class MSKBenchResidualRunWrapper(gymnasium.Wrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        base_model_dir: str | None,
        residual_scale: float = 0.6,
        propulsive_velocity_weight: float = 0.3,
        grf_penalty_weight: float = 0.1,
        grf_limit_body_weights: float = 2.5,
    ):
        super().__init__(env)

        self.base_model_dir = (
            _require_existing_path(base_model_dir, "Base model dir")
            if base_model_dir
            else None
        )
        self.residual_scale = float(residual_scale)
        self.expected_obs_dim = DEFAULT_OBS_DIM

        self.action_space = env.action_space
        self._action_shape = self.action_space.shape
        self._action_dim = int(np.prod(self._action_shape))

        if self._action_dim != int(self.env.unwrapped._model.nu):
            raise ValueError(
                "Action dimension mismatch: "
                f"space={self._action_dim}, model.nu="
                f"{self.env.unwrapped._model.nu}."
            )

        if self.base_model_dir is None:
            self._base_policy_fn = None
            self._base_train_state = None
        else:
            self._base_policy_fn, self._base_train_state = get_jax_policy(
                self.base_model_dir,
                expected_obs_dim=self.expected_obs_dim,
                act_dim=self._action_dim,
            )

        self._prev_blended_action = np.zeros(
            self._action_shape,
            dtype=np.float32,
        )
        # ------------------------------------------------------
        # QACC safety shield
        #
        # residual_scale 保持 0.6。这里限制的是每个控制周期内
        # 残差分量的变化速度，而不是允许的最终幅值。
        # 因此残差仍可逐步达到 [-0.6, 0.6]。
        # ------------------------------------------------------
        self._applied_scaled_residual = np.zeros(
            self._action_shape,
            dtype=np.float32,
        )
        self._episode_step = 0

        # 每个 10 ms 控制周期，单维残差最多变化 0.06。
        # 从 0 达到 0.6 约需 10 个控制步。
        self.max_scaled_residual_delta = 0.06

        # 预测 QACC 安全阈值。实际阈值同时参考基础策略本身
        # 的预测 QACC，避免在正常接触阶段误判。
        self.qacc_absolute_limit = 2.0e6
        self.qacc_hard_limit = 1.0e8
        self.qacc_relative_factor = 20.0

        # 危险时依次减弱“当前这一步”的残差。
        # 这不会更改 residual_scale=0.6，只是安全回退。
        self.qacc_backtrack_factors = (
            1.0,
            0.75,
            0.5,
            0.25,
            0.0,
        )

        self.physics_failure_penalty = -50.0
        self._qacc_probe_data = mujoco.MjData(
            self.env.unwrapped._model
        )

        # 动作叠加照 stair：configured residual_scale 固定为 0.6。
        # QACC shield 只在危险控制步临时回退当前残差。
        self.alpha_stance = self.residual_scale
        self.alpha_swing = self.residual_scale

        self.propulsive_velocity_weight = float(
            propulsive_velocity_weight
        )
        self.grf_penalty_weight = float(grf_penalty_weight)
        self.grf_limit_body_weights = float(
            grf_limit_body_weights
        )
        self._residual_l2_weight = 0.02

        self._qpos_w_exp = 10.0
        self._qvel_w_exp = 2.0
        self._root_pos_w_exp = 10.0
        self._root_vel_w_exp = 10.0
        self._rpos_w_exp = 20.0
        self._rquat_w_exp = 2.0
        self._rvel_w_exp = 0.1

        self._qpos_w_sum = 0.0
        self._qvel_w_sum = 0.0
        self._root_pos_w_sum = 0.0
        self._rpos_w_sum = 0.5
        self._rquat_w_sum = 0.3
        self._rvel_w_sum = 0.0
        self._root_vel_w_sum = 0.2

        model = self.env.unwrapped._model

        (
            self._left_action_mask,
            self._right_action_mask,
        ) = self._infer_action_side_masks(model)

        self._left_foot_body_ids = self._body_ids(
            model,
            (
                "talus_l",
                "calcn_l",
                "foot_l",
                "toes_l",
            ),
        )
        self._right_foot_body_ids = self._body_ids(
            model,
            (
                "talus_r",
                "calcn_r",
                "foot_r",
                "toes_r",
            ),
        )
        self._support_geom_ids = self._infer_support_geom_ids(
            model
        )
        self._foot_body_ids = frozenset(
            self._left_foot_body_ids
            + self._right_foot_body_ids
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self._prev_blended_action = np.zeros(
            self._action_shape,
            dtype=np.float32,
        )
        self._applied_scaled_residual = np.zeros(
            self._action_shape,
            dtype=np.float32,
        )
        self._episode_step = 0

        return obs, info

    def _copy_state_to_probe(self) -> None:
        source_data = self.env.unwrapped._data
        probe_data = self._qacc_probe_data

        probe_data.time = float(source_data.time)
        probe_data.qpos[:] = source_data.qpos
        probe_data.qvel[:] = source_data.qvel

        if probe_data.act.size > 0:
            probe_data.act[:] = source_data.act
        if probe_data.ctrl.size > 0:
            probe_data.ctrl[:] = source_data.ctrl

        if probe_data.qacc_warmstart.size > 0:
            probe_data.qacc_warmstart[:] = (
                source_data.qacc_warmstart
            )
        if probe_data.qfrc_applied.size > 0:
            probe_data.qfrc_applied[:] = (
                source_data.qfrc_applied
            )
        if probe_data.xfrc_applied.size > 0:
            probe_data.xfrc_applied[:] = (
                source_data.xfrc_applied
            )

        if probe_data.mocap_pos.size > 0:
            probe_data.mocap_pos[:] = source_data.mocap_pos
        if probe_data.mocap_quat.size > 0:
            probe_data.mocap_quat[:] = source_data.mocap_quat

    def _normalized_action_to_ctrl(
        self,
        normalized_action: np.ndarray,
    ) -> np.ndarray:
        base_env = self.env.unwrapped
        action = np.asarray(
            normalized_action,
            dtype=np.float64,
        ).reshape(self._action_shape)

        control_function = getattr(
            base_env,
            "_control_func",
            None,
        )

        if (
            control_function is not None
            and hasattr(
                control_function,
                "_unnormalize_action",
            )
        ):
            return np.asarray(
                control_function._unnormalize_action(
                    action
                ),
                dtype=np.float64,
            ).reshape(-1)

        action_indices = np.asarray(
            base_env._action_indices,
            dtype=np.int64,
        ).reshape(-1)
        ctrl_range = np.asarray(
            base_env._model.actuator_ctrlrange[
                action_indices
            ],
            dtype=np.float64,
        )

        low = ctrl_range[:, 0]
        high = ctrl_range[:, 1]
        mean = 0.5 * (low + high)
        delta = 0.5 * (high - low)

        return np.clip(
            mean + action.reshape(-1) * delta,
            low,
            high,
        )

    def _predict_max_abs_qacc(
        self,
        normalized_action: np.ndarray,
    ) -> float:
        base_env = self.env.unwrapped
        model = base_env._model
        probe = self._qacc_probe_data

        self._copy_state_to_probe()

        ctrl_action = self._normalized_action_to_ctrl(
            normalized_action
        )
        action_indices = np.asarray(
            base_env._action_indices,
            dtype=np.int64,
        ).reshape(-1)

        if ctrl_action.size != action_indices.size:
            raise ValueError(
                "Predicted ctrl dimension mismatch: "
                f"ctrl={ctrl_action.size}, "
                f"indices={action_indices.size}."
            )

        probe.ctrl[action_indices] = ctrl_action

        try:
            mujoco.mj_forward(
                model,
                probe,
            )
        except RuntimeError:
            return float("inf")

        qacc = np.asarray(
            probe.qacc,
            dtype=np.float64,
        )

        if not np.all(np.isfinite(qacc)):
            return float("inf")

        return float(
            np.max(np.abs(qacc))
        )

    def _apply_qacc_shield(
        self,
        base_action: np.ndarray,
        target_scaled_residual: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
        float,
        float,
        bool,
    ]:
        # Slew-rate limit: keep the full 0.6 target range, but prevent
        # a one-step jump across a large portion of that range.
        residual_delta = np.clip(
            (
                target_scaled_residual
                - self._applied_scaled_residual
            ),
            -self.max_scaled_residual_delta,
            self.max_scaled_residual_delta,
        )

        rate_limited_residual = (
            self._applied_scaled_residual
            + residual_delta
        ).astype(np.float32)

        base_qacc = self._predict_max_abs_qacc(
            base_action
        )

        if not np.isfinite(base_qacc):
            return (
                np.asarray(base_action, dtype=np.float32),
                np.zeros_like(rate_limited_residual),
                0.0,
                float("inf"),
                float("inf"),
                True,
            )

        allowed_qacc = max(
            float(self.qacc_absolute_limit),
            float(base_qacc)
            * float(self.qacc_relative_factor),
        )

        last_candidate_qacc = float("inf")

        for factor in self.qacc_backtrack_factors:
            candidate_residual = (
                rate_limited_residual
                * float(factor)
            ).astype(np.float32)

            candidate_action = np.clip(
                base_action + candidate_residual,
                -1.0,
                1.0,
            ).astype(np.float32)

            candidate_qacc = self._predict_max_abs_qacc(
                candidate_action
            )
            last_candidate_qacc = candidate_qacc

            safe = (
                np.isfinite(candidate_qacc)
                and candidate_qacc <= allowed_qacc
                and candidate_qacc <= self.qacc_hard_limit
            )

            if safe:
                self._applied_scaled_residual = (
                    candidate_residual.copy()
                )
                return (
                    candidate_action,
                    candidate_residual,
                    float(factor),
                    float(base_qacc),
                    float(candidate_qacc),
                    False,
                )

        return (
            np.asarray(base_action, dtype=np.float32),
            np.zeros_like(rate_limited_residual),
            0.0,
            float(base_qacc),
            float(last_candidate_qacc),
            True,
        )

    def step(self, residual_action):
        current_obs = self.env.unwrapped._get_full_obs()

        if self._base_policy_fn is None:
            base_action = np.zeros(self._action_shape, dtype=np.float32)
        else:
            base_action = np.asarray(
                self._base_policy_fn(
                    self._base_train_state,
                    current_obs,
                ),
                dtype=np.float32,
            ).reshape(self._action_shape)

        if not np.all(np.isfinite(base_action)):
            raise FloatingPointError(
                "Base policy returned NaN or Inf actions."
            )

        residual_action = np.asarray(
            residual_action,
            dtype=np.float32,
        ).reshape(self._action_shape)

        # residual_scale 始终保持 0.6。
        target_scaled_residual = (
            np.clip(
                residual_action,
                -1.0,
                1.0,
            )
            * self.residual_scale
        ).astype(np.float32)

        (
            blended_action,
            safe_residual,
            qacc_shield_factor,
            predicted_base_qacc,
            predicted_candidate_qacc,
            qacc_shield_failed,
        ) = self._apply_qacc_shield(
            base_action,
            target_scaled_residual,
        )

        alpha = np.full(
            self._action_shape,
            self.residual_scale
            * qacc_shield_factor,
            dtype=np.float32,
        )

        effective_scale = float(
            self.residual_scale
            * qacc_shield_factor
        )

        if qacc_shield_failed:
            self._episode_step += 1

            failure_info = {
                "reward_total": float(
                    self.physics_failure_penalty
                ),
                "reward_mimic": 0.0,
                "penalty_residual_l2": 0.0,
                "flight_phase": False,
                "vertical_grf_body_weights": 0.0,
                "propulsive": 0.0,
                "impact": 0.0,
                "alpha_mean": 0.0,
                "configured_residual_scale": float(
                    self.residual_scale
                ),
                "qacc_shield_factor": 0.0,
                "predicted_base_qacc": float(
                    predicted_base_qacc
                ),
                "predicted_candidate_qacc": float(
                    predicted_candidate_qacc
                ),
                "termination_reason": (
                    "preemptive_qacc_guard"
                ),
                "constraint_violation": True,
                "success": False,
                "reward_terms": {
                    "mimic": 0.0,
                    "propulsive": 0.0,
                    "impact": 0.0,
                    "residual_l2": 0.0,
                },
            }
            return (
                np.asarray(
                    current_obs,
                    dtype=np.float32,
                ),
                float(self.physics_failure_penalty),
                True,
                False,
                failure_info,
            )

        # 保存仿真状态。MuJoCo 的 warning callback 会把 QACC 数值异常
        # 转成 RuntimeError；发生时恢复到上一个有效状态并结束当前 episode。
        base_env = self.env.unwrapped
        sim_data = base_env._data
        previous_time = float(sim_data.time)
        previous_qpos = np.asarray(sim_data.qpos).copy()
        previous_qvel = np.asarray(sim_data.qvel).copy()
        previous_act = (
            np.asarray(sim_data.act).copy()
            if sim_data.act.size > 0
            else None
        )
        previous_ctrl = (
            np.asarray(sim_data.ctrl).copy()
            if sim_data.ctrl.size > 0
            else None
        )

        try:
            obs, _, base_terminated, truncated, info = self.env.step(
                blended_action
            )
        except RuntimeError as exc:
            message = str(exc)
            numerical_warning = (
                "Got MuJoCo Warning" in message
                or "QACC" in message
                or "Nan, Inf or huge value" in message
                or "simulation is unstable" in message.lower()
            )

            if not numerical_warning:
                raise

            sim_data.time = previous_time
            sim_data.qpos[:] = previous_qpos
            sim_data.qvel[:] = previous_qvel
            if previous_act is not None:
                sim_data.act[:] = previous_act
            if previous_ctrl is not None:
                sim_data.ctrl[:] = previous_ctrl

            mujoco.mj_forward(
                base_env._model,
                sim_data,
            )

            recovered_obs = np.asarray(
                base_env._get_full_obs(),
                dtype=np.float32,
            )

            self._prev_blended_action = blended_action.copy()
            self._episode_step += 1

            failure_info = {
                "reward_total": float(self.physics_failure_penalty),
                "reward_mimic": 0.0,
                "penalty_residual_l2": 0.0,
                "flight_phase": False,
                "vertical_grf_body_weights": 0.0,
                "propulsive": 0.0,
                "impact": 0.0,
                "alpha_mean": float(effective_scale),
                "configured_residual_scale": float(
                    self.residual_scale
                ),
                "qacc_shield_factor": float(
                    qacc_shield_factor
                ),
                "predicted_base_qacc": float(
                    predicted_base_qacc
                ),
                "predicted_candidate_qacc": float(
                    predicted_candidate_qacc
                ),
                "residual_rms": float(
                    np.sqrt(np.mean(np.square(residual_action)))
                ),
                "blended_action_saturation": float(
                    np.mean(np.abs(blended_action) > 0.999)
                ),
                "termination_reason": "physics_instability",
                "physics_error": message,
                "constraint_violation": True,
                "success": False,
                "reward_terms": {
                    "mimic": 0.0,
                    "propulsive": 0.0,
                    "impact": 0.0,
                    "residual_l2": 0.0,
                },
            }
            return (
                recovered_obs,
                float(self.physics_failure_penalty),
                True,
                False,
                failure_info,
            )

        self._episode_step += 1

        left_foot_contact = self._left_foot_contact()
        right_foot_contact = self._right_foot_contact()
        flight_phase = bool(
            not left_foot_contact
            and not right_foot_contact
        )

        data = self.env.unwrapped._data
        model = self.env.unwrapped._model
        current_step = self.env.unwrapped.ref_step
        ref_data = self.env.unwrapped.th.get_traj_data_at(
            0,
            current_step,
        )

        goal_wrapper = self.env.unwrapped._goal_wrapper
        goal_wrapper._lazy_init()

        site_rpos, site_rangles, site_rvel = (
            calculate_relative_site_quantities(
                data,
                goal_wrapper._rel_site_ids,
                goal_wrapper._rel_body_ids,
                goal_wrapper._body_rootid,
                np,
            )
        )
        ref_site_rpos, ref_site_rangles, ref_site_rvel = (
            calculate_relative_site_quantities(
                ref_data,
                goal_wrapper._rel_site_ids,
                goal_wrapper._rel_body_ids,
                goal_wrapper._body_rootid,
                np,
            )
        )

        rpos_dist = float(
            np.mean(
                np.square(
                    site_rpos - ref_site_rpos
                )
            )
        )
        rangles_dist = float(
            np.mean(
                np.square(
                    site_rangles - ref_site_rangles
                )
            )
        )
        rvel_rot_dist = float(
            np.mean(
                np.square(
                    site_rvel[:, :3]
                    - ref_site_rvel[:, :3]
                )
            )
        )
        rvel_lin_dist = float(
            np.mean(
                np.square(
                    site_rvel[:, 3:]
                    - ref_site_rvel[:, 3:]
                )
            )
        )
        root_vel_dist = float(
            np.mean(
                np.square(
                    root_local_velocity(data)
                    - root_local_velocity(ref_data)
                )
            )
        )
        qpos_dist = float(
            np.mean(
                np.square(
                    data.qpos[7:]
                    - ref_data.qpos[7:]
                )
            )
        )
        qvel_dist = float(
            np.mean(
                np.square(
                    data.qvel[6:]
                    - ref_data.qvel[6:]
                )
            )
        )
        root_pos_dist = float(
            np.mean(
                np.square(
                    data.qpos[:3]
                    - ref_data.qpos[:3]
                )
            )
        )

        rpos_dist = _run_tracking_distance(
            rpos_dist,
            flight_phase,
        )
        rangles_dist = _run_tracking_distance(
            rangles_dist,
            flight_phase,
        )
        rvel_rot_dist = _run_tracking_distance(
            rvel_rot_dist,
            flight_phase,
        )
        rvel_lin_dist = _run_tracking_distance(
            rvel_lin_dist,
            flight_phase,
        )
        root_vel_dist = _run_tracking_distance(
            root_vel_dist,
            flight_phase,
        )
        qpos_dist = _run_tracking_distance(
            qpos_dist,
            flight_phase,
        )
        qvel_dist = _run_tracking_distance(
            qvel_dist,
            flight_phase,
        )
        root_pos_dist = _run_tracking_distance(
            root_pos_dist,
            flight_phase,
        )

        qpos_reward = float(
            np.exp(
                -self._qpos_w_exp * qpos_dist
            )
        )
        qvel_reward = float(
            np.exp(
                -self._qvel_w_exp * qvel_dist
            )
        )
        root_pos_reward = float(
            np.exp(
                -self._root_pos_w_exp * root_pos_dist
            )
        )
        root_vel_reward = float(
            np.exp(
                -self._root_vel_w_exp * root_vel_dist
            )
        )
        rpos_reward = float(
            np.exp(
                -self._rpos_w_exp * rpos_dist
            )
        )
        rquat_reward = float(
            np.exp(
                -self._rquat_w_exp * rangles_dist
            )
        )
        rvel_rot_reward = float(
            np.exp(
                -self._rvel_w_exp * rvel_rot_dist
            )
        )
        rvel_lin_reward = float(
            np.exp(
                -self._rvel_w_exp * rvel_lin_dist
            )
        )

        mimic_reward = float(
            self._qpos_w_sum * qpos_reward
            + self._qvel_w_sum * qvel_reward
            + self._root_pos_w_sum * root_pos_reward
            + self._root_vel_w_sum * root_vel_reward
            + self._rpos_w_sum * rpos_reward
            + self._rquat_w_sum * rquat_reward
            + self._rvel_w_sum
            * (rvel_rot_reward + rvel_lin_reward)
        )

        residual_l2_penalty = (
            -self._residual_l2_weight
            * float(
                np.mean(
                    np.square(residual_action)
                )
            )
        )

        # Reward 保持 walk 版本：
        # Mimic tracking plus run-specific rewards and residual L2.
        # 同时保留原 walk 代码的非负截断。
        current_forward_velocity = float(
            root_local_velocity(data)[0]
        )
        reference_forward_velocity = float(
            root_local_velocity(ref_data)[0]
        )
        propulsive_reward = _run_propulsive_reward(
            current_forward_velocity,
            reference_forward_velocity,
            flight_phase,
            weight=self.propulsive_velocity_weight,
        )
        vertical_grf_body_weights = (
            self._vertical_grf_body_weights()
        )
        impact_penalty = _run_impact_penalty(
            vertical_grf_body_weights,
            limit_body_weights=self.grf_limit_body_weights,
            weight=self.grf_penalty_weight,
        )
        total_reward = max(
            float(mimic_reward)
            + propulsive_reward
            + impact_penalty
            + residual_l2_penalty,
            0.0,
        )

        self._prev_blended_action = blended_action.copy()

        pelvis_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "pelvis",
        )

        if pelvis_id != -1:
            actual_pelvis_pos = np.asarray(
                data.xpos[pelvis_id],
                dtype=np.float32,
            )
            ref_pelvis_pos = np.asarray(
                ref_data.xpos[pelvis_id],
                dtype=np.float32,
            )
        else:
            actual_pelvis_pos = np.asarray(
                data.qpos[:3],
                dtype=np.float32,
            )
            ref_pelvis_pos = np.asarray(
                ref_data.qpos[:3],
                dtype=np.float32,
            )

        pelvis_z = float(actual_pelvis_pos[2])
        mean_site_dev = float(
            np.mean(
                np.linalg.norm(
                    site_rpos - ref_site_rpos,
                    axis=-1,
                )
            )
        )
        root_dev = float(
            np.linalg.norm(
                actual_pelvis_pos
                - ref_pelvis_pos
            )
        )

        residual_terminated = bool(
            not np.isfinite(mean_site_dev)
            or not np.isfinite(root_dev)
            or not np.isfinite(pelvis_z)
            or mean_site_dev > 0.8
            or root_dev > 2.0
            or pelvis_z < 0.35
            or pelvis_z > 2.0
        )
        terminated = bool(
            base_terminated
            or residual_terminated
        )

        if base_terminated:
            termination_reason = "base"
        elif residual_terminated:
            termination_reason = "residual_tracking"
        elif truncated:
            termination_reason = "time_limit"
        else:
            termination_reason = ""

        info = dict(info)
        info.update(
            {
                "reward_total": float(total_reward),
                "reward_mimic": float(mimic_reward),
                "reward_rpos": float(rpos_reward),
                "reward_rquat": float(rquat_reward),
                "reward_root_vel": float(root_vel_reward),
                "penalty_residual_l2": float(
                    residual_l2_penalty
                ),
                "flight_phase": bool(flight_phase),
                "vertical_grf_body_weights": float(
                    vertical_grf_body_weights
                ),
                "propulsive": float(propulsive_reward),
                "impact": float(impact_penalty),
                "alpha_mean": float(np.mean(alpha)),
                "configured_residual_scale": float(
                    self.residual_scale
                ),
                "qacc_shield_factor": float(
                    qacc_shield_factor
                ),
                "predicted_base_qacc": float(
                    predicted_base_qacc
                ),
                "predicted_candidate_qacc": float(
                    predicted_candidate_qacc
                ),
                "applied_residual_rms": float(
                    np.sqrt(np.mean(np.square(safe_residual)))
                ),
                "residual_rms": float(
                    np.sqrt(np.mean(np.square(residual_action)))
                ),
                "applied_scaled_residual_rms": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                self._applied_scaled_residual
                            )
                        )
                    )
                ),
                "blended_action_saturation": float(
                    np.mean(np.abs(blended_action) > 0.999)
                ),
                "max_abs_qacc": float(
                    np.max(np.abs(np.asarray(data.qacc)))
                ) if np.asarray(data.qacc).size > 0 else 0.0,
                "left_foot_contact": bool(left_foot_contact),
                "right_foot_contact": bool(right_foot_contact),
                "tracking_error": float(root_dev),
                "mean_site_dev": float(mean_site_dev),
                "termination_reason": termination_reason,
                "constraint_violation": bool(
                    residual_terminated
                ),
                "success": bool(truncated and not terminated),
                "reward_terms": {
                    "mimic": float(mimic_reward),
                    "propulsive": float(propulsive_reward),
                    "impact": float(impact_penalty),
                    "residual_l2": float(
                        residual_l2_penalty
                    ),
                },
            }
        )

        return (
            obs,
            float(total_reward),
            terminated,
            truncated,
            info,
        )

    def _vertical_grf_body_weights(self) -> float:
        model = self.env.unwrapped._model
        data = self.env.unwrapped._data
        total_vertical_force = 0.0
        contact_force = np.zeros(6, dtype=np.float64)

        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            geom_1 = int(contact.geom1)
            geom_2 = int(contact.geom2)
            if not self._is_foot_support_contact(
                geom_1,
                geom_2,
            ):
                continue

            mujoco.mj_contactForce(
                model,
                data,
                contact_index,
                contact_force,
            )
            contact_frame = np.asarray(
                contact.frame,
                dtype=np.float64,
            ).reshape(3, 3)
            world_force = contact_frame.T @ contact_force[:3]
            total_vertical_force += abs(float(world_force[2]))

        body_weight = float(
            np.sum(model.body_mass) * abs(model.opt.gravity[2])
        )
        if body_weight <= 0.0:
            return 0.0
        return total_vertical_force / body_weight

    def _phase_alpha(self) -> np.ndarray:
        alpha = np.full(
            self._action_shape,
            self.alpha_stance,
            dtype=np.float32,
        )

        left_swing = not self._left_foot_contact()
        right_swing = not self._right_foot_contact()

        if left_swing:
            alpha[self._left_action_mask] = (
                self.alpha_swing
            )
        if right_swing:
            alpha[self._right_action_mask] = (
                self.alpha_swing
            )

        if not np.any(
            self._left_action_mask
            | self._right_action_mask
        ):
            alpha[:] = (
                self.alpha_swing
                if left_swing or right_swing
                else self.alpha_stance
            )

        return alpha * self.residual_scale

    def _left_foot_contact(self) -> bool:
        return self._foot_contact(
            self._left_foot_body_ids
        )

    def _right_foot_contact(self) -> bool:
        return self._foot_contact(
            self._right_foot_body_ids
        )

    def _foot_contact(
        self,
        foot_body_ids: tuple[int, ...],
    ) -> bool:
        data = self.env.unwrapped._data
        selected_foot_body_ids = frozenset(foot_body_ids)

        if not selected_foot_body_ids:
            return False

        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            if self._is_foot_support_contact(
                int(contact.geom1),
                int(contact.geom2),
                selected_foot_body_ids,
            ):
                return True

        return False

    def _is_foot_support_contact(
        self,
        geom_1: int,
        geom_2: int,
        foot_body_ids: frozenset[int] | None = None,
    ) -> bool:
        model = self.env.unwrapped._model
        selected_foot_body_ids = (
            self._foot_body_ids
            if foot_body_ids is None
            else foot_body_ids
        )
        body_1 = int(model.geom_bodyid[geom_1])
        body_2 = int(model.geom_bodyid[geom_2])
        return bool(
            (
                body_1 in selected_foot_body_ids
                and geom_2 in self._support_geom_ids
            )
            or (
                body_2 in selected_foot_body_ids
                and geom_1 in self._support_geom_ids
            )
        )

    @staticmethod
    def _infer_support_geom_ids(
        model: mujoco.MjModel,
    ) -> frozenset[int]:
        support_tokens = (
            "floor",
            "ground",
            "terrain",
            "plane",
            "step",
            "stair",
            "stone",
        )
        world_surface_types = {
            int(mujoco.mjtGeom.mjGEOM_PLANE),
            int(mujoco.mjtGeom.mjGEOM_HFIELD),
        }
        support_ids: set[int] = set()

        for index in range(int(model.ngeom)):
            name = (
                mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    index,
                )
                or ""
            ).lower()
            world_surface = (
                int(model.geom_bodyid[index]) == 0
                and int(model.geom_type[index]) in world_surface_types
            )
            if world_surface or any(
                token in name
                for token in support_tokens
            ):
                support_ids.add(index)

        if not support_ids:
            raise ValueError(
                "No support geometries found. Configure a named support "
                "geom or a world-attached PLANE/HFIELD surface."
            )
        return frozenset(support_ids)

    def _infer_action_side_masks(
        self,
        model: mujoco.MjModel,
    ) -> tuple[np.ndarray, np.ndarray]:
        left = np.zeros(
            self._action_shape,
            dtype=bool,
        )
        right = np.zeros(
            self._action_shape,
            dtype=bool,
        )

        for index in range(int(model.nu)):
            name = (
                mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    index,
                )
                or ""
            )
            lower = name.lower()

            is_left = (
                any(
                    token in lower
                    for token in (
                        "_l",
                        "-l",
                        "left",
                        "_lt",
                    )
                )
                or lower.endswith("l")
            )
            is_right = (
                any(
                    token in lower
                    for token in (
                        "_r",
                        "-r",
                        "right",
                        "_rt",
                    )
                )
                or lower.endswith("r")
            )

            if is_left and not is_right:
                left[index] = True
            elif is_right and not is_left:
                right[index] = True

        return left, right

    @staticmethod
    def _body_ids(
        model: mujoco.MjModel,
        names: tuple[str, ...],
    ) -> tuple[int, ...]:
        ids: list[int] = []

        for name in names:
            body_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                name,
            )
            if body_id != -1:
                ids.append(int(body_id))

        return tuple(ids)



# ==================================================
# 7. 注册与工厂
# ==================================================
def make_env(**kwargs):
    base_model_dir = kwargs.pop(
        "base_model_dir",
        DEFAULT_BASE_MODEL_DIR,
    )
    residual_scale = kwargs.pop(
        "residual_scale",
        0.6,
    )
    propulsive_velocity_weight = kwargs.pop(
        "propulsive_velocity_weight",
        0.3,
    )
    grf_penalty_weight = kwargs.pop(
        "grf_penalty_weight",
        0.1,
    )
    grf_limit_body_weights = kwargs.pop(
        "grf_limit_body_weights",
        2.5,
    )
    env = MSKBenchResidualRunEnvV0(**kwargs)
    return MSKBenchResidualRunWrapper(
        env,
        base_model_dir=base_model_dir,
        residual_scale=residual_scale,
        propulsive_velocity_weight=propulsive_velocity_weight,
        grf_penalty_weight=grf_penalty_weight,
        grf_limit_body_weights=grf_limit_body_weights,
    )


def _make_wrapped_env(**kwargs):
    return make_env(**kwargs)


def register_env() -> str:
    env_id = "MSKBenchResidualRunEnvV0-v0"

    if env_id not in gymnasium.envs.registry:
        current_module = (
            __name__
            if __name__ != "__main__"
            else "rl_paradigms.residualrl.run"
        )

        gymnasium.register(
            id=env_id,
            entry_point=(
                f"{current_module}:"
                "_make_wrapped_env"
            ),
            max_episode_steps=1000,
        )

    return env_id