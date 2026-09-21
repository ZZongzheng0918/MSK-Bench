from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import gymnasium
import gymnasium.spaces as spaces
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from loco_mujoco.core.utils.math import calculate_relative_site_quantities

try:
    from musclemimic.integrations.msk_bench import (
        MyoFullBody,
        get_jax_policy,
        resolve_checkpoint_source,
    )
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("musclemimic"):
        raise ModuleNotFoundError(
            "ResidualRL requires MuscleMimic. Install the local musclemimic project "
            "and Gymnasium; see the MSK-Bench workspace README."
        ) from exc
    raise

RESIDUAL_DIR = Path(__file__).resolve().parent
__all__ = ["get_jax_policy"]

DEFAULT_OBS_DIM = 2418
# The canonical anatomical model has 416 muscles; ResidualRL controls the active non-finger actuators.
DEFAULT_ANATOMICAL_MUSCLE_COUNT = 416
DEFAULT_ACT_DIM = 354


def resolve_path(value: str | os.PathLike | None, default: Path) -> str:
    if value is None:
        return str(default)
    return str(Path(value))


def resolve_model_path(value: str | os.PathLike | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model XML does not exist: {path}")
    return str(path)


class NPZTrajectoryAdapter:
    def __init__(self, npz_path, model, dt: float = 0.01):
        self.npz_path = resolve_path(npz_path, RESIDUAL_DIR / "missing_motion.npz")
        data = np.load(self.npz_path, allow_pickle=True)
        if "states" in data:
            self.qpos_data = data["states"]
        elif "qpos" in data:
            self.qpos_data = data["qpos"]
        else:
            raise KeyError(f"Could not find 'states' or 'qpos' in {self.npz_path}")

        self.n_frames = self.qpos_data.shape[0]
        self.dt = float(dt)
        self.model = model
        self.is_numpy = True

        if "qvel" in data:
            self.qvel_data = data["qvel"]
            self.has_true_qvel = True
        else:
            self.has_true_qvel = False
            root_pos = self.qpos_data[:, 0:3]
            self.root_lin_vel = np.gradient(root_pos, axis=0) / self.dt
            self.joint_vel = np.gradient(self.qpos_data[:, 7:], axis=0) / self.dt

        self._ref_data = mujoco.MjData(model)
        self.internal_step = 0

    def init_state(self, env, key, model, data, backend):
        self.internal_step = 0
        return SimpleNamespace(traj_no=0, subtraj_step_no=0)

    def reset_state(self, env, model, data, carry, backend):
        self.internal_step = 0
        return data, carry

    def update_state(self, env, model, data, carry, backend):
        self.internal_step += 1
        return carry

    def len_trajectory(self, traj_no=0):
        return self.n_frames

    def get_traj_data_at(self, traj_no, step, carry=None, backend=None):
        step = int(np.clip(step, 0, self.n_frames - 1))
        ref_qpos = self.qpos_data[step]
        if self.has_true_qvel:
            ref_qvel = self.qvel_data[step]
        else:
            ref_qvel = np.concatenate([self.root_lin_vel[step], np.zeros(3), self.joint_vel[step]])

        self._ref_data.qpos[: len(ref_qpos)] = ref_qpos[: self.model.nq]
        self._ref_data.qvel[: len(ref_qvel)] = ref_qvel[: self.model.nv]
        mujoco.mj_forward(self.model, self._ref_data)
        return self._ref_data


class FullBodyGoalWrapper:
    SITES_FOR_MIMIC: ClassVar[list[str]] = [
        "pelvis_mimic", "upper_body_mimic", "head_mimic",
        "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic",
        "right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
        "left_hip_mimic", "left_knee_mimic", "left_ankle_mimic", "left_toes_mimic",
        "right_hip_mimic", "right_knee_mimic", "right_ankle_mimic", "right_toes_mimic",
    ]
    N_STEP_LOOKAHEAD = 5
    N_STEP_STRIDE = 20
    GOAL_DIM = 469

    def __init__(self, env):
        self.env = env
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return
        model = self.env._model
        self._ref_data = mujoco.MjData(model)
        site_ids: list[int] = []
        missing_sites: list[str] = []
        for site in self.SITES_FOR_MIMIC:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
            if site_id == -1:
                missing_sites.append(site)
            else:
                site_ids.append(int(site_id))
        if missing_sites:
            raise ValueError(f"Missing mimic sites required by ResidualRL: {', '.join(missing_sites)}")
        self._rel_site_ids = np.asarray(site_ids, dtype=np.int32)
        self._body_rootid = model.body_rootid
        self._site_bodyid = model.site_bodyid
        self._rel_body_ids = self._site_bodyid[self._rel_site_ids]

        root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if root_joint_id == -1:
            raise ValueError("Missing mimic joint required by ResidualRL: root")
        qpos_adr = model.jnt_qposadr[root_joint_id]
        qvel_adr = model.jnt_dofadr[root_joint_id]
        self._root_qpos_ind = np.array([qpos_adr, qpos_adr + 1, qpos_adr + 2])
        self._root_qvel_ind = np.arange(qvel_adr, qvel_adr + 6)
        self._initialized = True

    def get_goal_obs(self) -> np.ndarray:
        self._lazy_init()
        data = self.env._data
        current_step = getattr(self.env, "ref_step", 0)
        traj_state = SimpleNamespace(traj_no=0, subtraj_step_no=current_step)

        site_rpos, site_rangles, site_rvel = calculate_relative_site_quantities(
            data, self._rel_site_ids, self._rel_body_ids, self._body_rootid, np
        )
        cur_rpos = np.ravel(site_rpos).astype(np.float32)
        cur_rang = np.ravel(site_rangles).astype(np.float32)
        cur_rvel = np.ravel(site_rvel).astype(np.float32)

        traj_goal_obs = self._build_concise_traj_goal(traj_state)
        traj_len = self.env.th.len_trajectory(traj_state.traj_no)
        motion_phase = np.array([float(traj_state.subtraj_step_no) / max(float(traj_len), 1.0)], dtype=np.float32)
        return np.concatenate([cur_rpos, cur_rang, cur_rvel, traj_goal_obs, motion_phase])

    def _build_concise_traj_goal(self, traj_state) -> np.ndarray:
        traj_len = self.env.th.len_trajectory(traj_state.traj_no)
        ref_mjdata = self.env.th.get_traj_data_at(traj_state.traj_no, traj_state.subtraj_step_no)
        ref_root_pos = ref_mjdata.qpos[self._root_qpos_ind].copy()
        ref_root_vel = ref_mjdata.qvel[self._root_qvel_ind].copy()

        site_rpos_0, _, _ = calculate_relative_site_quantities(
            ref_mjdata, self._rel_site_ids, self._rel_body_ids, self._body_rootid, np
        )
        all_site_rpos = [np.ravel(site_rpos_0).astype(np.float32)]
        all_pos_delta, all_vel_delta = [], []

        for step_offset in range(1, self.N_STEP_LOOKAHEAD):
            future_step = int(np.clip(traj_state.subtraj_step_no + step_offset * self.N_STEP_STRIDE, 0, traj_len - 1))
            future_mjdata = self.env.th.get_traj_data_at(traj_state.traj_no, future_step)
            site_rpos, _, _ = calculate_relative_site_quantities(
                future_mjdata, self._rel_site_ids, self._rel_body_ids, self._body_rootid, np
            )
            all_site_rpos.append(np.ravel(site_rpos).astype(np.float32))
            all_pos_delta.append((future_mjdata.qpos[self._root_qpos_ind] - ref_root_pos).astype(np.float32))
            all_vel_delta.append((future_mjdata.qvel[self._root_qvel_ind] - ref_root_vel).astype(np.float32))

        components = [all_site_rpos[0]]
        for index in range(len(all_pos_delta)):
            components += [all_pos_delta[index], all_vel_delta[index], all_site_rpos[index + 1]]
        return np.concatenate(components, axis=0)


class FullBodyReferenceEnv(gymnasium.Env, MyoFullBody):
    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["human", "rgb_array"]}
    motion_filename = "walking_run04_poses.npz"
    max_episode_steps = 1000

    class _SimCompat:
        def __init__(self, model, data):
            self.model = model
            self.data = data

    @property
    def sim(self):
        return self._SimCompat(self._model, self._data)

    def __init__(self, model_path=None, motion_path=None, **kwargs):
        model_path = resolve_model_path(model_path)
        motion_path = resolve_path(motion_path, RESIDUAL_DIR / self.motion_filename)
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

        self.th = NPZTrajectoryAdapter(motion_path, self._model)
        self._goal_wrapper = FullBodyGoalWrapper(self)
        self.horizon = self.th.n_frames
        self._max_episode_steps = self.th.n_frames
        self.ref_step = 0

        n_act = self._model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(DEFAULT_OBS_DIM,), dtype=np.float32)

    def reset(self, seed=None, options=None, **kwargs):
        kwargs.pop("seed", None)
        kwargs.pop("options", None)
        if seed is not None:
            try:
                self.np_random, _ = gymnasium.utils.seeding.np_random(seed)
            except Exception:
                self.np_random = np.random.default_rng(seed)
        MyoFullBody.reset(self, **kwargs)
        self.ref_step = 0
        self._data.time = 0.0

        ref_qpos = self.th.qpos_data[0]
        nq = min(len(ref_qpos), self._model.nq)
        self._data.qpos[:nq] = ref_qpos[:nq]

        if self.th.has_true_qvel:
            ref_qvel = self.th.qvel_data[0]
        else:
            ref_qvel = np.concatenate([self.th.root_lin_vel[0], np.zeros(3), self.th.joint_vel[0]])
        nv = min(len(ref_qvel), self._model.nv)
        self._data.qvel[:nv] = ref_qvel[:nv]

        if self._model.na > 0:
            self._data.act[:] = 0.15

        mujoco.mj_forward(self._model, self._data)
        return self._get_full_obs(), {}

    def step(self, action):
        MyoFullBody.step(self, action)
        self._advance_reference()
        truncated = bool(self.ref_step >= self.th.n_frames - 1)
        return self._get_full_obs(), 0.0, False, truncated, {}

    def _advance_reference(self):
        if self.ref_step < self.th.n_frames - 1:
            self.ref_step += 1

    def _get_full_obs(self):
        obs_dict = self._create_observation(self._model, self._data, getattr(self, "_carry", None))
        base_obs = np.asarray(obs_dict[0] if isinstance(obs_dict, tuple) else obs_dict, dtype=np.float32).ravel()
        goal_obs = self._goal_wrapper.get_goal_obs()
        return np.concatenate([base_obs, goal_obs])


def root_local_velocity(mj_data):
    lin_vel_global = mj_data.qvel[:3]
    ang_vel_global = mj_data.qvel[3:6]
    quat = mj_data.qpos[3:7]
    root_rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    lin_vel_local = root_rot.inv().apply(lin_vel_global)
    return np.concatenate([lin_vel_local, ang_vel_global])


def default_base_model_dir(value=None) -> str:
    return resolve_checkpoint_source(value)

