from __future__ import annotations

import gymnasium
import mujoco
import numpy as np
from loco_mujoco.core.utils.math import calculate_relative_site_quantities

from .common import (
    DEFAULT_OBS_DIM,
    FullBodyReferenceEnv,
    default_base_model_dir,
    get_jax_policy,
    root_local_velocity,
)


class MSKBenchResidualRunEnvV0(FullBodyReferenceEnv):
    """Residual-control full-body running task driven by a reference policy."""

    motion_filename = "walking_run04_poses.npz"
    max_episode_steps = 5000


class MSKBenchResidualRunWrapper(gymnasium.Wrapper):
    def __init__(self, env, base_model_dir=None, residual_scale: float = 0.6):
        super().__init__(env)
        self.base_model_dir = default_base_model_dir(base_model_dir)
        self.residual_scale = float(residual_scale)
        self.expected_obs_dim = DEFAULT_OBS_DIM
        self.action_space = env.action_space
        self._prev_blended_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.max_pelvis_z = -np.inf
        self.last_pelvis_z = None

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

        self._action_out_of_bounds_coeff = 0.0
        self._action_rate_coeff = 0.0
        self._activation_energy_coeff = 0.0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.max_pelvis_z = -np.inf
        self.last_pelvis_z = None
        self._prev_blended_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return obs, info

    def step(self, residual_action):
        current_obs = self.env.unwrapped._get_full_obs()
        policy_fn, train_state = get_jax_policy(self.base_model_dir, expected_obs_dim=self.expected_obs_dim)
        base_action = np.array(policy_fn(train_state, current_obs))
        safe_residual = np.clip(residual_action, -1.0, 1.0) * self.residual_scale
        blended_action = np.clip(base_action + safe_residual, -1.0, 1.0)

        if hasattr(self.env.unwrapped, "_data"):
            self.env.unwrapped._data.ctrl[:] = blended_action

        obs, _, terminated, truncated, info = self.env.step(blended_action)
        data = self.env.unwrapped._data
        model = self.env.unwrapped._model
        current_step = self.env.unwrapped.ref_step
        ref_data = self.env.unwrapped.th.get_traj_data_at(0, current_step)

        goal_wrapper = self.env.unwrapped._goal_wrapper
        goal_wrapper._lazy_init()
        site_rpos, site_rangles, site_rvel = calculate_relative_site_quantities(
            data, goal_wrapper._rel_site_ids, goal_wrapper._rel_body_ids, goal_wrapper._body_rootid, np
        )
        ref_site_rpos, ref_site_rangles, ref_site_rvel = calculate_relative_site_quantities(
            ref_data, goal_wrapper._rel_site_ids, goal_wrapper._rel_body_ids, goal_wrapper._body_rootid, np
        )

        rpos_dist = np.mean(np.square(site_rpos - ref_site_rpos))
        rangles_dist = np.mean(np.square(site_rangles - ref_site_rangles))
        rvel_rot_dist = np.mean(np.square(site_rvel[:, :3] - ref_site_rvel[:, :3]))
        rvel_lin_dist = np.mean(np.square(site_rvel[:, 3:] - ref_site_rvel[:, 3:]))
        root_vel_dist = np.mean(np.square(root_local_velocity(data) - root_local_velocity(ref_data)))
        qpos_dist = np.mean(np.square(data.qpos[7:] - ref_data.qpos[7:]))
        qvel_dist = np.mean(np.square(data.qvel[6:] - ref_data.qvel[6:]))
        root_pos_dist = np.mean(np.square(data.qpos[:3] - ref_data.qpos[:3]))

        total_mimic_reward = (
            self._qpos_w_sum * np.exp(-self._qpos_w_exp * qpos_dist)
            + self._qvel_w_sum * np.exp(-self._qvel_w_exp * qvel_dist)
            + self._root_pos_w_sum * np.exp(-self._root_pos_w_exp * root_pos_dist)
            + self._root_vel_w_sum * np.exp(-self._root_vel_w_exp * root_vel_dist)
            + self._rpos_w_sum * np.exp(-self._rpos_w_exp * rpos_dist)
            + self._rquat_w_sum * np.exp(-self._rquat_w_exp * rangles_dist)
            + self._rvel_w_sum * (
                np.exp(-self._rvel_w_exp * rvel_rot_dist) + np.exp(-self._rvel_w_exp * rvel_lin_dist)
            )
        )

        residual_l2_penalty = -0.05 * np.mean(np.square(residual_action))
        activation_energy_penalty = -np.mean(np.square(data.act)) if data.act is not None else 0.0
        action_rate_penalty = -np.sum(np.square(blended_action - self._prev_blended_action))
        total_penalties = max(
            residual_l2_penalty
            + self._activation_energy_coeff * activation_energy_penalty
            + self._action_rate_coeff * action_rate_penalty,
            -1.0,
        )
        total_reward = max(total_mimic_reward + total_penalties, 0.0)
        self._prev_blended_action = blended_action.copy()

        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        actual_pelvis_pos = data.xpos[pelvis_id] if pelvis_id != -1 else np.array([0.0, 0.0, 0.9])
        ref_pelvis_pos = ref_data.xpos[pelvis_id] if pelvis_id != -1 else np.array([0.0, 0.0, 0.9])
        pelvis_z = actual_pelvis_pos[2]
        self.max_pelvis_z = max(self.max_pelvis_z, pelvis_z)
        self.last_pelvis_z = pelvis_z

        mean_site_dev = np.mean(np.linalg.norm(site_rpos - ref_site_rpos, axis=-1))
        root_dev = np.linalg.norm(actual_pelvis_pos - ref_pelvis_pos)
        terminated = bool(mean_site_dev > 0.8 or root_dev > 2.0 or pelvis_z < 0.35 or pelvis_z > 2.0)

        info["tracking_error"] = float(root_dev)
        info["mean_site_dev"] = float(mean_site_dev)
        info["mimic_reward"] = float(total_reward)
        return obs, float(total_reward), terminated, truncated, info


def make_env(**kwargs):
    base_model_dir = kwargs.pop("base_model_dir", None)
    residual_scale = kwargs.pop("residual_scale", 0.6)
    env = MSKBenchResidualRunEnvV0(**kwargs)
    return MSKBenchResidualRunWrapper(env, base_model_dir=base_model_dir, residual_scale=residual_scale)