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


class MSKBenchResidualStairEnvV0(FullBodyReferenceEnv):
    """Residual-control stair navigation task driven by a reference policy."""

    motion_filename = "stair_prior_89d.npz"
    max_episode_steps = 2000

    def reset(self, *args, **kwargs):
        self._slow_wait_counter = 0
        return super().reset(*args, **kwargs)

    def _advance_reference(self):
        if self.ref_step >= self.th.n_frames - 1:
            return
        pelvis_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        actual_pelvis_pos = self._data.xpos[pelvis_id] if pelvis_id != -1 else np.array([0.0, 0.0, 0.9])
        ref_data = self.th.get_traj_data_at(0, self.ref_step)
        ref_pelvis_pos = ref_data.xpos[pelvis_id] if pelvis_id != -1 else np.array([0.0, 0.0, 0.9])
        root_dev = np.linalg.norm(actual_pelvis_pos[:2] - ref_pelvis_pos[:2])
        if root_dev < 0.4:
            self.ref_step += 1
            return
        if self._slow_wait_counter % 2 == 0:
            self.ref_step += 1
        self._slow_wait_counter += 1


class MSKBenchResidualStairWrapper(gymnasium.Wrapper):
    def __init__(self, env, base_model_dir=None, residual_scale: float = 0.6):
        super().__init__(env)
        self.base_model_dir = default_base_model_dir(base_model_dir)
        self.residual_scale = float(residual_scale)
        self.expected_obs_dim = DEFAULT_OBS_DIM
        self.action_space = env.action_space
        self._prev_blended_action = np.zeros(self.action_space.shape, dtype=np.float32)

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

        self._action_rate_coeff = 0.0
        self._activation_energy_coeff = 0.0

    def reset(self, **kwargs):
        self._prev_blended_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return self.env.reset(**kwargs)

    def step(self, residual_action):
        current_obs = self.env.unwrapped._get_full_obs()
        policy_fn, train_state = get_jax_policy(
            self.base_model_dir,
            expected_obs_dim=self.expected_obs_dim,
            act_dim=self.action_space.shape[0],
        )
        base_action = np.array(policy_fn(train_state, current_obs))
        safe_residual = np.clip(residual_action, -1.0, 1.0) * self.residual_scale
        blended_action = np.clip(base_action + safe_residual, -1.0, 1.0)

        if hasattr(self.env.unwrapped, "_data"):
            self.env.unwrapped._data.ctrl[:] = blended_action

        obs, _, terminated, truncated, env_info = self.env.step(blended_action)
        data = self.env.unwrapped._data
        model = self.env.unwrapped._model
        th = self.env.unwrapped.th
        current_step = self.env.unwrapped.ref_step
        ref_data = th.get_traj_data_at(0, current_step)

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
        root_pos_dist = np.mean(np.square(data.qpos[:2] - ref_data.qpos[:2]))

        qpos_reward = np.exp(-self._qpos_w_exp * qpos_dist)
        qvel_reward = np.exp(-self._qvel_w_exp * qvel_dist)
        root_pos_reward = np.exp(-self._root_pos_w_exp * root_pos_dist)
        root_vel_reward = np.exp(-self._root_vel_w_exp * root_vel_dist)
        rpos_reward = np.exp(-self._rpos_w_exp * rpos_dist)
        rquat_reward = np.exp(-self._rquat_w_exp * rangles_dist)
        rvel_rot_reward = np.exp(-self._rvel_w_exp * rvel_rot_dist)
        rvel_lin_reward = np.exp(-self._rvel_w_exp * rvel_lin_dist)

        total_mimic_reward = (
            self._qpos_w_sum * qpos_reward
            + self._qvel_w_sum * qvel_reward
            + self._root_pos_w_sum * root_pos_reward
            + self._root_vel_w_sum * root_vel_reward
            + self._rpos_w_sum * rpos_reward
            + self._rquat_w_sum * rquat_reward
            + self._rvel_w_sum * (rvel_rot_reward + rvel_lin_reward)
        )

        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        actual_pelvis_pos = data.xpos[pelvis_id] if pelvis_id != -1 else np.array([0.0, 0.0, 0.9])
        future_step = min(current_step + int(1.0 / th.dt), th.n_frames - 1)
        future_ref_pos = th.get_traj_data_at(0, future_step).xpos[pelvis_id] if pelvis_id != -1 else np.array([0.0, 0.0, 0.9])
        dir_to_future = future_ref_pos - actual_pelvis_pos
        dist_to_future = np.linalg.norm(dir_to_future)
        progress_reward = 0.0
        if dist_to_future > 0.1:
            dir_to_future = dir_to_future / dist_to_future
            vel_proj = np.dot(data.qvel[:3], dir_to_future)
            climb_vel = data.qvel[2]
            progress_reward = 0.3 * (np.clip(vel_proj, 0.0, 2.0) + 0.5 * np.clip(climb_vel, 0.0, 1.5))

        residual_l2_penalty = -0.02 * np.mean(np.square(residual_action))
        activation_energy_penalty = -np.mean(np.square(data.act)) if data.act is not None else 0.0
        action_rate_penalty = -np.mean(np.square(blended_action - self._prev_blended_action))
        total_penalties = max(
            residual_l2_penalty
            + self._activation_energy_coeff * activation_energy_penalty
            + self._action_rate_coeff * action_rate_penalty,
            -1.0,
        )
        total_reward = max(total_mimic_reward + progress_reward + total_penalties, 0.0)
        self._prev_blended_action = blended_action.copy()

        tracking_error = 0.0
        if pelvis_id != -1:
            ref_pelvis_pos = ref_data.xpos[pelvis_id]
            tracking_error = np.linalg.norm(actual_pelvis_pos - ref_pelvis_pos)
            if actual_pelvis_pos[2] < (ref_pelvis_pos[2] - 0.45) or actual_pelvis_pos[2] < 0.35 or tracking_error > 1.0:
                terminated = True

        env_info.update(
            {
                "reward_total": float(total_reward),
                "reward_rpos": float(rpos_reward),
                "reward_rquat": float(rquat_reward),
                "reward_root_vel": float(root_vel_reward),
                "penalty_total": float(total_penalties),
                "err_rpos": float(np.sqrt(rpos_dist)),
                "tracking_error": float(tracking_error),
                "termination_reason": "residual_tracking" if bool(terminated) else "",
                "constraint_violation": bool(terminated),
                "success": False,
                "reward_terms": {
                    "mimic": float(total_mimic_reward),
                    "progress": float(progress_reward),
                    "penalties": float(total_penalties),
                },
            }
        )
        return obs, float(total_reward), bool(terminated), truncated, env_info


def make_env(**kwargs):
    base_model_dir = kwargs.pop("base_model_dir", None)
    residual_scale = kwargs.pop("residual_scale", 0.6)
    env = MSKBenchResidualStairEnvV0(**kwargs)
    return MSKBenchResidualStairWrapper(env, base_model_dir=base_model_dir, residual_scale=residual_scale)
