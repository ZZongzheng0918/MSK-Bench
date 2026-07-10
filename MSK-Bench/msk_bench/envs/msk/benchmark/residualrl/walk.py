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


class MSKBenchResidualWalkEnvV0(FullBodyReferenceEnv):
    """Residual-control flat walking task with energy-aware adaptation."""

    motion_filename = "walking_medium09_poses.npz"
    max_episode_steps = 1000


class MSKBenchResidualWalkWrapper(gymnasium.Wrapper):
    def __init__(self, env, base_model_dir=None, residual_scale: float = 1.0):
        super().__init__(env)
        self.base_model_dir = default_base_model_dir(base_model_dir)
        self.residual_scale = float(residual_scale)
        self.expected_obs_dim = DEFAULT_OBS_DIM
        self.action_space = env.action_space
        self._prev_blended_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._prev_activation = np.zeros(self.action_space.shape, dtype=np.float32)

        self.alpha_stance = 0.2
        self.alpha_swing = 0.6
        self.metabolic_weight = 0.05
        self.metabolic_smoothing = 0.1
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
        self._left_action_mask, self._right_action_mask = self._infer_action_side_masks(model)
        self._left_foot_body_ids = self._body_ids(model, ("talus_l", "calcn_l", "foot_l", "toes_l"))
        self._right_foot_body_ids = self._body_ids(model, ("talus_r", "calcn_r", "foot_r", "toes_r"))

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._prev_blended_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._prev_activation = np.zeros(self.action_space.shape, dtype=np.float32)
        return obs, info

    def step(self, residual_action):
        current_obs = self.env.unwrapped._get_full_obs()
        policy_fn, train_state = get_jax_policy(self.base_model_dir, expected_obs_dim=self.expected_obs_dim)
        base_action = np.array(policy_fn(train_state, current_obs))
        safe_residual = np.clip(residual_action, -1.0, 1.0)
        alpha = self._phase_alpha()
        blended_action = np.clip(base_action + alpha * safe_residual, -1.0, 1.0)

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

        qpos_reward = np.exp(-self._qpos_w_exp * qpos_dist)
        qvel_reward = np.exp(-self._qvel_w_exp * qvel_dist)
        root_pos_reward = np.exp(-self._root_pos_w_exp * root_pos_dist)
        root_vel_reward = np.exp(-self._root_vel_w_exp * root_vel_dist)
        rpos_reward = np.exp(-self._rpos_w_exp * rpos_dist)
        rquat_reward = np.exp(-self._rquat_w_exp * rangles_dist)
        rvel_rot_reward = np.exp(-self._rvel_w_exp * rvel_rot_dist)
        rvel_lin_reward = np.exp(-self._rvel_w_exp * rvel_lin_dist)

        mimic_reward = (
            self._qpos_w_sum * qpos_reward
            + self._qvel_w_sum * qvel_reward
            + self._root_pos_w_sum * root_pos_reward
            + self._root_vel_w_sum * root_vel_reward
            + self._rpos_w_sum * rpos_reward
            + self._rquat_w_sum * rquat_reward
            + self._rvel_w_sum * (rvel_rot_reward + rvel_lin_reward)
        )

        activation = self._activation_vector(data, blended_action)
        activation_delta = activation - self._align_like(self._prev_activation, activation)
        metabolic_penalty = -self.metabolic_weight * float(
            np.sum(np.square(activation) + self.metabolic_smoothing * np.square(activation_delta))
        )
        residual_l2_penalty = -self._residual_l2_weight * float(np.mean(np.square(residual_action)))
        total_reward = max(float(mimic_reward) + residual_l2_penalty + metabolic_penalty, 0.0)

        self._prev_activation = activation.copy()
        self._prev_blended_action = blended_action.copy()

        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        actual_pelvis_pos = data.xpos[pelvis_id] if pelvis_id != -1 else np.array([0.0, 0.0, 0.9])
        ref_pelvis_pos = ref_data.xpos[pelvis_id] if pelvis_id != -1 else np.array([0.0, 0.0, 0.9])
        pelvis_z = actual_pelvis_pos[2]
        mean_site_dev = np.mean(np.linalg.norm(site_rpos - ref_site_rpos, axis=-1))
        root_dev = np.linalg.norm(actual_pelvis_pos - ref_pelvis_pos)
        terminated = bool(mean_site_dev > 0.8 or root_dev > 2.0 or pelvis_z < 0.35 or pelvis_z > 2.0)

        info.update(
            {
                "reward_total": float(total_reward),
                "reward_mimic": float(mimic_reward),
                "reward_rpos": float(rpos_reward),
                "reward_rquat": float(rquat_reward),
                "reward_root_vel": float(root_vel_reward),
                "penalty_metabolic": float(metabolic_penalty),
                "penalty_residual_l2": float(residual_l2_penalty),
                "alpha_mean": float(np.mean(alpha)),
                "left_foot_contact": bool(self._left_foot_contact()),
                "right_foot_contact": bool(self._right_foot_contact()),
                "tracking_error": float(root_dev),
                "mean_site_dev": float(mean_site_dev),
            }
        )
        return obs, float(total_reward), terminated, truncated, info

    def _phase_alpha(self):
        alpha = np.full(self.action_space.shape, self.alpha_stance, dtype=np.float32)
        left_swing = not self._left_foot_contact()
        right_swing = not self._right_foot_contact()
        if left_swing:
            alpha[self._left_action_mask] = self.alpha_swing
        if right_swing:
            alpha[self._right_action_mask] = self.alpha_swing
        if not np.any(self._left_action_mask | self._right_action_mask):
            alpha[:] = self.alpha_swing if left_swing or right_swing else self.alpha_stance
        return alpha * self.residual_scale

    def _left_foot_contact(self):
        return self._foot_contact(self._left_foot_body_ids)

    def _right_foot_contact(self):
        return self._foot_contact(self._right_foot_body_ids)

    def _foot_contact(self, foot_body_ids):
        model = self.env.unwrapped._model
        data = self.env.unwrapped._data
        foot_body_ids = set(foot_body_ids)
        if foot_body_ids:
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                body_1 = int(model.geom_bodyid[contact.geom1])
                body_2 = int(model.geom_bodyid[contact.geom2])
                if body_1 in foot_body_ids or body_2 in foot_body_ids:
                    return True
            for body_id in foot_body_ids:
                try:
                    if data.xpos[body_id][2] < 0.08:
                        return True
                except Exception:
                    continue
        return False

    def _infer_action_side_masks(self, model):
        left = np.zeros(self.action_space.shape, dtype=bool)
        right = np.zeros(self.action_space.shape, dtype=bool)
        for index in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or ""
            lower = name.lower()
            is_left = any(token in lower for token in ("_l", "-l", "left", "_lt")) or lower.endswith("l")
            is_right = any(token in lower for token in ("_r", "-r", "right", "_rt")) or lower.endswith("r")
            if is_left and not is_right:
                left[index] = True
            elif is_right and not is_left:
                right[index] = True
        return left, right

    def _body_ids(self, model, names):
        ids = []
        for name in names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id != -1:
                ids.append(int(body_id))
        return tuple(ids)

    def _activation_vector(self, data, fallback):
        activation = getattr(data, "act", None)
        if activation is None or np.asarray(activation).size == 0:
            return np.asarray(fallback, dtype=np.float32).copy()
        return np.asarray(activation, dtype=np.float32).copy()

    def _align_like(self, value, target):
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        if value.shape == target.shape:
            return value
        aligned = np.zeros_like(target)
        aligned[: min(value.size, target.size)] = value[: min(value.size, target.size)]
        return aligned


def make_env(**kwargs):
    base_model_dir = kwargs.pop("base_model_dir", None)
    residual_scale = kwargs.pop("residual_scale", 1.0)
    env = MSKBenchResidualWalkEnvV0(**kwargs)
    return MSKBenchResidualWalkWrapper(env, base_model_dir=base_model_dir, residual_scale=residual_scale)