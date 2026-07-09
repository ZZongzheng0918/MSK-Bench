"""=================================================
# Copyright (c) MSK-Bench Authors
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), Pierre Schumacher (schumacherpier@gmail.com), Cameron Berg (cam.h.berg@gmail.com)
================================================="""

import collections
import numpy as np
import mujoco
import os

from msk_bench.envs.msk.base_v0 import BaseV0
from msk_bench.utils import gym
from msk_bench.utils.quat_math import quat2mat


class ReachEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ["qpos", "qvel", "tip_pos", "reach_err"]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": 1.0,
        "bonus": 4.0,
        "penalty": 50,
        "act_reg": 1,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(
            model_path=model_path,
            obsd_model_path=obsd_model_path,
            seed=seed,
            env_credits=self.MYO_CREDIT,
        )
        self._setup(**kwargs)

    def _setup(
        self,
        target_reach_range: dict,
        joint_random_range: tuple = (0.0, 0.0),
        far_th=0.35,
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        **kwargs,
    ):
        self.far_th = far_th
        self.target_reach_range = target_reach_range
        self.joint_random_range = joint_random_range
        super()._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            sites=self.target_reach_range.keys(),
            **kwargs,
        )
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = self.sim.model.key_qvel[0]

        geom_1_indices = np.where(self.sim.model.geom_group == 1)
        self.sim.model.geom_rgba[geom_1_indices, 3] = 0
        self.sim.model.geom_rgba[self.sim.model.geom_name2id("terrain")][-1] = 0.0
        self.sim.model.geom_pos[self.sim.model.geom_name2id("terrain")] = np.array(
            [0, 0, -10]
        )

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict["time"] = np.array([sim.data.time])
        obs_dict["qpos"] = sim.data.qpos[:].copy()
        obs_dict["qvel"] = sim.data.qvel[:].copy() * self.dt
        if sim.model.na > 0:
            obs_dict["act"] = sim.data.act[:].copy()

        obs_dict["tip_pos"] = np.array([])
        obs_dict["target_pos"] = np.array([])
        for isite in range(len(self.tip_sids)):
            obs_dict["tip_pos"] = np.append(
                obs_dict["tip_pos"], sim.data.site_xpos[self.tip_sids[isite]].copy()
            )
            obs_dict["target_pos"] = np.append(
                obs_dict["target_pos"],
                sim.data.site_xpos[self.target_sids[isite]].copy(),
            )
        obs_dict["reach_err"] = np.array(obs_dict["target_pos"]) - np.array(
            obs_dict["tip_pos"]
        )
        return obs_dict

    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict["reach_err"], axis=-1)
        vel_dist = np.linalg.norm(obs_dict["qvel"], axis=-1)
        act_mag = (
            np.linalg.norm(self.obs_dict["act"], axis=-1) / self.sim.model.na
            if self.sim.model.na != 0
            else 0
        )
        far_th = (
            self.far_th * len(self.tip_sids)
            if np.squeeze(obs_dict["time"]) > 2 * self.dt
            else np.inf
        )
        near_th = len(self.tip_sids) * 0.050
        rwd_dict = collections.OrderedDict(
            (
                ("reach", 10.0 - 1.0 * reach_dist - 10.0 * vel_dist),
                (
                    "bonus",
                    1.0 * (reach_dist < 2 * near_th) + 1.0 * (reach_dist < near_th),
                ),
                ("act_reg", -100.0 * act_mag),
                ("penalty", -1.0 * (reach_dist > far_th)),
                ("sparse", -1.0 * reach_dist),
                ("solved", reach_dist < near_th),
                ("done", reach_dist > far_th),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict

    def generate_targets(self):
        for site, span in self.target_reach_range.items():
            sid = self.sim.model.site_name2id(site)
            sid_target = self.sim.model.site_name2id(site + "_target")
            self.sim.model.site_pos[sid_target] = self.sim.data.site_xpos[
                sid
            ].copy() + self.np_random.uniform(low=span[0], high=span[1])
        self.sim.forward()

    def generate_qpos(self):
        qpos_rand = self.np_random.uniform(
            low=self.joint_random_range[0],
            high=self.joint_random_range[1],
            size=self.init_qpos.shape,
        )
        qpos_new = self.init_qpos.copy()
        qpos_new[self.sim.model.jnt_qposadr] += qpos_rand[
            self.sim.model.jnt_qposadr
        ]
        qpos_new[self.sim.model.jnt_qposadr] = np.clip(
            qpos_new[self.sim.model.jnt_qposadr],
            self.sim.model.jnt_range[:, 0],
            self.sim.model.jnt_range[:, 1],
        )
        return qpos_new

    def reset(self, **kwargs):
        if np.ptp(self.joint_random_range) > 0:
            self.sim.data.qpos = self.generate_qpos()
        self.sim.forward()
        self.generate_targets()
        self.robot.sync_sims(self.sim, self.sim_obsd)
        if np.ptp(self.joint_random_range) > 0:
            obs = super().reset(reset_qpos=self.generate_qpos(), **kwargs)
        else:
            obs = super().reset(**kwargs)
        return obs


# ==================================================

# ==================================================
class ReferenceMotion:
    def __init__(self, motion_file, dt=0.01):
        print(f"Loading Reference Motion: {motion_file}")
        self.dt = dt


        self.qpos_data = np.load(motion_file)
        self.n_frames = self.qpos_data.shape[0]
        self.duration = self.n_frames * dt






        root_pos = self.qpos_data[:, 0:3]
        self.root_lin_vel = np.gradient(root_pos, axis=0) / dt


        joint_pos = self.qpos_data[:, 7:]
        self.joint_vel = np.gradient(joint_pos, axis=0) / dt

        print(f"Reference loaded: {self.n_frames} frames, {self.duration:.2f}s duration.")

    def get_reference_state(self, time):
        """MSK-Bench task environment."""
        phase_time = time % self.duration
        frame_idx = int(phase_time / self.dt)
        frame_idx = min(frame_idx, self.n_frames - 1)

        return (
            self.qpos_data[frame_idx],      # qpos
            self.root_lin_vel[frame_idx],
            self.joint_vel[frame_idx]
        )


class WalkEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "height", "feet_rel_positions",
        "phase_var", "muscle_length", "muscle_velocity", "muscle_force",
        "qpos_error",
        "act",
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "vel_dir_bonus": 5.0,
        "tracking_vel": 10.0,
        "tracking_joint": 3.0,
        "upright_reward": 6.0,
        "alive": 1.0,
        "act_reg": 0,
        "done": -100.0,
        "face_front": 8.0,
        "anti_hunch": 8.0,
    }

    # ============================================================

    # ============================================================
    SAFE_OBS_CLIP = 1e4
    SAFE_RWD_CLIP = 1e4
    SAFE_EXP_CLIP = 60.0
    SAFE_ACT_CLIP = 1.0
    SAFE_EPS = 1e-8

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        super().__init__(
            model_path=model_path,
            obsd_model_path=obsd_model_path,
            seed=seed,
            env_credits=self.MYO_CREDIT,
        )

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        motion_path = os.path.join(curr_dir, "clean_walk.npy")

        if os.path.exists(motion_path):
            self.ref_motion = ReferenceMotion(motion_path, dt=1.0 / 30.0)
        else:
            print(f"Warning: reference motion {motion_path} was not found; using the default static reference.")
            self.ref_motion = None

        self._setup(**kwargs)

    def _setup(
        self,
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        min_height=0.5,
        max_rot=0.8,
        reset_type="init",
        **kwargs,
    ):
        self.min_height = min_height
        self.max_rot = max_rot
        self.reset_type = reset_type

        self._numerical_error = False
        self._last_step_return_len = 5

        self.torso_id = self.sim.model.body_name2id("torso")
        self.pelvis_id = self.sim.model.body_name2id("pelvis")

        super()._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            **kwargs,
        )

    # ============================================================

    # ============================================================
    def _mark_bad_number(self, name=""):
        self._numerical_error = True

        # print(f"[WalkEnvV0] numerical issue detected: {name}")

    def _safe_scalar(self, x, default=0.0, clip=None, name=""):
        if clip is None:
            clip = self.SAFE_RWD_CLIP
        try:
            x = float(np.asarray(x).reshape(-1)[0])
        except Exception:
            self._mark_bad_number(name)
            return float(default)
        if not np.isfinite(x):
            self._mark_bad_number(name)
            return float(default)
        return float(np.clip(x, -clip, clip))

    def _safe_array(self, x, default=0.0, clip=None, name=""):
        if clip is None:
            clip = self.SAFE_OBS_CLIP
        try:
            arr = np.asarray(x, dtype=np.float64).copy()
        except Exception:
            self._mark_bad_number(name)
            return np.asarray(default, dtype=np.float64)

        if not np.all(np.isfinite(arr)):
            self._mark_bad_number(name)

        arr = np.nan_to_num(arr, nan=default, posinf=clip, neginf=-clip)
        arr = np.clip(arr, -clip, clip)
        return arr

    def _fit_array_like(self, x, target, default=0.0, clip=None, name=""):
        """MSK-Bench task environment."""
        target = np.asarray(target)
        out = np.full(target.shape, default, dtype=np.float64)
        arr = self._safe_array(x, default=default, clip=clip, name=name).reshape(-1)
        n = min(out.size, arr.size)
        if n > 0:
            out.reshape(-1)[:n] = arr[:n]
        return self._safe_array(out, default=default, clip=clip, name=name)

    def _safe_exp(self, x):
        x = self._safe_scalar(x, default=-self.SAFE_EXP_CLIP, clip=self.SAFE_EXP_CLIP, name="exp_input")
        return float(np.exp(np.clip(x, -self.SAFE_EXP_CLIP, self.SAFE_EXP_CLIP)))

    def _safe_norm(self, x, default=0.0, clip=None, name="norm"):
        arr = self._safe_array(x, default=0.0, clip=clip, name=name)
        val = np.linalg.norm(arr)
        return self._safe_scalar(val, default=default, clip=self.SAFE_RWD_CLIP, name=name)

    def _safe_quat_to_mat(self, quat):
        quat = self._safe_array(quat, default=0.0, clip=1.0, name="quat").reshape(-1)
        if quat.size < 4:
            self._mark_bad_number("quat_size")
            return np.eye(3)
        quat = quat[:4]
        qn = np.linalg.norm(quat)
        if (not np.isfinite(qn)) or qn < self.SAFE_EPS:
            self._mark_bad_number("quat_norm")
            return np.eye(3)
        quat = quat / qn
        try:
            mat = quat2mat(quat)
            return self._safe_array(mat, default=0.0, clip=1.0, name="quat2mat")
        except Exception:
            self._mark_bad_number("quat2mat_exception")
            return np.eye(3)

    def _sanitize_obs_dict(self, obs_dict):
        clean = {}
        for k, v in obs_dict.items():
            clean[k] = self._safe_array(v, default=0.0, clip=self.SAFE_OBS_CLIP, name=f"obs/{k}")
        return clean

    def _sanitize_obs_like(self, obs):
        if isinstance(obs, dict):
            return self._sanitize_obs_dict(obs)
        if isinstance(obs, tuple):
            return tuple(self._sanitize_obs_like(x) if i == 0 else x for i, x in enumerate(obs))
        return self._safe_array(obs, default=0.0, clip=self.SAFE_OBS_CLIP, name="obs")

    def _sanitize_reward_dict(self, rwd_dict):
        clean = collections.OrderedDict()
        for k, v in rwd_dict.items():
            if k == "solved":
                clean[k] = bool(v)
            else:
                clean[k] = self._safe_scalar(v, default=0.0, clip=self.SAFE_RWD_CLIP, name=f"reward/{k}")
        return clean

    def _sim_has_bad_numbers(self):
        try:
            arrays = [
                self.sim.data.qpos,
                self.sim.data.qvel,
                self.sim.data.body_xpos,
                self.sim.data.body_xquat,
            ]
            if self.sim.model.na > 0:
                arrays.extend([
                    self.sim.data.act,
                    self.sim.data.actuator_length,
                    self.sim.data.actuator_velocity,
                    self.sim.data.actuator_force,
                ])
            return any(not np.all(np.isfinite(a)) for a in arrays)
        except Exception:
            return True

    def _safe_action(self, action):
        arr = self._safe_array(action, default=0.0, clip=self.SAFE_ACT_CLIP, name="action")
        try:
            low = np.asarray(self.action_space.low, dtype=np.float64)
            high = np.asarray(self.action_space.high, dtype=np.float64)
            low = np.nan_to_num(low, nan=-self.SAFE_ACT_CLIP, posinf=self.SAFE_ACT_CLIP, neginf=-self.SAFE_ACT_CLIP)
            high = np.nan_to_num(high, nan=self.SAFE_ACT_CLIP, posinf=self.SAFE_ACT_CLIP, neginf=-self.SAFE_ACT_CLIP)
            arr = np.clip(arr, low, high)
        except Exception:
            arr = np.clip(arr, -self.SAFE_ACT_CLIP, self.SAFE_ACT_CLIP)
        return arr

    # ============================================================

    # ============================================================
    def reset(self, **kwargs):
        self._numerical_error = False

        try:
            if self.ref_motion is not None:
                duration = self._safe_scalar(self.ref_motion.duration, default=1.0, clip=1e6, name="ref_duration")
                duration = max(duration, self.SAFE_EPS)
                self.start_time_offset = self._safe_scalar(
                    self.np_random.uniform(0, duration),
                    default=0.0,
                    clip=duration,
                    name="start_time_offset",
                )
                ref_qpos, ref_root_vel, ref_joint_vel = self.ref_motion.get_reference_state(self.start_time_offset)

                init_qpos = self._fit_array_like(ref_qpos, self.sim.model.key_qpos[0], name="reset_qpos")
                init_qpos[0] = 0.0

                init_qvel = np.zeros_like(self.sim.data.qvel, dtype=np.float64)
                ref_root_vel = self._safe_array(ref_root_vel, default=0.0, clip=self.SAFE_OBS_CLIP, name="reset_root_vel")
                ref_joint_vel = self._safe_array(ref_joint_vel, default=0.0, clip=self.SAFE_OBS_CLIP, name="reset_joint_vel")

                init_qvel[0:min(3, len(init_qvel), len(ref_root_vel))] = ref_root_vel[:min(3, len(init_qvel), len(ref_root_vel))]
                num_joints = min(len(ref_joint_vel), len(init_qvel) - 6)
                if num_joints > 0:
                    init_qvel[6:6 + num_joints] = ref_joint_vel[:num_joints]
                init_qvel = self._fit_array_like(init_qvel, self.sim.data.qvel, name="reset_qvel")
            else:
                self.start_time_offset = 0.0
                init_qpos = self._fit_array_like(self.sim.model.key_qpos[0], self.sim.data.qpos, name="reset_key_qpos")
                init_qvel = self._fit_array_like(self.sim.model.key_qvel[0], self.sim.data.qvel, name="reset_key_qvel")

            self.robot.sync_sims(self.sim, self.sim_obsd)
            obs = super().reset(reset_qpos=init_qpos, reset_qvel=init_qvel, **kwargs)
            return self._sanitize_obs_like(obs)

        except Exception as e:

            self._mark_bad_number(f"reset_exception:{e}")
            init_qpos = self._fit_array_like(self.sim.model.key_qpos[0], self.sim.data.qpos, name="fallback_qpos")
            init_qvel = np.zeros_like(self.sim.data.qvel, dtype=np.float64)
            obs = super().reset(reset_qpos=init_qpos, reset_qvel=init_qvel, **kwargs)
            return self._sanitize_obs_like(obs)

    def step(self, *args, **kwargs):
        self._numerical_error = False


        if len(args) > 0:
            args = (self._safe_action(args[0]),) + args[1:]
        elif "action" in kwargs:
            kwargs["action"] = self._safe_action(kwargs["action"])
        elif "a" in kwargs:
            kwargs["a"] = self._safe_action(kwargs["a"])

        try:
            out = super().step(*args, **kwargs)
            if isinstance(out, tuple):
                self._last_step_return_len = len(out)

                if len(out) == 5:
                    obs, reward, terminated, truncated, info = out
                    obs = self._sanitize_obs_like(obs)
                    reward = self._safe_scalar(reward, default=0.0, clip=self.SAFE_RWD_CLIP, name="step_reward")
                    if self._numerical_error or self._sim_has_bad_numbers():
                        terminated = True
                        reward = min(reward, -100.0)
                        info = dict(info) if isinstance(info, dict) else {}
                        info["numerical_error"] = True
                    return obs, reward, bool(terminated), bool(truncated), info

                if len(out) == 4:
                    obs, reward, done, info = out
                    obs = self._sanitize_obs_like(obs)
                    reward = self._safe_scalar(reward, default=0.0, clip=self.SAFE_RWD_CLIP, name="step_reward")
                    if self._numerical_error or self._sim_has_bad_numbers():
                        done = True
                        reward = min(reward, -100.0)
                        info = dict(info) if isinstance(info, dict) else {}
                        info["numerical_error"] = True
                    return obs, reward, bool(done), info

            return out

        except Exception as e:
            self._mark_bad_number(f"step_exception:{e}")
            obs = self.reset()
            info = {"numerical_error": True, "exception": repr(e)}
            if self._last_step_return_len == 4:
                return obs, -100.0, True, info
            return obs, -100.0, True, False, info

    # ============================================================

    # ============================================================
    def get_obs_dict(self, sim):
        obs_dict = {}

        obs_dict["t"] = np.array([self._safe_scalar(sim.data.time, default=0.0, clip=1e6, name="time")])
        obs_dict["time"] = obs_dict["t"].copy()

        obs_dict["qpos"] = self._safe_array(sim.data.qpos.copy(), name="qpos")
        obs_dict["qpos_without_xy"] = self._safe_array(sim.data.qpos[2:].copy(), name="qpos_without_xy")
        obs_dict["qvel"] = self._safe_array(sim.data.qvel[:].copy(), name="qvel")
        obs_dict["com_vel"] = self._safe_array(self._get_com_velocity().copy(), name="com_vel")
        obs_dict["torso_angle"] = self._safe_array(self._get_torso_angle().copy(), clip=1.0, name="torso_angle")
        obs_dict["feet_heights"] = self._safe_array(self._get_feet_heights().copy(), name="feet_heights")
        obs_dict["height"] = self._safe_array(np.array([self._get_height()]), name="height")
        obs_dict["feet_rel_positions"] = self._safe_array(self._get_feet_relative_position().copy(), name="feet_rel_positions")

        if sim.model.na > 0:
            obs_dict["act"] = self._safe_array(sim.data.act[:].copy(), clip=self.SAFE_ACT_CLIP, name="act")
            obs_dict["muscle_length"] = self.muscle_lengths()
            obs_dict["muscle_velocity"] = self.muscle_velocities()
            obs_dict["muscle_force"] = self.muscle_forces()
        else:
            obs_dict["act"] = np.zeros(0, dtype=np.float64)
            obs_dict["muscle_length"] = np.zeros(0, dtype=np.float64)
            obs_dict["muscle_velocity"] = np.zeros(0, dtype=np.float64)
            obs_dict["muscle_force"] = np.zeros(0, dtype=np.float64)

        duration = self.ref_motion.duration if self.ref_motion else 1.0
        duration = max(self._safe_scalar(duration, default=1.0, clip=1e6, name="phase_duration"), self.SAFE_EPS)
        phase = (self._safe_scalar(sim.data.time, default=0.0, clip=1e6, name="phase_time") / duration) % 1.0
        obs_dict["phase_var"] = np.array([phase], dtype=np.float64)

        qpos_err_len = max(int(sim.model.nq) - 7, 0)
        qpos_error = np.zeros(qpos_err_len, dtype=np.float64)
        if self.ref_motion is not None:
            try:
                current_time = self._safe_scalar(sim.data.time, default=0.0, clip=1e6, name="ref_time") + getattr(self, "start_time_offset", 0.0)
                ref_qpos, _, _ = self.ref_motion.get_reference_state(current_time)
                current_joints = self._safe_array(sim.data.qpos[7:], name="current_joints")
                target_joints = self._safe_array(ref_qpos[7:], name="target_joints")
                min_len = min(len(current_joints), len(target_joints), qpos_err_len)
                if min_len > 0:
                    qpos_error[:min_len] = target_joints[:min_len] - current_joints[:min_len]
            except Exception:
                self._mark_bad_number("qpos_error")
        obs_dict["qpos_error"] = self._safe_array(qpos_error, name="qpos_error")

        return self._sanitize_obs_dict(obs_dict)

    # ============================================================

    # ============================================================
    def get_reward_dict(self, obs_dict):
        current_time = self._safe_scalar(self.sim.data.time, default=0.0, clip=1e6, name="rwd_time") + getattr(self, "start_time_offset", 0.0)

        if self.ref_motion:
            try:
                ref_qpos, ref_root_vel, _ = self.ref_motion.get_reference_state(current_time)
                ref_qpos = self._safe_array(ref_qpos, name="rwd_ref_qpos")
                target_root_vel_xy = self._safe_array(ref_root_vel[:2], name="target_root_vel_xy")
            except Exception:
                self._mark_bad_number("ref_motion_reward")
                ref_qpos = self._safe_array(self.sim.data.qpos.copy(), name="fallback_ref_qpos")
                target_root_vel_xy = np.array([1.0, 0.0], dtype=np.float64)
        else:
            ref_qpos = self._safe_array(self.sim.data.qpos.copy(), name="self_qpos")
            target_root_vel_xy = np.array([1.0, 0.0], dtype=np.float64)

        curr_root_vel = self._safe_array(self._get_com_velocity(), name="curr_root_vel")

        mat = self._safe_quat_to_mat(self._get_torso_angle())
        head_dir = self._safe_array(mat @ np.array([0.0, 0.0, 1.0]), clip=1.0, name="head_dir")
        face_dir = self._safe_array(mat @ np.array([1.0, 0.0, 0.0]), clip=1.0, name="face_dir")


        facing_front_score = self._safe_scalar(face_dir[0], default=0.0, clip=1.0, name="facing_front_score")
        r_face_front = self._safe_exp(-5.0 * (1.0 - facing_front_score))
        if facing_front_score < 0.7:
            r_face_front *= 0.1
        r_face_front = self._safe_scalar(r_face_front, default=0.0, clip=1.0, name="r_face_front")


        z_project = self._safe_scalar(head_dir[2], default=0.0, clip=1.0, name="z_project")
        r_upright = self._safe_exp(-10.0 * (1.0 - z_project))

        pitch_val = self._safe_scalar(face_dir[2], default=0.0, clip=1.0, name="pitch_val")
        r_pitch = 1.0
        if pitch_val < -0.1:
            r_pitch = self._safe_exp(-20.0 * (pitch_val + 0.1) ** 2)

        torso_z = self._safe_scalar(self.sim.data.body_xpos[self.torso_id][2], default=0.0, clip=self.SAFE_OBS_CLIP, name="torso_z")
        if torso_z < 1.15:
            r_upright = 0.0


        vel_dir_reward = 0.0
        target_speed = self._safe_norm(target_root_vel_xy, default=0.0, clip=self.SAFE_OBS_CLIP, name="target_speed")
        dot_prod = 0.0
        if target_speed > 0.01:
            dot_prod = self._safe_scalar(np.dot(curr_root_vel, target_root_vel_xy), default=0.0, clip=self.SAFE_RWD_CLIP, name="dot_prod")
            vel_dir_reward = 1.0 if dot_prod > 0.0 else -1.0

        vel_diff = curr_root_vel - target_root_vel_xy
        vel_diff_sq = self._safe_scalar(np.sum(vel_diff ** 2), default=self.SAFE_RWD_CLIP, clip=self.SAFE_RWD_CLIP, name="vel_diff_sq")
        r_tracking_vel = self._safe_exp(-2.0 * vel_diff_sq)
        if dot_prod < 0.0:
            r_tracking_vel *= 0.1
        r_tracking_vel = self._safe_scalar(r_tracking_vel, default=0.0, clip=1.0, name="r_tracking_vel")


        curr_joints = self._safe_array(self.sim.data.qpos[7:], name="curr_joints_rwd")
        ref_joints = self._safe_array(ref_qpos[7:], name="ref_joints_rwd")
        min_j_len = min(len(curr_joints), len(ref_joints))
        if min_j_len > 0:
            joint_diff = curr_joints[:min_j_len] - ref_joints[:min_j_len]
            joint_diff_sq = self._safe_scalar(np.sum(joint_diff ** 2), default=self.SAFE_RWD_CLIP, clip=self.SAFE_RWD_CLIP, name="joint_diff_sq")
            r_tracking_joint = self._safe_exp(-2.0 * joint_diff_sq)
        else:
            r_tracking_joint = 0.0
        r_tracking_joint = self._safe_scalar(r_tracking_joint, default=0.0, clip=1.0, name="r_tracking_joint")

        act_mag = 0.0
        if self.sim.model.na != 0:
            act = self._safe_array(obs_dict.get("act", np.zeros(self.sim.model.na)), default=0.0, clip=self.SAFE_ACT_CLIP, name="act_rwd")
            act_mag = self._safe_norm(act, default=0.0, clip=self.SAFE_ACT_CLIP, name="act_mag") / max(int(self.sim.model.na), 1)
            act_mag = self._safe_scalar(act_mag, default=0.0, clip=self.SAFE_RWD_CLIP, name="act_mag_scalar")

        done_val = self._get_done()
        done_val = self._safe_scalar(done_val, default=1.0, clip=1.0, name="done_val")

        rwd_dict = collections.OrderedDict((
            ("tracking_joint", r_tracking_joint),
            ("tracking_vel", r_tracking_vel),
            ("vel_dir_bonus", vel_dir_reward),
            ("upright_reward", r_upright),
            ("face_front", r_face_front),
            ("anti_hunch", r_pitch),
            ("alive", 1.0),
            ("act_reg", -1.0 * act_mag),
            ("done", done_val),
            ("sparse", r_tracking_joint),
            ("solved", bool(r_tracking_joint > 0.8)),
        ))

        dense = 0.0
        for key, wt in self.rwd_keys_wt.items():
            val = rwd_dict.get(key, 0.0)
            if key != "solved":
                dense += self._safe_scalar(wt, default=0.0, clip=self.SAFE_RWD_CLIP, name=f"wt/{key}") * self._safe_scalar(val, default=0.0, clip=self.SAFE_RWD_CLIP, name=f"dense/{key}")

        if dot_prod < -0.1:
            dense -= 2.0


        if self._numerical_error or self._sim_has_bad_numbers():
            rwd_dict["done"] = 1.0
            dense = min(dense, -100.0)
            rwd_dict["solved"] = False

        rwd_dict["dense"] = self._safe_scalar(dense, default=-100.0, clip=self.SAFE_RWD_CLIP, name="dense")
        return self._sanitize_reward_dict(rwd_dict)

    # ============================================================

    # ============================================================
    def muscle_lengths(self):
        return self._safe_array(self.sim.data.actuator_length, default=0.0, clip=self.SAFE_OBS_CLIP, name="muscle_length")

    def muscle_forces(self):
        force = self._safe_array(self.sim.data.actuator_force, default=0.0, clip=1e6, name="actuator_force")
        return self._safe_array(force / 1000.0, default=0.0, clip=100.0, name="muscle_force")

    def muscle_velocities(self):
        vel = self._safe_array(self.sim.data.actuator_velocity, default=0.0, clip=1e6, name="actuator_velocity")
        return self._safe_array(vel, default=0.0, clip=100.0, name="muscle_velocity")

    def _get_done(self):
        if self._numerical_error or self._sim_has_bad_numbers():
            return 1
        height = self._safe_scalar(self._get_height(), default=0.0, clip=self.SAFE_OBS_CLIP, name="done_height")
        if height < self.min_height:
            return 1
        if self._get_rot_condition():
            return 1
        return 0

    def _get_feet_heights(self):
        try:
            f_l = self.sim.model.body_name2id("talus_l")
            f_r = self.sim.model.body_name2id("talus_r")
            val = np.array([self.sim.data.body_xpos[f_l][2], self.sim.data.body_xpos[f_r][2]])
            return self._safe_array(val, default=0.0, clip=self.SAFE_OBS_CLIP, name="feet_heights")
        except Exception:
            return np.array([0.0, 0.0], dtype=np.float64)

    def _get_feet_relative_position(self):
        try:
            f_l = self.sim.model.body_name2id("talus_l")
            f_r = self.sim.model.body_name2id("talus_r")
            pelvis = self.pelvis_id
            val = np.array([
                self.sim.data.body_xpos[f_l] - self.sim.data.body_xpos[pelvis],
                self.sim.data.body_xpos[f_r] - self.sim.data.body_xpos[pelvis],
            ])
            return self._safe_array(val, default=0.0, clip=self.SAFE_OBS_CLIP, name="feet_rel_positions")
        except Exception:
            return np.zeros((2, 3), dtype=np.float64)

    def _get_torso_angle(self):
        try:
            quat = self._safe_array(self.sim.data.body_xquat[self.torso_id], default=0.0, clip=1.0, name="torso_quat")
            qn = np.linalg.norm(quat)
            if (not np.isfinite(qn)) or qn < self.SAFE_EPS:
                self._mark_bad_number("torso_quat_norm")
                return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            return quat / qn
        except Exception:
            self._mark_bad_number("torso_quat_exception")
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _get_com_velocity(self):
        return self._safe_array(self.sim.data.qvel[:2].copy(), default=0.0, clip=self.SAFE_OBS_CLIP, name="com_velocity")

    def _get_height(self):
        try:
            mass = self._safe_array(np.expand_dims(self.sim.model.body_mass, -1), default=0.0, clip=1e6, name="body_mass")
            com = self._safe_array(self.sim.data.xipos, default=0.0, clip=self.SAFE_OBS_CLIP, name="xipos")
            mass_sum = self._safe_scalar(np.sum(mass), default=0.0, clip=1e12, name="mass_sum")
            if mass_sum <= self.SAFE_EPS:
                self._mark_bad_number("mass_sum_zero")
                return 0.0
            height = (np.sum(mass * com, axis=0) / mass_sum)[2]
            return self._safe_scalar(height, default=0.0, clip=self.SAFE_OBS_CLIP, name="height")
        except Exception:
            self._mark_bad_number("height_exception")
            return 0.0

    def _get_rot_condition(self):
        quat = self._safe_array(self.sim.data.qpos[3:7].copy(), default=0.0, clip=1.0, name="root_quat")
        mat = self._safe_quat_to_mat(quat)
        z_axis = self._safe_array(mat @ np.array([0.0, 0.0, 1.0]), default=0.0, clip=1.0, name="root_z_axis")
        z_val = abs(self._safe_scalar(z_axis[2], default=0.0, clip=1.0, name="root_z_val"))
        return 1 if z_val < self.max_rot else 0



class TerrainEnvV0(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy",
        "qvel",
        "com_vel",
        "torso_angle",
        "feet_heights",
        "height",
        "feet_rel_positions",
        "phase_var",
        "muscle_length",
        "muscle_velocity",
        "muscle_force",
    ]


    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "vel_reward": 6.0,
        "height_reward": 1.0,
        "upright_reward": 1.0,
        "heading_reward": 0.5,
        "lateral_penalty": -0.5,
        "energy_penalty": -1.0,
        "act_reg": 2.0,
        "cyclic_hip": 0.5,
        "ref_rot": 0.5,
        "joint_angle_rew": 0.5,
        "arm_stability": 0.2,
        "done": -50.0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        BaseV0.__init__(
            self,
            model_path=model_path,
            obsd_model_path=obsd_model_path,
            seed=seed,
            env_credits=self.MYO_CREDIT,
        )
        self._setup(**kwargs)

    def _setup(
        self,
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        min_height=0.8,
        max_rot=0.8,
        hip_period=100,
        reset_type="init",
        target_x_vel=0.0,
        target_y_vel=1.2,
        target_rot=None,
        terrain="rough",
        variant=None,
        **kwargs,
    ):
        self.min_height = min_height
        self.max_rot = max_rot
        self.hip_period = hip_period
        self.reset_type = reset_type
        self.target_x_vel = target_x_vel
        self.target_y_vel = target_y_vel
        self.target_rot = target_rot
        self.terrain = terrain
        self.variant = variant
        self.steps = 0

        self.arm_joint_indices = []
        for i in range(self.sim.model.njnt):
            try:
                name = mujoco.mj_id2name(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            except:
                try:
                    name = self.sim.model.id2name(i, 'joint')
                except:
                    name = None
            if name and any(part in name for part in ["shoulder", "elbow", "wrist", "arm", "cervical", "lumbar"]):
                self.arm_joint_indices.append(self.sim.model.jnt_qposadr[i])
        self.arm_joint_indices = np.array(self.arm_joint_indices, dtype=int)

        BaseV0._setup(
            self, obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, **kwargs
        )
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = 0.0

    def reset(self, **kwargs):
        self.steps = 0
        if self.terrain == "rough":
            rough = self.np_random.uniform(low=-0.5, high=0.5, size=(10000,))
            normalized_data = (rough - np.min(rough)) / (np.max(rough) - np.min(rough))
            scalar, offset = 0.08, 0.02
            self.sim.model.hfield_data[:] = normalized_data * scalar - offset

        elif self.terrain == "hilly":
            flat_length, frequency = 3000, 3
            scalar = (
                0.63
                if self.variant == "fixed"
                else self.np_random.uniform(low=0.53, high=0.73)
            )

            combined_data = np.concatenate(
                (
                    -2 * np.ones(flat_length),
                    -2
                    + 0.5
                    * (
                        np.sin(
                            np.linspace(0, frequency * np.pi, int(1e4 - flat_length))
                            + np.pi / 2
                        )
                        - 1
                    ),
                )
            )
            normalized_data = (combined_data - combined_data.min()) / (
                combined_data.max() - combined_data.min()
            )

            self.sim.model.hfield_data[:] = np.flip(
                normalized_data.reshape(100, 100) * scalar, [0, 1]
            ).reshape(
                10000,
            )

        elif self.terrain == "stairs":
            num_stairs = 12
            stair_height = 0.1
            flat = 5200 - (1e4 - 5200) % num_stairs
            stairs_width = (1e4 - flat) // num_stairs
            scalar = (
                2.5
                if self.variant == "fixed"
                else self.np_random.uniform(low=1.5, high=3.5)
            )

            stair_parts = [
                np.full((int(stairs_width // 100), 100), -2 + stair_height * j)
                for j in range(num_stairs)
            ]
            new_terrain_data = np.concatenate(
                [np.full((int(flat // 100), 100), -2)] + stair_parts, axis=0
            )

            normalized_data = (new_terrain_data + 2) / (2 + stair_height * num_stairs)
            self.sim.model.hfield_data[:] = np.flip(
                normalized_data.reshape(100, 100) * scalar, [0, 1]
            ).reshape(
                10000,
            )

        self.sim.model.geom_rgba[self.sim.model.geom_name2id("terrain")][-1] = 1.0
        self.sim.model.geom_pos[self.sim.model.geom_name2id("terrain")] = np.array(
            [0, 0, 0]
        )
        self.sim.model.geom_contype[self.sim.model.geom_name2id("terrain")] = 1
        self.sim.model.geom_conaffinity[self.sim.model.geom_name2id("terrain")] = 1

        if self.reset_type == "random":
            qpos, qvel = self.get_randomized_initial_state()
        elif self.reset_type == "init":
            qpos, qvel = self.sim.model.key_qpos[2], self.sim.model.key_qvel[2]
        else:
            qpos, qvel = self.sim.model.key_qpos[0], self.sim.model.key_qvel[0]
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = BaseV0.reset(self, reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        return obs

    def _get_done(self):
        height = self._get_height()
        if height < self.min_height:
            return 1
        if self._get_rot_condition():
            return 1
        if self._get_knee_condition():
            return 1
        return 0

    def _get_knee_condition(self):
        """
        Checks if the agent is on its knees by comparing the distance between the center of mass and the feet.
        """
        feet_heights = self._get_feet_heights()
        com_height = self._get_height()
        if com_height - np.mean(feet_heights) < 0.61:
            return 1
        else:
            return 0
