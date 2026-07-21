"""MSK-Bench task environments.

This module contains the 22 full-body musculoskeletal benchmark tasks
for the canonical MSK-Bench suite. Base walking primitives remain in walk_v0.py.
"""

import collections
import os
import traceback
import warnings
from collections import OrderedDict

import gymnasium as gymnasium
import mujoco
import numpy as np

try:
    import gymnasium as gym
except Exception:  # pragma: no cover
    from msk_bench.utils import gym

from msk_bench.envs.msk.benchmark.walk_v0 import WalkEnvV0
from msk_bench.rewards import PowerliftRewardConfig, compute_powerlift_reward
from msk_bench.utils.quat_math import euler2quat, quat2mat


class MSKBenchWalkEnvV0(WalkEnvV0):
    """MSK-Bench flat-ground walking task."""

    pass


class MSKBenchStandEnvV0(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "height", "muscle_length",
        "muscle_velocity", "muscle_force"
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {

        "pose": 10.0,
        "stillness": 15.0,
        "feet_still": 10.0,


        "balance": 10.0,
        "alive": 5.0,



        "act_reg": 0.01,
        "done": -100,
    }

    def _setup(
        self,
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        min_height=0.8,
        max_rot=0.8,
        reset_type="init",
        max_episode_steps=1000,
        **kwargs,
    ):
        self.min_height = min_height
        self.max_rot = max_rot
        self.reset_type = reset_type
        self.max_episode_steps = max_episode_steps
        self.steps = 0


        for k in ["target_reach_range", "far_th", "target_x_vel", "target_y_vel", "target_rot", "hip_period"]:
            if k in kwargs: del kwargs[k]


        self.ref_qpos = self.sim.model.qpos0.copy()

        super(WalkEnvV0, self)._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            sites=[],
            **kwargs
        )
        self.init_qpos[:] = self.sim.model.qpos0.copy()
        self.init_qvel[:] = 0.0

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict["t"] = np.array([sim.data.time])
        obs_dict["time"] = np.array([sim.data.time])
        obs_dict["qpos_without_xy"] = sim.data.qpos[2:].copy()
        obs_dict["qvel"] = sim.data.qvel[:].copy() * self.dt
        obs_dict["com_vel"] = np.array([self._get_com_velocity().copy()])
        obs_dict["torso_angle"] = np.array([self._get_torso_angle().copy()])
        obs_dict["feet_heights"] = self._get_feet_heights().copy()
        obs_dict["height"] = np.array([self._get_height()]).copy()
        obs_dict["muscle_length"] = self.muscle_lengths()
        obs_dict["muscle_velocity"] = self.muscle_velocities()
        obs_dict["muscle_force"] = self.muscle_forces()
        if sim.model.na > 0: obs_dict["act"] = sim.data.act[:].copy()
        return obs_dict

    def get_reward_dict(self, obs_dict):



        curr_qpos = self.sim.data.qpos[2:]
        ref_qpos = self.ref_qpos[2:]
        pose_dist = np.linalg.norm(curr_qpos - ref_qpos)
        pose_reward = np.exp(-1.0 * pose_dist)



        com_vel = self._get_com_velocity()
        com_vel_reward = np.exp(-10.0 * np.linalg.norm(com_vel))


        qvel = self.sim.data.qvel
        qvel_reward = np.exp(-0.1 * np.linalg.norm(qvel))

        stillness_reward = com_vel_reward * qvel_reward



        try:
            f_l_id = self.sim.model.body_name2id("talus_l")
            f_r_id = self.sim.model.body_name2id("talus_r")

            vel_l = self.sim.data.cvel[f_l_id][:3]
            vel_r = self.sim.data.cvel[f_r_id][:3]
            feet_vel_mag = np.linalg.norm(vel_l) + np.linalg.norm(vel_r)

            feet_still_reward = np.exp(-10.0 * feet_vel_mag)
        except:
            feet_still_reward = 0.0


        torso_quat = self._get_torso_angle()

        torso_up = (quat2mat(torso_quat) @ np.array([0, 0, 1]))[2]
        balance_reward = np.exp(-5.0 * max(0, 1.0 - torso_up))

        act_mag = np.linalg.norm(self.obs_dict["act"]) / self.sim.model.na if self.sim.model.na != 0 else 0
        done = self._get_done()

        rwd_dict = collections.OrderedDict(
            (
                ("pose", pose_reward),
                ("stillness", stillness_reward),
                ("feet_still", feet_still_reward),
                ("balance", balance_reward),
                ("alive", 1.0),
                ("act_reg", -1.0 * act_mag),
                ("done", 1.0 if done else 0.0),
                ("sparse", pose_reward * stillness_reward),
                ("solved", pose_dist < 0.2 and stillness_reward > 0.9),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict.get(key, 0.0) for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict

    def reset(self, **kwargs):
        self.steps = 0

        qpos = self.sim.model.qpos0.copy()
        qvel = np.zeros(self.sim.model.nv)


        qpos[:] += self.np_random.normal(0, 0.005, size=qpos.shape)
        qvel[:] += self.np_random.normal(0, 0.005, size=qvel.shape)

        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super(WalkEnvV0, self).reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        return obs

    def _get_done(self):
        if self.steps >= self.max_episode_steps: return 1


        height = self._get_height()
        if height < self.min_height: return 1


        quat = self.sim.data.qpos[3:7].copy()
        z_up = (quat2mat(quat) @ np.array([0, 0, 1]))[2]
        if z_up < self.max_rot: return 1

        return 0


    def _get_angle(self, names):
        vals = []
        for name in names:
            try:
                id = self.sim.model.joint_name2id(name)
                addr = self.sim.model.jnt_qposadr[id]
                vals.append(self.sim.data.qpos[addr])
            except: vals.append(0.0)
        return np.array(vals)

    def muscle_lengths(self): return self.sim.data.actuator_length
    def muscle_forces(self): return np.clip(self.sim.data.actuator_force / 1000, -100, 100)
    def muscle_velocities(self): return np.clip(self.sim.data.actuator_velocity, -100, 100)

    def _get_torso_angle(self):
        return self.sim.data.body_xquat[self.sim.model.body_name2id("torso")]

    def _get_com_velocity(self):
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        cvel = -self.sim.data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]

    def _get_height(self):
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        com = self.sim.data.xipos
        return (np.sum(mass * com, 0) / np.sum(mass))[2]

    def _get_feet_heights(self):
        try:
            f_l = self.sim.model.body_name2id("talus_l")
            f_r = self.sim.model.body_name2id("talus_r")
            return np.array([self.sim.data.body_xpos[f_l][2], self.sim.data.body_xpos[f_r][2]])
        except: return np.array([0.0, 0.0])


class MSKBenchPowerliftEnvV0(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "height", "muscle_length",
        "muscle_velocity", "muscle_force",
        "dumbbell_pos", "dumbbell_rel_pos", "bar_tilt",
        "feet_err", "act"
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "hold_bonus": 100.0,
        "overhead_height": 50.0,
        "torso_upright": 100.0,
        "spine_straight": 80.0,
        "lumbar_effort": 30.0,
        "feet_anchor": 40.0,
        "bar_level": 30.0,
        "balance": 20.0,
        "drop_penalty": -100.0,
        "bend_over_penalty": -200.0,
        "act_reg": 0.001,
        "alive": 5.0,
        "done": -100.0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, normalize_act=True, reset_type='none', render_mode=None, **kwargs):
        self._init_complete = False
        self.dumbbell_id = -1
        self.torso_id = -1
        self.head_id = -1
        self.hand_l_id = -1; self.hand_r_id = -1

        self.render_mode = render_mode
        if 'render_mode' in kwargs: del kwargs['render_mode']
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, normalize_act=normalize_act, **kwargs)
        self.reset_type = reset_type

    def _setup(self, obs_keys=DEFAULT_OBS_KEYS, weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS, **kwargs):
        self.max_episode_steps = kwargs.get('max_episode_steps', 1000)
        self.steps = 0

        super()._setup(obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, sites=[], **kwargs)


        try:
            self.dumbbell_id = self.sim.model.body_name2id("dumbbell")
            self.torso_id = self.sim.model.body_name2id("torso")
            self.head_id = self.sim.model.body_name2id("head")
            self.foot_l_id = self.sim.model.body_name2id("talus_l")
            self.foot_r_id = self.sim.model.body_name2id("talus_r")
        except: pass


        for j_name in ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation']:
            try:
                j_id = self.sim.model.joint_name2id(j_name)
                dof_adr = self.sim.model.jnt_dofadr[j_id]
                self.sim.model.dof_damping[dof_adr] = 5.0
                self.sim.model.dof_frictionloss[dof_adr] = 2.0
            except: pass


        power_keywords = ['lumbar', 'erector', 'glute', 'quad', 'hamstring']
        for i in range(self.sim.model.nu):
            name = self.sim.model.id2name(i, 'actuator')
            if any(s in name.lower() for s in power_keywords):
                self.sim.model.actuator_gainprm[i, 2] *= 2.5

        self._init_complete = True

    def reset(self, **kwargs):
        self.steps = 0
        qpos = self.sim.model.qpos0.copy()
        qvel = np.zeros(self.sim.model.nv)


        qpos[2] = 1.02

        self._set_joint(qpos, "lumbar_extension", -0.5)


        self._set_joint(qpos, "shoulder_elv", 3.0)
        self._set_joint(qpos, "shoulder_elv_l", 3.0)

        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel


        for i in range(self.sim.model.nu):
            name = self.sim.model.id2name(i, 'actuator')
            if name and 'lumbar' in name.lower():
                self.sim.data.act[i] = 0.8
                self.sim.data.ctrl[i] = 0.8

        self.sim.forward()


        for _ in range(25):
            mujoco.mj_step(self.sim.model.ptr, self.sim.data.ptr)

        if 'reset_qpos' in kwargs: del kwargs['reset_qpos']
        if 'reset_qvel' in kwargs: del kwargs['reset_qvel']

        ret = super(WalkEnvV0, self).reset(reset_qpos=self.sim.data.qpos.copy(), reset_qvel=self.sim.data.qvel.copy(), **kwargs)
        return ret if isinstance(ret, tuple) else (ret, {})

    def step(self, a):
        self.steps += 1
        return super().step(a)

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        obs_dict['act'] = sim.data.act.copy()


        bar_pos = sim.data.body_xpos[self.dumbbell_id].copy()
        torso_pos = sim.data.body_xpos[self.torso_id].copy()
        obs_dict["dumbbell_pos"] = bar_pos
        obs_dict["dumbbell_rel_pos"] = bar_pos - torso_pos


        bar_rot = quat2mat(sim.data.body_xquat[self.dumbbell_id])
        bar_y_axis = bar_rot @ np.array([0, 1, 0])
        obs_dict["bar_tilt"] = np.array([abs(bar_y_axis[2])])


        obs_dict["feet_err"] = np.zeros(6)
        return obs_dict

    def get_reward_dict(self, obs_dict):

        hold_bonus = 0.0; overhead_height = 0.0; spine_straight = 0.0
        torso_upright = 0.0; lumbar_effort = 0.0; bend_over_penalty = 0.0

        try:
            bar_h = self.sim.data.body_xpos[self.dumbbell_id][2]


            if bar_h > 1.6:
                overhead_height = 1.0
                if abs(self.sim.data.cvel[self.dumbbell_id][2]) < 0.1:
                    hold_bonus = 1.0


            lumbar_ext = self._get_joint_val("lumbar_extension")
            spine_straight = np.exp(-15.0 * abs(lumbar_ext - (-0.4)))


            if lumbar_ext > 0.1:
                bend_over_penalty = 1.0


            act = obs_dict.get("act", np.zeros(self.sim.model.nu))

            lumbar_effort = np.mean(act)


            z_up = (quat2mat(self.sim.data.body_xquat[self.torso_id]) @ np.array([0, 0, 1]))[2]
            torso_upright = np.exp(-20.0 * (1.0 - z_up))

        except: pass

        default_config = PowerliftRewardConfig()
        weight_kwargs = {
            key: float(self.rwd_keys_wt.get(key, getattr(default_config, key)))
            for key in default_config.__dataclass_fields__
        }
        reward_result = compute_powerlift_reward(
            hold_bonus=hold_bonus,
            overhead_height=overhead_height,
            torso_upright=torso_upright,
            spine_straight=spine_straight,
            lumbar_effort=lumbar_effort,
            bend_over_penalty=bend_over_penalty,
            activation=obs_dict.get("act", np.zeros(self.sim.model.nu)),
            done=bool(self._get_done()),
            config=PowerliftRewardConfig(**weight_kwargs),
        )
        rwd_dict = collections.OrderedDict(
            (key, reward_result.terms.get(key, 0.0))
            for key in (*self.rwd_keys_wt.keys(), "sparse", "solved")
        )
        rwd_dict["dense"] = reward_result.dense
        return rwd_dict

    def _get_done(self):
        if self.steps >= self.max_episode_steps: return 1

        if self._get_joint_val("lumbar_extension") > 0.5: return 1
        if self.sim.data.qpos[2] < 0.7: return 1
        return 0

    def _set_joint(self, qpos_arr, name, val):
        try:
            id = self.sim.model.joint_name2id(name)
            addr = self.sim.model.jnt_qposadr[id]
            qpos_arr[addr] = val
        except: pass

    def _get_joint_val(self, name):
        try:
            id = self.sim.model.joint_name2id(name)
            addr = self.sim.model.jnt_qposadr[id]
            return self.sim.data.qpos[addr]
        except: return 0.0


class MSKBenchSingleLegStandEnvV0(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "height", "muscle_length",
        "muscle_velocity", "muscle_force"
    ]


    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "com_balance": 500.0,
        "lifted_leg_height": 300.0,
        "ground_touch_penalty": -800.0,
        "stillness_reward": 400.0,
        "torso_upright": 200.0,
        "side_lean_penalty": -200.0,
        "act_reg": -2.0,
        "act_rate_penalty": -1.0,
        "alive": 100.0,
        "done": -1500.0,
        "sparse": 0.0,
        "solved": 0.0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, normalize_act=True, reset_type='init', render_mode=None, max_episode_steps=1000, **kwargs):
        self._in_setup = True
        self.steps = 0
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.last_action = None

        if 'render_mode' in kwargs: del kwargs['render_mode']

        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, normalize_act=normalize_act, **kwargs)
        self.reset_type = reset_type

        self._set_flamingo_pose()

        self._in_setup = False
        print("MSK-Bench task message.")

    def _set_flamingo_pose(self):
        prep_qpos = self.sim.model.qpos0.copy()
        prep_qpos[2] = 0.95

        target_joints = {
            "hip_flexion_r": 0.0, "knee_angle_r": 0.0, "ankle_angle_r": 0.0,
            "hip_flexion_l": 1.0, "knee_angle_l": 1.5, "ankle_angle_l": -0.2,
            "lumbar_extension": 0.0,
            "shoulder_add_l": -0.5, "shoulder_add_r": -0.5,
            "elbow_flex_l": 0.5, "elbow_flex_r": 0.5,
        }

        for joint_name, target_angle in target_joints.items():
            try:
                joint_id = self.sim.model.joint_name2id(joint_name)
                qpos_adr = self.sim.model.jnt_qposadr[joint_id]
                prep_qpos[qpos_adr] = target_angle
            except Exception: pass

        self.init_qpos = prep_qpos

    def _setup(self, obs_keys=DEFAULT_OBS_KEYS, weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS, **kwargs):
        self._in_setup = True

        def _get_id_safe(names, type='body'):
            for name in names:
                try:
                    if type == 'body': return self.sim.model.body_name2id(name)
                except: pass
            return -1

        self.head_id = _get_id_safe(["head", "cervical"])
        self.pelvis_id = _get_id_safe(["pelvis"])
        self.foot_l_id = _get_id_safe(["talus_l", "calcn_l", "foot_l"])
        self.foot_r_id = _get_id_safe(["talus_r", "calcn_r", "foot_r"])

        super()._setup(obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, sites=[], **kwargs)
        self._in_setup = False

    def reset(self, **kwargs):
        self.steps = 0
        self.last_action = None

        if 'seed' in kwargs: del kwargs['seed']
        if 'options' in kwargs: del kwargs['options']

        ret = super().reset(**kwargs)
        obs = ret[0] if isinstance(ret, tuple) else ret
        info = ret[1] if isinstance(ret, tuple) else {}

        if self.reset_type == 'init' and hasattr(self, 'init_qpos'):
            noise = np.random.normal(0, 0.01, size=self.init_qpos[7:].shape)
            new_qpos = self.init_qpos.copy()
            new_qpos[7:] += noise

            self.sim.data.qpos[:] = new_qpos
            self.sim.data.qvel[:] = 0.0
            self.sim.forward()

            import mujoco
            for _ in range(20):
                mujoco.mj_step(self.sim.model.ptr, self.sim.data.ptr)

            if hasattr(self, 'robot') and hasattr(self.robot, 'sync_sims'):
                self.robot.sync_sims(self.sim, self.sim_obsd)
            obs = self.get_obs()

        return obs, info

    def step(self, a):

        if getattr(self, '_in_setup', True):
            try:
                obs = self.get_obs()
            except Exception:
                obs_shape = self.observation_space.shape[0] if hasattr(self, 'observation_space') else 1932
                obs = np.zeros(obs_shape, dtype=np.float32)
            return obs, 0.0, False, False, {}

        self.steps += 1
        a = np.nan_to_num(a, nan=0.0)
        a_clipped = np.clip(a, -1.0, 1.0)

        try:
            ret = super().step(a_clipped)
            if len(ret) == 5:
                obs_vec, reward, done, truncated, info = ret
            else:
                obs_vec, reward, done, info = ret
                truncated = False
        except Exception as e:
            obs_shape = self.observation_space.shape[0] if hasattr(self, 'observation_space') else 1932
            obs = np.zeros(obs_shape, dtype=np.float32)
            safe_info = {f"rwd_{k}": 0.0 for k in self.DEFAULT_RWD_KEYS_AND_WEIGHTS.keys()}
            safe_info.update({"rwd_dense": -500.0, "error_flag": 1.0})
            return obs, -500.0, True, False, safe_info

        self.obs_dict = self.get_obs_dict(self.sim)
        self.obs_dict['current_action'] = a_clipped

        rwd_dict = self.get_reward_dict(self.obs_dict)
        self.last_action = a_clipped

        for k, v in rwd_dict.items():
            info['rwd_' + k] = v

        final_obs = self.get_obs()
        safe_reward = float(np.clip(rwd_dict['dense'], -3000.0, 3000.0))
        final_done = bool(self._get_done())

        return final_obs, safe_reward, final_done, bool(truncated), info

    def _get_spine_vector(self):
        try:
            p_pos = self.sim.data.body_xpos[self.pelvis_id]
            h_pos = self.sim.data.body_xpos[self.head_id]
            vec = h_pos - p_pos
            norm = np.linalg.norm(vec)
            if norm > 1e-6: return vec / norm
        except: pass
        return np.array([0.0, 0.0, 1.0])

    def get_reward_dict(self, obs_dict):
        com_balance = 0.0
        lifted_leg_height = 0.0
        ground_touch_penalty = 0.0
        stillness_reward = 0.0
        torso_upright = 0.0
        side_lean_penalty = 0.0

        try:
            pelvis_pos = self.sim.data.body_xpos[self.pelvis_id]
            f_l_pos = self.sim.data.body_xpos[self.foot_l_id]
            f_r_pos = self.sim.data.body_xpos[self.foot_r_id]

            com_dist = np.linalg.norm(pelvis_pos[:2] - f_r_pos[:2])
            com_balance = np.exp(-15.0 * com_dist**2)

            lifted_z = f_l_pos[2]
            lifted_leg_height = np.clip(lifted_z / 0.3, 0.0, 1.0)

            if lifted_z < 0.05: ground_touch_penalty = 1.0

            vel_penalty = np.sum(np.square(self.sim.data.qvel[:3]))
            stillness_reward = np.exp(-5.0 * vel_penalty)

            spine_vec = self._get_spine_vector()
            z_up = spine_vec[2]
            torso_upright = np.exp(-10.0 * (1.0 - z_up)**2)
            side_lean_penalty = abs(spine_vec[1])

        except Exception: pass

        curr_act = obs_dict.get('current_action', np.zeros(self.sim.model.nu))
        act_mag = np.mean(np.square(curr_act))
        act_rate = np.mean(np.square(curr_act - self.last_action)) if self.last_action is not None else 0.0

        is_done = self._get_done()

        rwd_dict = collections.OrderedDict((
            ("com_balance", float(com_balance)),
            ("lifted_leg_height", float(lifted_leg_height)),
            ("ground_touch_penalty", float(ground_touch_penalty)),
            ("stillness_reward", float(stillness_reward)),
            ("torso_upright", float(torso_upright)),
            ("side_lean_penalty", float(side_lean_penalty)),
            ("act_reg", float(act_mag)),
            ("act_rate_penalty", float(act_rate)),
            ("alive", 1.0),
            ("done", float(1.0 if is_done else 0.0)),
            ("sparse", 0.0),
            ("solved", 0.0),
        ))

        rwd_dict["dense"] = np.sum([self.rwd_keys_wt.get(key, 0.0) * rwd_dict.get(key, 0.0) for key in self.rwd_keys_wt.keys()])
        return rwd_dict

    def _get_done(self):
        if getattr(self, '_in_setup', True): return False
        try:
            if not np.isfinite(self.sim.data.qpos).all() or not np.isfinite(self.sim.data.qvel).all(): return True
            if self.steps >= self.max_episode_steps: return True

            pelvis_pos = self.sim.data.body_xpos[self.pelvis_id]
            if pelvis_pos[2] < 0.60: return True

            spine_vec = self._get_spine_vector()
            if spine_vec[2] < 0.4: return True

            f_l_z = self.sim.data.body_xpos[self.foot_l_id][2]
            if f_l_z < 0.02 and self.steps > 10: return True
        except Exception: return True
        return False


class MSKBenchSitEnvV0(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "height", "muscle_length",
        "muscle_velocity", "muscle_force",
        "chair_rel_pos"
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {

        "sit_approach": 10.0,


        "sit_contact": 30.0,
        "sit_center": 30.0,

        "leg_geometry": 40.0,
        "feet_under_knee": 30.0,

        "upright_strict": 20.0,
        "feet_plant": 10.0,
        "sit_heading": 10.0,
        "soft_landing": 5.0,
        "arm_reg": 5.0,
        "stillness": 10.0,

        "alive": 10.0,
        "act_reg": 0.5,
        "done": -100,
    }

    def _setup(
        self,
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        min_height=0.30,
        max_rot=0.8,
        reset_type="init",
        chair_pos=np.array([-0.5, 0.0, 0.45]),
        max_episode_steps=1000,
        **kwargs,
    ):
        self.min_height = min_height
        self.max_rot = max_rot
        self.reset_type = reset_type
        self.chair_pos = chair_pos
        self.max_episode_steps = max_episode_steps
        self.steps = 0
        self.has_sat_down = False

        self.forbidden_contact_bodies = []
        for part in ["patella_l", "patella_r", "tibia_l", "tibia_r", "head"]:
            try: self.forbidden_contact_bodies.append(self.sim.model.body_name2id(part))
            except: pass

        self.arm_joints = []
        for i in range(self.sim.model.njnt):
            try: jname = mujoco.mj_id2name(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            except: jname = None
            if jname and any(part in jname for part in ["shoulder", "elbow", "wrist"]):
                self.arm_joints.append(self.sim.model.jnt_qposadr[i])


        try:
            self.foot_l_id = self.sim.model.body_name2id("talus_l")
            self.foot_r_id = self.sim.model.body_name2id("talus_r")
            self.pelvis_id = self.sim.model.body_name2id("pelvis")
        except: pass

        for k in ["target_reach_range", "far_th", "target_x_vel", "target_y_vel", "target_rot"]:
            if k in kwargs: del kwargs[k]

        self.ref_qpos = self.sim.model.qpos0.copy()
        super(WalkEnvV0, self)._setup(obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, sites=[], **kwargs)
        self.init_qpos[:] = self.sim.model.qpos0.copy()
        self.init_qvel[:] = 0.0

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict["t"] = np.array([sim.data.time])
        obs_dict["time"] = np.array([sim.data.time])
        obs_dict["qpos_without_xy"] = sim.data.qpos[2:].copy()
        obs_dict["qvel"] = sim.data.qvel[:].copy() * self.dt
        obs_dict["com_vel"] = np.array([self._get_com_velocity().copy()])
        obs_dict["torso_angle"] = np.array([self._get_torso_angle().copy()])
        obs_dict["feet_heights"] = self._get_feet_heights().copy()
        obs_dict["height"] = np.array([self._get_height()]).copy()
        obs_dict["muscle_length"] = self.muscle_lengths()
        obs_dict["muscle_velocity"] = self.muscle_velocities()
        obs_dict["muscle_force"] = self.muscle_forces()

        pelvis_pos = sim.data.qpos[:3]
        obs_dict["chair_rel_pos"] = self.chair_pos - pelvis_pos

        if sim.model.na > 0: obs_dict["act"] = sim.data.act[:].copy()
        return obs_dict

    def get_reward_dict(self, obs_dict):
        pelvis_pos = self.sim.data.qpos[:3]
        chair_height = self.chair_pos[2]

        vec_dist = pelvis_pos[:2] - self.chair_pos[:2]
        dist_x = np.abs(vec_dist[0])
        dist_y = np.abs(vec_dist[1])
        dist_z = np.abs(pelvis_pos[2] - chair_height)
        dist_xy = np.linalg.norm(vec_dist)

        approach_shaping = 1.0 / (dist_xy + 0.2)
        sit_approach_reward = approach_shaping * 2.0

        sit_contact_reward = np.exp(-dist_z**2 / (2 * 0.15**2)) * np.exp(-dist_x**2 / (2 * 0.15**2))
        sit_center_reward = np.exp(-dist_y**2 / (2 * 0.05**2))

        if dist_z < 0.05 and dist_xy < 0.15:
            self.has_sat_down = True

        torso_quat = self._get_torso_angle()
        R = quat2mat(torso_quat)
        vec_up = R @ np.array([0, 0, 1])
        upright_strict_reward = np.exp(-5.0 * (1.0 - vec_up[2]))


        leg_geometry_reward = 0.0
        if dist_xy < 0.3:
            hips = self._get_angle(["hip_flexion_l", "hip_flexion_r"])
            knees = self._get_angle(["knee_angle_l", "knee_angle_r"])
            avg_hip = np.mean(hips)
            avg_knee = np.mean(knees)

            target_hip = 1.5
            target_knee = 1.6


            knee_penalty_factor = 5.0 if avg_knee < 1.0 else 2.0

            err_hip = np.abs(avg_hip - target_hip)
            err_knee = np.abs(avg_knee - target_knee)

            leg_geometry_reward = np.exp(-2.0 * err_hip - knee_penalty_factor * err_knee)


        feet_under_knee_reward = 0.0
        try:
            p_x = self.sim.data.body_xpos[self.pelvis_id][0]
            f_l_x = self.sim.data.body_xpos[self.foot_l_id][0]
            f_r_x = self.sim.data.body_xpos[self.foot_r_id][0]





            rel_l = f_l_x - p_x
            rel_r = f_r_x - p_x


            target_rel = 0.2

            dist_l = np.abs(rel_l - target_rel)
            dist_r = np.abs(rel_r - target_rel)


            feet_under_knee_reward = np.exp(-5.0 * (dist_l + dist_r))
        except: pass

        feet_heights = self._get_feet_heights()
        feet_plant_reward = np.exp(-10.0 * np.mean(feet_heights)) * np.exp(-20.0 * np.abs(feet_heights[0] - feet_heights[1]))

        torso_forward = R @ np.array([1.0, 0.0, 0.0])
        sit_heading_reward = np.exp(-5.0 * np.abs(torso_forward[1]))

        pelvis_vel_z = self.sim.data.qvel[2]
        soft_landing = np.exp(-2.0 * np.abs(pelvis_vel_z))

        arm_qvel = 0.0
        if hasattr(self, 'arm_joints') and len(self.arm_joints) > 0:
            for idx in self.arm_joints:
                arm_qvel += np.abs(self.sim.data.qvel[idx])
        arm_reg = np.exp(-0.1 * arm_qvel)

        com_vel = self._get_com_velocity()
        stillness_reward = 0.0
        if dist_xy < 0.15:
            stillness_reward = np.exp(-5.0 * np.linalg.norm(com_vel))

        act_mag = np.linalg.norm(self.obs_dict["act"]) / self.sim.model.na if self.sim.model.na != 0 else 0
        done = self._get_done()

        rwd_dict = collections.OrderedDict(
            (
                ("sit_approach", sit_approach_reward),
                ("sit_contact", sit_contact_reward),
                ("sit_center", sit_center_reward),
                ("leg_geometry", leg_geometry_reward),
                ("feet_under_knee", feet_under_knee_reward),
                ("upright_strict", upright_strict_reward),
                ("feet_plant", feet_plant_reward),
                ("sit_heading", sit_heading_reward),
                ("stillness", stillness_reward),
                ("soft_landing", soft_landing),
                ("arm_reg", arm_reg),
                ("alive", 1.0),
                ("act_reg", -1.0 * act_mag),
                ("done", 1.0 if done else 0.0),
                ("sparse", sit_contact_reward * sit_center_reward * upright_strict_reward),
                ("solved", dist_xy < 0.1 and dist_z < 0.05),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict.get(key, 0.0) for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict

    def _get_done(self):
        if self.steps >= self.max_episode_steps: return 1

        pelvis_pos = self.sim.data.qpos[:3]
        pelvis_z = pelvis_pos[2]

        if pelvis_z < 0.25: return 1

        for bid in self.forbidden_contact_bodies:
            if self.sim.data.body_xpos[bid][2] < 0.1: return 1

        try:
            head_id = self.sim.model.body_name2id("head")
            head_z = self.sim.data.body_xpos[head_id][2]
            if head_z < pelvis_z + 0.3: return 1
        except: pass

        torso_quat = self.sim.data.qpos[3:7]
        z_up = (quat2mat(torso_quat) @ np.array([0, 0, 1]))[2]

        limit = 0.4 if self.has_sat_down else 0.6
        if z_up < limit: return 1

        return 0

    def reset(self, **kwargs):
        self.steps = 0
        self.has_sat_down = False

        qpos = self.sim.model.qpos0.copy()
        qvel = np.zeros(self.sim.model.nv)

        qpos[0] = -0.35
        qpos[1] = 0.0
        qpos[2] = 0.93

        self._set_joint_angle(qpos, ["hip_flexion_l", "hip_flexion_r"], 0.3)
        self._set_joint_angle(qpos, ["knee_angle_l", "knee_angle_r"], 0.3)
        self._set_joint_angle(qpos, ["ankle_angle_l", "ankle_angle_r"], 0.25)

        qpos[:] += self.np_random.normal(0, 0.005, size=qpos.shape)
        qvel[:] += self.np_random.normal(0, 0.005, size=qvel.shape)

        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super(WalkEnvV0, self).reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        return obs


    def _set_joint_angle(self, qpos_arr, names, val):
        for name in names:
            try:
                id = self.sim.model.joint_name2id(name)
                addr = self.sim.model.jnt_qposadr[id]
                qpos_arr[addr] = val
            except: pass

    def _get_angle(self, names):
        vals = []
        for name in names:
            try:
                id = self.sim.model.joint_name2id(name)
                addr = self.sim.model.jnt_qposadr[id]
                vals.append(self.sim.data.qpos[addr])
            except: vals.append(0.0)
        return np.array(vals)

    def muscle_lengths(self): return self.sim.data.actuator_length
    def muscle_forces(self): return np.clip(self.sim.data.actuator_force / 1000, -100, 100)
    def muscle_velocities(self): return np.clip(self.sim.data.actuator_velocity, -100, 100)

    def _get_torso_angle(self):
        return self.sim.data.body_xquat[self.sim.model.body_name2id("torso")]

    def _get_com_velocity(self):
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        cvel = -self.sim.data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]

    def _get_height(self):
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        com = self.sim.data.xipos
        return (np.sum(mass * com, 0) / np.sum(mass))[2]

    def _get_feet_heights(self):
        try:
            f_l = self.sim.model.body_name2id("talus_l")
            f_r = self.sim.model.body_name2id("talus_r")
            return np.array([self.sim.data.body_xpos[f_l][2], self.sim.data.body_xpos[f_r][2]])
        except: return np.array([0.0, 0.0])


class MSKBenchBalanceEnvV0(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "height", "muscle_length",
        "muscle_velocity", "muscle_force",
        "feet_rel_positions",
        "board_pos",
        "board_quat",
        "board_ball_err",
        "com_board_err",
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {

        "board_align": 5.0,
        "board_level": 5.0,
        "com_align": 15.0,


        "keep_height": 4.0,
        "balance_torso": 4.0,
        "lean_forward": 15.0,


        "stand_still": 2.0,
        "feet_contact": 1.0,
        "done": -100,
        "act_reg": 0.05,
        "alive": 1.0,
    }

    def _setup(
        self,
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        min_height=0.8,
        max_episode_steps=1000,
        reset_type="init",
        **kwargs,
    ):
        self.max_episode_steps = max_episode_steps
        self.min_height = min_height
        self.reset_type = reset_type
        self.steps = 0


        for k in ["target_reach_range", "far_th", "target_rot", "hip_period", "target_x_vel", "target_z_vel"]:
            if k in kwargs: del kwargs[k]

        self.ref_qpos = self.sim.model.qpos0.copy()


        self.torso_id = self.sim.model.body_name2id("torso")
        self.board_id = self.sim.model.body_name2id("board")
        self.ball_id = self.sim.model.body_name2id("ball")
        self.pelvis_id = self.sim.model.body_name2id("pelvis")

        super(WalkEnvV0, self)._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            sites=[],
            max_episode_steps=max_episode_steps,
            **kwargs
        )
        self.init_qpos[:] = self.sim.model.qpos0.copy()
        self.init_qvel[:] = 0.0

        print("Balance environment V4 ready (step count and grace period fixes).")


    def step(self, a):

        self.steps += 1




        return super().step(a)

    def get_obs_dict(self, sim):
        obs_dict = {}

        obs_dict["t"] = np.array([sim.data.time])
        obs_dict["time"] = np.array([sim.data.time])
        obs_dict["qpos_without_xy"] = sim.data.qpos[2:].copy()
        obs_dict["qvel"] = sim.data.qvel[:].copy() * self.dt
        obs_dict["com_vel"] = np.array([self._get_com_velocity().copy()])
        obs_dict["torso_angle"] = np.array([self._get_torso_angle().copy()])
        obs_dict["feet_heights"] = self._get_feet_heights().copy()
        obs_dict["height"] = np.array([self._get_height()]).copy()
        obs_dict["feet_rel_positions"] = self._get_feet_relative_position().copy()

        if sim.model.na > 0:
            obs_dict["act"] = sim.data.act[:].copy()
            obs_dict["muscle_length"] = self.muscle_lengths()
            obs_dict["muscle_velocity"] = self.muscle_velocities()
            obs_dict["muscle_force"] = self.muscle_forces()
        else:
            obs_dict["act"] = np.zeros(0)
            obs_dict["muscle_length"] = np.zeros(0)
            obs_dict["muscle_velocity"] = np.zeros(0)
            obs_dict["muscle_force"] = np.zeros(0)


        try:
            board_pos = sim.data.body_xpos[self.board_id].copy()
            ball_pos = sim.data.body_xpos[self.ball_id].copy()
            board_quat = sim.data.body_xquat[self.board_id].copy()

            obs_dict["board_pos"] = board_pos
            obs_dict["board_quat"] = board_quat
            obs_dict["board_ball_err"] = board_pos[:2] - ball_pos[:2]

            com_pos = self.sim.data.subtree_com[1][:2]
            obs_dict["com_board_err"] = com_pos - board_pos[:2]

        except KeyError:
            obs_dict["board_pos"] = np.zeros(3)
            obs_dict["board_quat"] = np.array([1, 0, 0, 0])
            obs_dict["board_ball_err"] = np.zeros(2)
            obs_dict["com_board_err"] = np.zeros(2)

        return obs_dict

    def get_reward_dict(self, obs_dict):
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        torso_quat = self._get_torso_angle()
        R = quat2mat(torso_quat)




        x_axis = np.array([1, 0, 0])
        pitch = (R @ x_axis)[2]


        target_pitch = -0.15


        pitch_error = pitch - target_pitch


        if pitch > 0.05:
            r_lean = 0.0
        else:

            r_lean = np.exp(-10.0 * (pitch_error**2))

        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        try:
            com_pos = self.sim.data.subtree_com[1]
            board_pos = self.sim.data.body_xpos[self.board_id]
            com_dist = np.linalg.norm(com_pos[:2] - board_pos[:2])
            r_com_align = np.exp(-20.0 * com_dist**2)
        except:
            r_com_align = 0.0

        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        z_axis = np.array([0, 0, 1])
        z_up = (R @ z_axis)[2]
        r_balance_torso = np.exp(-5.0 * max(0, 1.0 - z_up))

        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        height = self._get_height()
        target_height = 1.30
        if height > target_height:
            r_keep_height = 1.0
        else:
            r_keep_height = np.exp(-10.0 * (target_height - height)**2)

        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        com_vel = self._get_com_velocity()
        r_stand_still = np.exp(-5.0 * np.linalg.norm(com_vel)**2)

        feet_h = obs_dict["feet_heights"]
        r_feet_contact = np.exp(-5.0 * np.sum(np.abs(feet_h - np.min(feet_h))))

        r_board_align = 0.0
        r_board_level = 0.0
        try:
            b_pos = self.sim.data.body_xpos[self.board_id]
            ball_pos = self.sim.data.body_xpos[self.ball_id]
            b_quat = self.sim.data.body_xquat[self.board_id]

            dist_xy = np.linalg.norm(b_pos[:2] - ball_pos[:2])
            r_board_align = np.exp(-5.0 * dist_xy)

            b_mat = quat2mat(b_quat)
            b_up = (b_mat @ np.array([0, 0, 1]))[2]
            r_board_level = np.exp(-5.0 * (1.0 - b_up))
        except: pass

        act_mag = 0.0
        if self.sim.model.na != 0:
            act_val = np.linalg.norm(obs_dict["act"], axis=-1)
            if isinstance(act_val, np.ndarray): act_val = act_val.item()
            act_mag = act_val / self.sim.model.na

        done = self._get_done()

        is_solved = (r_board_align > 0.8) and (r_keep_height > 0.8) and (r_com_align > 0.8)
        sparse_reward = 1.0 if is_solved else 0.0

        rwd_dict = collections.OrderedDict(
            (
                ("stand_still", r_stand_still),
                ("balance_torso", r_balance_torso),
                ("lean_forward", r_lean),
                ("com_align", r_com_align),
                ("feet_contact", r_feet_contact),
                ("keep_height", r_keep_height),
                ("board_align", r_board_align),
                ("board_level", r_board_level),
                ("alive", 1.0),
                ("act_reg", -0.1 * act_mag),
                ("done", 1.0 if done else 0.0),
                ("sparse", sparse_reward),
                ("solved", is_solved),
            )
        )

        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict.get(key, 0.0) for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict

    def reset(self, **kwargs):
        self.steps = 0
        qpos = self.sim.model.qpos0.copy()
        qvel = np.zeros(self.sim.model.nv)


        qpos[0] = 0.0
        qpos[1] = 0.0



        qpos[2] = 1.28



        self._set_joint_angle(qpos, ["hip_flexion_l", "hip_flexion_r"], 0.25)
        self._set_joint_angle(qpos, ["knee_angle_l", "knee_angle_r"], 0.35)
        self._set_joint_angle(qpos, ["ankle_angle_l", "ankle_angle_r"], 0.15)



        qvel[0] = 0.1


        qpos[7:] += self.np_random.normal(0, 0.01, size=qpos[7:].shape)
        qvel[:] += self.np_random.normal(0, 0.01, size=qvel.shape)

        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super(WalkEnvV0, self).reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        return obs

    def _get_done(self):
        if self.steps >= self.max_episode_steps: return 1


        height = self._get_height()
        quat = self.sim.data.qpos[3:7].copy()
        R = quat2mat(quat)
        z_up = (R @ np.array([0, 0, 1]))[2]
        x_proj_z = (R @ np.array([1, 0, 0]))[2]

        # ====================================================

        # ====================================================
        if self.steps < 20:
            if height < 0.6: return 1
            if z_up < 0.2: return 1


            try:
                if self.sim.data.body_xpos[self.board_id][2] < 0.1: return 1
            except: pass

            return 0

        # ====================================================

        # ====================================================


        if z_up < 0.5: return 1



        if height < 0.70: return 1



        if x_proj_z > 0.2: return 1


        try:
            b_pos = self.sim.data.body_xpos[self.board_id]
            ball_pos = self.sim.data.body_xpos[self.ball_id]
            if np.linalg.norm(b_pos[:2] - ball_pos[:2]) > 0.5: return 1
            if b_pos[2] < 0.15: return 1
            b_quat = self.sim.data.body_xquat[self.board_id]
            b_up = (quat2mat(b_quat) @ np.array([0, 0, 1]))[2]
            if b_up < 0.7: return 1
        except KeyError: pass

        return 0

    def _set_joint_angle(self, qpos_arr, names, val):
        for name in names:
            try:
                id = self.sim.model.joint_name2id(name)
                addr = self.sim.model.jnt_qposadr[id]
                qpos_arr[addr] = val
            except: pass

    def _get_feet_relative_position(self):
        try:
            f_l = self.sim.model.body_name2id("talus_l")
            f_r = self.sim.model.body_name2id("talus_r")
            pelvis = self.pelvis_id
            return np.array([
                self.sim.data.body_xpos[f_l] - self.sim.data.body_xpos[pelvis],
                self.sim.data.body_xpos[f_r] - self.sim.data.body_xpos[pelvis],
            ])
        except: return np.zeros((2, 3))

    def muscle_lengths(self): return self.sim.data.actuator_length
    def muscle_forces(self): return np.clip(self.sim.data.actuator_force / 1000, -100, 100)
    def muscle_velocities(self): return np.clip(self.sim.data.actuator_velocity, -100, 100)

    def _get_torso_angle(self):
        return self.sim.data.body_xquat[self.torso_id]

    def _get_com_velocity(self):
        return self.sim.data.qvel[:2].copy()

    def _get_height(self):
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        com = self.sim.data.xipos
        return (np.sum(mass * com, 0) / np.sum(mass))[2]

    def _get_feet_heights(self):
        try:
            f_l = self.sim.model.body_name2id("talus_l")
            f_r = self.sim.model.body_name2id("talus_r")
            return np.array([self.sim.data.body_xpos[f_l][2], self.sim.data.body_xpos[f_r][2]])
        except: return np.array([0.0, 0.0])


class MSKBenchSquatEnvV0(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "height", "muscle_length",
        "muscle_velocity", "muscle_force",
        "phase_clock", "target_height"
    ]


    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "height_tracking": 200.0,
        "knee_tracking": 300.0,
        "hip_tracking": 200.0,
        "velocity_tracking": 200.0,


        "com_balance": 300.0,


        "kneeling_penalty": -1000.0,

        "asymmetry_penalty": -200.0,
        "feet_planted_bonus": 100.0,

        "x_vel_penalty": -100.0,
        "y_vel_penalty": -100.0,
        "yaw_penalty": -150.0,
        "torso_upright": 100.0,

        "act_reg": -0.1,
        "act_rate_penalty": -0.5,

        "done": -300.0,
        "alive": 10.0,

        "sparse": 0.0,
        "solved": 0.0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, normalize_act=True, reset_type='init', render_mode=None, max_episode_steps=1000, **kwargs):
        self._in_setup = True
        self.steps = 0
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.last_action = None


        self.knee_l_id = -1
        self.knee_r_id = -1
        self.pelvis_id = -1
        self.foot_l_id = -1
        self.foot_r_id = -1
        self.head_id = -1

        if 'render_mode' in kwargs: del kwargs['render_mode']

        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, normalize_act=normalize_act, **kwargs)
        self.reset_type = reset_type

        self._set_perfect_starting_pose()

        self._in_setup = False
        print("MSK-Bench task message.")

    def _set_perfect_starting_pose(self):
        perfect_qpos = self.sim.model.qpos0.copy()
        perfect_qpos[2] = 0.95

        target_joints = {
            "hip_flexion_l": 0.1, "hip_flexion_r": 0.1,
            "knee_angle_l": -0.1, "knee_angle_r": -0.1,
            "ankle_angle_l": -0.05, "ankle_angle_r": -0.05,
            "lumbar_extension": 0.1,
            "shoulder_flex_l": 1.5, "shoulder_flex_r": 1.5,
            "shoulder_add_l": -0.2, "shoulder_add_r": -0.2,
            "elbow_flex_l": 1.2, "elbow_flex_r": 1.2,
        }

        for joint_name, target_angle in target_joints.items():
            try:
                joint_id = self.sim.model.joint_name2id(joint_name)
                qpos_adr = self.sim.model.jnt_qposadr[joint_id]
                perfect_qpos[qpos_adr] = target_angle
            except Exception:
                pass

        self.init_qpos = perfect_qpos
        self.sim.model.qpos0[:] = perfect_qpos

    def _setup(self, obs_keys=DEFAULT_OBS_KEYS, weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS, **kwargs):
        self._in_setup = True

        def _get_id_safe(names, type='body'):
            for name in names:
                try:
                    if type == 'body': return self.sim.model.body_name2id(name)
                    if type == 'site': return self.sim.model.site_name2id(name)
                    if type == 'joint': return self.sim.model.joint_name2id(name)
                except: pass
            return -1

        self.head_id = _get_id_safe(["head", "cervical"])
        self.pelvis_id = _get_id_safe(["pelvis"])
        self.foot_l_id = _get_id_safe(["talus_l", "calcn_l", "foot_l"])
        self.foot_r_id = _get_id_safe(["talus_r", "calcn_r", "foot_r"])


        self.knee_l_id = _get_id_safe(["tibia_l", "patella_l"])
        self.knee_r_id = _get_id_safe(["tibia_r", "patella_r"])

        self.phase = 0.0
        self.squat_freq = 0.25

        self.target_z_height = 0.95
        self.target_v_z = 0.0
        self.target_knee_angle = 0.0
        self.target_hip_angle = 0.0

        if 'min_height' not in kwargs: kwargs['min_height'] = 0.35
        if 'target_x_vel' not in kwargs: kwargs['target_x_vel'] = 0.0

        super()._setup(obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, sites=[], **kwargs)
        self._in_setup = False

    def reset(self, **kwargs):
        self.steps = 0
        self.phase = 0.0
        self.target_z_height = 0.95
        self.target_v_z = 0.0
        self.target_knee_angle = 0.0
        self.target_hip_angle = 0.0
        self.last_action = None
        if 'seed' in kwargs: del kwargs['seed']
        if 'options' in kwargs: del kwargs['options']

        ret = super().reset(**kwargs)

        if hasattr(self, 'init_qpos'):
            noise = np.random.normal(0, 0.001, size=self.init_qpos[7:].shape)
            new_qpos = self.init_qpos.copy()
            new_qpos[7:] += noise
            self.sim.data.qpos[:] = new_qpos
            self.sim.data.qvel[:] = 0.0
            self.sim.forward()

            if hasattr(self, 'robot') and hasattr(self.robot, 'sync_sims'):
                self.robot.sync_sims(self.sim, self.sim_obsd)

            ret = self.get_obs(), {}

        return ret

    def step(self, a):
        if getattr(self, '_in_setup', True):
            try:
                obs = self.get_obs()
            except:
                obs = np.zeros(self.observation_space.shape if hasattr(self, 'observation_space') else 1, dtype=np.float32)
            return obs, 0.0, False, False, {}

        self.steps += 1
        self.phase += self.dt * 2 * np.pi * self.squat_freq

        self.target_z_height = 0.75 + 0.20 * np.cos(self.phase)
        self.target_v_z = -0.20 * (2 * np.pi * self.squat_freq) * np.sin(self.phase)
        self.target_knee_angle = -0.75 + 0.75 * np.cos(self.phase)
        self.target_hip_angle = 0.75 - 0.75 * np.cos(self.phase)

        a = np.nan_to_num(a, nan=0.0)
        a_clipped = np.clip(a, -1.0, 1.0)

        try:
            ret = super().step(a_clipped)
            if len(ret) == 4:
                obs_vec, reward, done, info = ret
                truncated = False
            else:
                obs_vec, reward, done, truncated, info = ret
        except Exception as e:
            if hasattr(self, 'observation_space'):
                dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            else:
                try: dummy_obs = self.get_obs()
                except: dummy_obs = np.zeros(1, dtype=np.float32)
            safe_info = {f"rwd_{k}": 0.0 for k in self.DEFAULT_RWD_KEYS_AND_WEIGHTS.keys()}
            safe_info.update({"rwd_dense": -200.0, "error_flag": 1.0})
            return dummy_obs, -200.0, True, False, safe_info

        self.obs_dict = self.get_obs_dict(self.sim)
        self.obs_dict['current_action'] = a_clipped

        rwd_dict = self.get_reward_dict(self.obs_dict)
        self.last_action = a_clipped

        for k, v in rwd_dict.items():
            info['rwd_' + k] = v

        final_obs = self.get_obs()
        safe_reward = float(np.clip(rwd_dict['dense'], -2000.0, 5000.0))
        final_done = bool(self._get_done())

        return final_obs, safe_reward, final_done, bool(truncated), info

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        try:
            obs_dict["phase_clock"] = np.array([np.sin(self.phase), np.cos(self.phase)])
            obs_dict["target_height"] = np.array([self.target_z_height])
        except:
            obs_dict["phase_clock"] = np.zeros(2)
            obs_dict["target_height"] = np.zeros(1)
        return obs_dict

    def _get_spine_vector(self):
        try:
            p_pos = self.sim.data.body_xpos[self.pelvis_id] if self.pelvis_id != -1 else self.sim.data.qpos[:3]
            h_pos = self.sim.data.body_xpos[self.head_id]
            vec = h_pos - p_pos
            norm = np.linalg.norm(vec)
            if norm > 1e-6: return vec / norm
        except: pass
        return np.array([0.0, 0.0, 1.0])

    def get_reward_dict(self, obs_dict):
        com_vel = self._get_com_velocity()

        height_tracking = 0.0
        velocity_tracking = 0.0
        knee_tracking = 0.0
        hip_tracking = 0.0
        asymmetry_penalty = 0.0
        feet_planted_bonus = 0.0
        com_balance = 0.0
        kneeling_penalty = 0.0

        try:
            pelvis_pos = self.sim.data.body_xpos[self.pelvis_id] if self.pelvis_id != -1 else self.sim.data.qpos[:3]
            f_l = self.sim.data.body_xpos[self.foot_l_id] if self.foot_l_id != -1 else pelvis_pos
            f_r = self.sim.data.body_xpos[self.foot_r_id] if self.foot_r_id != -1 else pelvis_pos


            avg_foot_pos = (f_l[:2] + f_r[:2]) / 2.0
            com_dist = np.linalg.norm(pelvis_pos[:2] - avg_foot_pos)
            com_balance = np.exp(-15.0 * com_dist**2)


            if self.knee_l_id != -1 and self.knee_r_id != -1:
                knee_l_z = self.sim.data.body_xpos[self.knee_l_id][2]
                knee_r_z = self.sim.data.body_xpos[self.knee_r_id][2]
                if knee_l_z < 0.25 or knee_r_z < 0.25:
                    kneeling_penalty = 1.0

            h_error = abs(pelvis_pos[2] - self.target_z_height)
            height_tracking = np.exp(-30.0 * h_error**2)

            v_z = self.sim.data.qvel[2]
            v_error = abs(v_z - self.target_v_z)
            velocity_tracking = np.exp(-5.0 * v_error**2)

            knee_l_val = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id("knee_angle_l")]]
            knee_r_val = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id("knee_angle_r")]]
            hip_l_val = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id("hip_flexion_l")]]
            hip_r_val = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id("hip_flexion_r")]]

            avg_knee = (knee_l_val + knee_r_val) / 2.0
            knee_error = abs(avg_knee - self.target_knee_angle)
            knee_tracking = np.exp(-5.0 * knee_error**2)

            avg_hip = (hip_l_val + hip_r_val) / 2.0
            hip_error = abs(avg_hip - self.target_hip_angle)
            hip_tracking = np.exp(-5.0 * hip_error**2)

            knee_diff = abs(knee_l_val - knee_r_val)
            hip_diff = abs(hip_l_val - hip_r_val)
            asymmetry_penalty = np.clip(knee_diff + hip_diff, 0.0, 2.0)

            foot_z_max = max(f_l[2], f_r[2])
            if foot_z_max < 0.05:
                feet_planted_bonus = 1.0

            spine_vec = self._get_spine_vector()
            z_up = spine_vec[2]

        except:
            spine_vec = np.array([0,0,1])
            z_up = 1.0

        x_vel_penalty = abs(com_vel[0])
        y_vel_penalty = abs(com_vel[1])

        torso_upright = np.exp(-10.0 * (1.0 - z_up)**2)

        try:
            pelvis_quat = self.sim.data.qpos[3:7]
            mat = quat2mat(pelvis_quat)
            y_fwd = (mat @ np.array([1, 0, 0]))[1]
            yaw_penalty = np.abs(y_fwd)
        except:
            yaw_penalty = 0.0

        curr_act = obs_dict.get('current_action', np.zeros(self.sim.model.nu))
        act_mag = np.mean(np.square(curr_act))
        act_rate = np.mean(np.square(curr_act - self.last_action)) if self.last_action is not None else 0.0

        done = self._get_done()

        rwd_dict = collections.OrderedDict((
            ("height_tracking", float(height_tracking)),
            ("velocity_tracking", float(velocity_tracking)),
            ("knee_tracking", float(knee_tracking)),
            ("hip_tracking", float(hip_tracking)),
            ("com_balance", float(com_balance)),
            ("kneeling_penalty", float(kneeling_penalty)),
            ("asymmetry_penalty", float(asymmetry_penalty)),
            ("feet_planted_bonus", float(feet_planted_bonus)),
            ("x_vel_penalty", float(x_vel_penalty)),
            ("y_vel_penalty", float(y_vel_penalty)),
            ("yaw_penalty", float(yaw_penalty)),
            ("torso_upright", float(torso_upright)),
            ("act_reg", float(act_mag)),
            ("act_rate_penalty", float(act_rate)),
            ("alive", 1.0),
            ("done", float(1.0 if done else 0.0)),
            ("sparse", 0.0),
            ("solved", 0.0),
        ))

        rwd_dict["dense"] = np.sum([self.rwd_keys_wt.get(key, 0.0) * rwd_dict.get(key, 0.0) for key in self.rwd_keys_wt.keys()])
        return rwd_dict

    def _get_done(self):
        if getattr(self, '_in_setup', True): return 0
        try:
            if not np.isfinite(self.sim.data.qpos).all() or not np.isfinite(self.sim.data.qvel).all(): return 1
            if self.steps >= self.max_episode_steps: return 1

            if self.sim.data.qpos[2] < 0.35: return 1


            if self.knee_l_id != -1 and self.knee_r_id != -1:
                knee_l_z = self.sim.data.body_xpos[self.knee_l_id][2]
                knee_r_z = self.sim.data.body_xpos[self.knee_r_id][2]
                if knee_l_z < 0.20 or knee_r_z < 0.20:
                    return 1

            spine_vec = self._get_spine_vector()
            z_up = spine_vec[2]

            if z_up < 0.2: return 1

            pelvis_quat = self.sim.data.qpos[3:7]
            mat = quat2mat(pelvis_quat)
            y_fwd = (mat @ np.array([1, 0, 0]))[1]
            if abs(y_fwd) > 0.8:
                return 1

        except: pass
        return 0

    def _get_torso_angle(self): return self.sim.data.body_xquat[self.sim.model.body_name2id("torso")]
    def _get_com_velocity(self): return self.sim.data.qvel[:2].copy()


class MSKBenchCrawlEnvV0(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "muscle_length", "muscle_velocity",
        "muscle_force"
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {

        "backward_vel": 500.0,

        "crawling_posture": 100.0,

        "hand_stride": 150.0,
        "alternating_limbs": 150.0,

        "hand_width_penalty": -100.0,
        "head_up": 50.0,


        "heading_penalty": -150.0,
        "spine_twist_penalty": -100.0,

        "act_reg": 0.05,

        "alive": 5.0,
        "done": -200.0,
        "sparse": 1.0,
        "solved": 10.0,
    }

    def _find_body_id(self, potential_names):
        for name in potential_names:
            try:
                return self.sim.model.body_name2id(name)
            except Exception:
                continue
        return None

    def _setup(self, **kwargs):
        self.target_vel = 1.0


        self.hand_r_id = self._find_body_id(["lunate", "radius"])
        self.hand_l_id = self._find_body_id(["lunate_l", "radius_l"])
        self.foot_l_id = self._find_body_id(["talus_l"])
        self.foot_r_id = self._find_body_id(["talus_r"])

        self.head_id = self._find_body_id(["head", "cervical"])
        self.torso_id = self._find_body_id(["torso"])
        self.pelvis_id = self._find_body_id(["pelvis"])

        super()._setup(obs_keys=self.DEFAULT_OBS_KEYS,
                       weighted_reward_keys=self.DEFAULT_RWD_KEYS_AND_WEIGHTS,
                       **kwargs)
        print("MSK-Bench task message.")

    def reset(self, **kwargs):
        if 'seed' in kwargs: del kwargs['seed']
        if 'options' in kwargs: del kwargs['options']

        qpos = self.sim.model.qpos0.copy()
        qvel = np.zeros(self.sim.model.nv)


        qpos[2] = 0.25
        qpos[3:7] = [0.7071068, 0.0, 0.7071068, 0.0]


        pose_dict = {

            "hip_flexion_r": 1.0,
            "hip_flexion_l": 1.0,
            "knee_angle_r": 1.5,
            "knee_angle_l": 1.5,


            "shoulder_elv": 1.5,
            "elv_angle": 0.5,
            "elbow_flexion": 0.5,
            "pro_sup": 1.57,


            "shoulder_elv_l": 1.5,
            "elv_angle_l": 0.5,
            "elbow_flexion_l": 0.5,
            "pro_sup_l": 1.57
        }

        for jnt, val in pose_dict.items():
            try:
                jnt_id = self.sim.model.joint_name2id(jnt)
                adr = self.sim.model.jnt_qposadr[jnt_id]
                qpos[adr] = val
            except Exception as e:
                pass


        qpos[7:] += self.np_random.normal(0, 0.01, size=qpos[7:].shape)

        self.robot.sync_sims(self.sim, self.sim_obsd)
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()


        for _ in range(10):
            mujoco.mj_step(self.sim.model.ptr, self.sim.data.ptr)

        ret = super(WalkEnvV0, self).reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        return ret if isinstance(ret, tuple) else (ret, {})

    def get_reward_dict(self, obs_dict):
        rwd_dict = collections.OrderedDict((k, 0.0) for k in self.rwd_keys_wt.keys())

        v_x = self.sim.data.qvel[0]
        v_y = self.sim.data.qvel[1]
        pelvis_pos = self.sim.data.qpos[:3]
        pelvis_z = pelvis_pos[2]


        is_moving_backward = 1.0 if v_x < -0.05 else 0.0


        r_vel = np.clip(-v_x / self.target_vel, -0.2, 1.0)
        rwd_dict["backward_vel"] = r_vel


        rwd_dict["heading_penalty"] = np.abs(v_y) * 2.0 + np.abs(v_x) * (1.0 if v_x > 0 else 0)

        R_pelvis = quat2mat(self.sim.data.body_xquat[self.pelvis_id])
        pelvis_chest_z = R_pelvis[2, 0]
        pelvis_head_x = R_pelvis[0, 2]

        r_posture = np.exp(-5.0 * (pelvis_chest_z - (-1.0))**2) * np.exp(-5.0 * (pelvis_head_x - 1.0)**2)
        if not (0.05 < pelvis_z < 0.6):
            r_posture = 0.0

        rwd_dict["crawling_posture"] = r_posture * is_moving_backward

        if self.hand_r_id and self.hand_l_id and self.foot_l_id and self.foot_r_id:
            hand_r_x = self.sim.data.xpos[self.hand_r_id][0]
            hand_l_x = self.sim.data.xpos[self.hand_l_id][0]
            foot_r_x = self.sim.data.xpos[self.foot_r_id][0]
            foot_l_x = self.sim.data.xpos[self.foot_l_id][0]

            hand_diff = hand_l_x - hand_r_x
            rwd_dict["hand_stride"] = np.clip(np.abs(hand_diff) / 0.5, 0.0, 1.0) * is_moving_backward

            foot_diff = foot_r_x - foot_l_x
            if hand_diff * foot_diff > 0:
                alt_score = np.clip(np.abs(hand_diff) / 0.3, 0.0, 1.0) * np.clip(np.abs(foot_diff) / 0.3, 0.0, 1.0)
                rwd_dict["alternating_limbs"] = alt_score * is_moving_backward

            hand_dist_y = np.abs(self.sim.data.xpos[self.hand_l_id][1] - self.sim.data.xpos[self.hand_r_id][1])
            if hand_dist_y < 0.4:
                rwd_dict["hand_width_penalty"] = np.exp(-10.0 * (hand_dist_y - 0.4)**2)

        r_head = 0.0
        if self.head_id:
            head_z = self.sim.data.xpos[self.head_id][2]
            if head_z > 0.35:
                r_head = 1.0
            else:
                r_head = np.exp(-10.0 * (0.35 - head_z)**2)
        rwd_dict["head_up"] = r_head * is_moving_backward

        spine_twist = 0.0
        if self.torso_id:
            R_torso = quat2mat(self.sim.data.body_xquat[self.torso_id])
            torso_chest_z = R_torso[2, 0]
            if torso_chest_z > -0.7:
                spine_twist = np.abs(torso_chest_z - (-0.7))
        rwd_dict["spine_twist_penalty"] = spine_twist

        act = obs_dict.get("act", np.zeros(self.sim.model.nu))
        rwd_dict["act_reg"] = -1.0 * np.mean(np.square(act))
        rwd_dict["alive"] = 1.0

        is_done = self._get_done()

        is_solved = 1.0 if (v_x < -0.8 and not is_done) else 0.0

        rwd_dict["done"] = float(is_done)
        rwd_dict["sparse"] = is_solved
        rwd_dict["solved"] = is_solved

        dense_reward = 0.0
        for key, weight in self.rwd_keys_wt.items():
            if key != "dense" and key in rwd_dict:
                dense_reward += weight * rwd_dict[key]

        rwd_dict["dense"] = dense_reward
        self.rwd_dict = rwd_dict

        return rwd_dict

    def _get_done(self):
        if getattr(self, '_in_setup', True): return False


        if self.sim.data.qvel[0] > 0.1:
            return True

        pelvis_z = self.sim.data.qpos[2]
        if pelvis_z < 0.08: return True
        if pelvis_z > 0.65: return True

        try:
            R_pelvis = quat2mat(self.sim.data.body_xquat[self.pelvis_id])

            pelvis_chest_z = R_pelvis[2, 0]
            pelvis_head_x  = R_pelvis[0, 2]
            pelvis_left_y  = R_pelvis[1, 1]


            if pelvis_chest_z > -0.7: return True
            if pelvis_head_x < 0.7: return True
            if pelvis_left_y < 0.7: return True

            if self.torso_id:
                R_torso = quat2mat(self.sim.data.body_xquat[self.torso_id])
                if R_torso[2, 0] > -0.6: return True

        except: pass

        return False


class MSKBenchRunEnvV0(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "height", "muscle_length",
        "muscle_velocity", "muscle_force",
        "phase_clock"
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "forward_vel": 400.0,
        "gait_sync": 250.0,
        "pelvis_height": 150.0,
        "torso_upright": 50.0,
        "feet_clearance": 80.0,

        "yaw_penalty": -100.0,
        "act_reg": -2.0,
        "done": -100.0,
        "alive": 5.0,
    }

    def __init__(self, model_path, **kwargs):
        self._in_setup = True
        self.steps = 0
        self.last_action = None

        self.head_id = -1
        self.pelvis_id = -1
        self.torso_id = -1
        self.foot_l_id = -1
        self.foot_r_id = -1
        self.hip_l_id = -1
        self.hip_r_id = -1
        self.knee_joint_l_id = -1
        self.knee_joint_r_id = -1


        self.phase = 0.0
        self.gait_freq = 1.5

        super().__init__(model_path=model_path, **kwargs)
        self._in_setup = False
        print("MSK-Bench task message.")

    def _setup(self, **kwargs):
        self._in_setup = True

        def _get_id_safe(names, type='body'):
            for name in names:
                try:
                    if type == 'body': return self.sim.model.body_name2id(name)
                    if type == 'site': return self.sim.model.site_name2id(name)
                    if type == 'joint': return self.sim.model.joint_name2id(name)
                except: pass
            return -1

        self.pelvis_id = _get_id_safe(["pelvis"])
        self.torso_id = _get_id_safe(["thorax", "torso", "spine"])
        self.foot_l_id = _get_id_safe(["talus_l", "calcn_l", "foot_l"])
        self.foot_r_id = _get_id_safe(["talus_r", "calcn_r", "foot_r"])
        self.head_id = _get_id_safe(["head", "cervical"])


        self.hip_l_id = _get_id_safe(["hip_flexion_l"], 'joint')
        self.hip_r_id = _get_id_safe(["hip_flexion_r"], 'joint')
        self.knee_joint_l_id = _get_id_safe(["knee_angle_l"], 'joint')
        self.knee_joint_r_id = _get_id_safe(["knee_angle_r"], 'joint')

        super()._setup(obs_keys=self.DEFAULT_OBS_KEYS,
                       weighted_reward_keys=self.DEFAULT_RWD_KEYS_AND_WEIGHTS,
                       **kwargs)

    def reset(self, *, seed=None, options=None, **kwargs):
        self.steps = 0
        self.last_action = None
        self.phase = 0.0

        if seed is not None:
            self._np_random, seed = gym.utils.seeding.np_random(seed)

        qpos = self.sim.model.qpos0.copy()
        qvel = np.zeros(self.sim.model.nv)


        qpos[7:] += np.random.normal(0, 0.01, size=qpos[7:].shape)
        qvel[:] += np.random.normal(0, 0.01, size=qvel.shape)


        qpos[2] = 0.95

        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()

        if hasattr(self, 'robot') and hasattr(self.robot, 'sync_sims'):
            self.robot.sync_sims(self.sim, self.sim_obsd)

        settled_qpos = self.sim.data.qpos.copy()
        settled_qvel = self.sim.data.qvel.copy()

        ret = super(WalkEnvV0, self).reset(reset_qpos=settled_qpos, reset_qvel=settled_qvel, **kwargs)
        return ret if isinstance(ret, tuple) else (ret, {})

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        obs_dict["phase_clock"] = np.array([np.sin(self.phase), np.cos(self.phase)], dtype=np.float32)
        return obs_dict

    def step(self, a):
        if getattr(self, '_in_setup', True):
            try: obs = self.get_obs()
            except: obs = np.zeros(self.observation_space.shape if hasattr(self, 'observation_space') else 1, dtype=np.float32)
            return obs, 0.0, False, False, {}

        self.steps += 1

        self.phase += self.dt * 2 * np.pi * self.gait_freq
        self.phase = self.phase % (2 * np.pi)

        if np.isnan(a).any() or np.isinf(a).any():
            a = np.zeros_like(a)

        a_clipped = np.clip(a, -1.0, 1.0)
        self.last_action = a_clipped

        try:
            ret = super().step(a_clipped)
            if len(ret) == 5: obs_vec, reward, done, truncated, info = ret
            else: obs_vec, reward, done, info = ret; truncated = False
        except Exception as e:
            if hasattr(self, 'observation_space'): dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            else:
                try: dummy_obs = self.get_obs()
                except: dummy_obs = np.zeros(1, dtype=np.float32)
            safe_info = {f"rwd_{k}": 0.0 for k in self.DEFAULT_RWD_KEYS_AND_WEIGHTS.keys()}
            safe_info.update({"rwd_dense": -200.0, "error_flag": 1.0})
            return dummy_obs, -200.0, True, False, safe_info

        self.obs_dict = self.get_obs_dict(self.sim)
        self.obs_dict['current_action'] = a_clipped

        rwd_dict = self.get_reward_dict(self.obs_dict)
        for k, v in rwd_dict.items(): info['rwd_' + k] = v

        final_obs = self.get_obs()
        safe_reward = float(np.clip(rwd_dict['dense'], -1500.0, 5000.0))
        final_done = bool(self._get_done())

        return final_obs, safe_reward, final_done, bool(truncated), info

    def get_reward_dict(self, obs_dict):
        rwd_dict = collections.OrderedDict()
        for k in self.rwd_keys_wt.keys(): rwd_dict[k] = 0.0
        rwd_dict["dense"] = 0.0; rwd_dict["sparse"] = 0.0; rwd_dict["solved"] = 0.0

        try:
            v_x = self.sim.data.qvel[0]
            v_x_reward = float(np.clip(v_x / 2.5, -0.2, 1.0))

            pelvis_z = self.sim.data.qpos[2]


            r_pelvis_height = np.exp(-10.0 * (0.95 - pelvis_z)**2)

            torso_quat = self.sim.data.body_xquat[self.torso_id] if self.torso_id != -1 else self.sim.data.qpos[3:7]
            R_torso = quat2mat(torso_quat)
            z_up = R_torso[2, 2]
            r_upright = np.exp(-10.0 * (1.0 - z_up)**2)

            yaw_penalty = abs(self.sim.data.qvel[1]) * 0.5

            feet_clearance = 0.0
            if self.foot_l_id != -1 and self.foot_r_id != -1:
                f_l_z = self.sim.data.body_xpos[self.foot_l_id][2]
                f_r_z = self.sim.data.body_xpos[self.foot_r_id][2]
                max_foot_h = max(f_l_z, f_r_z)
                feet_clearance = np.clip((max_foot_h - 0.05) / 0.15, 0.0, 1.0)


            gait_sync_reward = 0.0
            if self.hip_l_id != -1 and self.knee_joint_l_id != -1:
                hip_l_pos = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.hip_l_id]]
                hip_r_pos = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.hip_r_id]]
                knee_l_pos = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.knee_joint_l_id]]
                knee_r_pos = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.knee_joint_r_id]]


                target_hip_l = 0.25 * np.sin(self.phase)
                target_hip_r = 0.25 * np.sin(self.phase + np.pi)


                target_knee_l = 0.2 * np.sin(self.phase) + 0.25
                target_knee_r = 0.2 * np.sin(self.phase + np.pi) + 0.25

                err_hip = (hip_l_pos - target_hip_l)**2 + (hip_r_pos - target_hip_r)**2
                err_knee = (knee_l_pos - target_knee_l)**2 + (knee_r_pos - target_knee_r)**2


                gait_sync_reward = np.exp(-3.0 * (err_hip + err_knee * 0.5))

            act_mag = np.mean(np.square(obs_dict.get('act', np.zeros(1))))
            is_done = self._get_done()

            rwd_dict["forward_vel"] = v_x_reward
            rwd_dict["pelvis_height"] = float(r_pelvis_height)
            rwd_dict["gait_sync"] = float(gait_sync_reward)
            rwd_dict["torso_upright"] = float(r_upright)
            rwd_dict["feet_clearance"] = float(feet_clearance)
            rwd_dict["yaw_penalty"] = float(yaw_penalty)
            rwd_dict["act_reg"] = float(act_mag)
            rwd_dict["alive"] = 1.0
            rwd_dict["done"] = float(1.0 if is_done else 0.0)

            total_rwd = sum([self.rwd_keys_wt.get(k, 0.0) * rwd_dict[k] for k in self.rwd_keys_wt.keys()])
            rwd_dict["dense"] = float(np.clip(total_rwd, -1000.0, 3000.0))

        except Exception: pass
        return rwd_dict

    def _get_done(self):
        if getattr(self, '_in_setup', True): return False

        try:
            if not np.isfinite(self.sim.data.qpos).all() or not np.isfinite(self.sim.data.qvel).all():
                return True

            pelvis_pos = self.sim.data.qpos[:3]



            if pelvis_pos[2] < 0.70:
                return True

            if self.head_id != -1:
                head_z = self.sim.data.body_xpos[self.head_id][2]
                if head_z < 1.2:
                    return True

        except Exception as e:
            print(f"Low-level termination check error: {e}")
            traceback.print_exc()
            return True

        return False


class MSKBenchJumpEnvV0(WalkEnvV0):
    """MSK-Bench task environment."""

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "height", "muscle_length",
        "muscle_velocity", "muscle_force",
        "feet_distance"
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {

        "flight_height": 600.0,
        "upward_vel": 300.0,
        "forward_vel": 150.0,

        "feet_separation_penalty": -200.0,
        "desync_penalty": -150.0,


        "torso_leaning_penalty": -200.0,
        "lumbar_penalty": -200.0,
        "yaw_penalty": -100.0,

        "act_reg": -1.0,
        "alive": 0.0,
        "done": -300.0,
        "sparse": 0.0,
        "solved": 0.0,
    }

    def _setup(self, **kwargs):
        self._in_setup = True

        def _get_id_safe(names):
            for name in names:
                try: return self.sim.model.body_name2id(name)
                except: pass
            return -1

        self.pelvis_id = _get_id_safe(["pelvis"])
        self.head_id = _get_id_safe(["head", "cervical"])
        self.foot_l_id = _get_id_safe(["talus_l", "calcn_l", "foot_l"])
        self.foot_r_id = _get_id_safe(["talus_r", "calcn_r", "foot_r"])

        super()._setup(
            obs_keys=self.DEFAULT_OBS_KEYS,
            weighted_reward_keys=self.DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs
        )
        self._in_setup = False
        print("MSK-Bench task message.")

    def reset(self, **kwargs):
        self.steps = 0

        if 'seed' in kwargs: del kwargs['seed']
        if 'options' in kwargs: del kwargs['options']

        qpos = self.sim.model.qpos0.copy()
        qvel = np.zeros(self.sim.model.nv)


        stance_angles = {
            "hip_flexion_l": 0.3, "hip_flexion_r": 0.3,
            "knee_angle_l": 0.4, "knee_angle_r": 0.4,
            "hip_adduction_l": 0.05, "hip_adduction_r": 0.05,
            "ankle_angle_l": 0.1, "ankle_angle_r": 0.1,
            "lumbar_extension": 0.0, "lumbar_bending": 0.0, "lumbar_rotation": 0.0
        }

        for joint, angle in stance_angles.items():
            try:
                jnt_id = self.sim.model.joint_name2id(joint)
                qpos[self.sim.model.jnt_qposadr[jnt_id]] = angle
            except: pass

        qpos[7:] += self.np_random.normal(0, 0.01, size=qpos[7:].shape)
        qvel[:] += self.np_random.normal(0, 0.01, size=qvel.shape)

        qpos[2] = 0.95

        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()

        import mujoco
        for _ in range(10):
            mujoco.mj_step(self.sim.model.ptr, self.sim.data.ptr)

        if hasattr(self, 'robot') and hasattr(self.robot, 'sync_sims'):
            self.robot.sync_sims(self.sim, self.sim_obsd)

        kwargs.pop('reset_qpos', None)
        kwargs.pop('reset_qvel', None)
        ret = super(WalkEnvV0, self).reset(reset_qpos=self.sim.data.qpos.copy(), reset_qvel=self.sim.data.qvel.copy(), **kwargs)
        return ret if isinstance(ret, tuple) else (ret, {})

    def step(self, a):
        if getattr(self, '_in_setup', True):
            try: obs = self.get_obs()
            except: obs = np.zeros(self.observation_space.shape[0] if hasattr(self, 'observation_space') else 1, dtype=np.float32)
            return obs, 0.0, False, False, {}

        self.steps += 1
        a_clipped = np.clip(np.nan_to_num(a, nan=0.0), -1.0, 1.0)

        try:
            ret = super().step(a_clipped)
            if len(ret) == 5: obs_vec, reward, done, truncated, info = ret
            else: obs_vec, reward, done, info = ret; truncated = False
        except Exception:
            obs_shape = self.observation_space.shape[0] if hasattr(self, 'observation_space') else 1
            safe_info = {f"rwd_{k}": 0.0 for k in self.DEFAULT_RWD_KEYS_AND_WEIGHTS.keys()}
            safe_info.update({"rwd_dense": -200.0, "error_flag": 1.0})
            return np.zeros(obs_shape, dtype=np.float32), -200.0, True, False, safe_info

        self.obs_dict = self.get_obs_dict(self.sim)
        rwd_dict = self.get_reward_dict(self.obs_dict)
        for k, v in rwd_dict.items(): info['rwd_' + k] = v

        final_obs = self.get_obs()
        safe_reward = float(np.clip(rwd_dict['dense'], -1500.0, 5000.0))
        final_done = bool(self._get_done())

        return final_obs, safe_reward, final_done, bool(truncated), info

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        try:
            if self.foot_l_id != -1 and self.foot_r_id != -1:
                f_l = sim.data.body_xpos[self.foot_l_id]
                f_r = sim.data.body_xpos[self.foot_r_id]
                dist_xy = np.linalg.norm(f_l[:2] - f_r[:2])
                obs_dict["feet_distance"] = np.array([dist_xy], dtype=np.float32)
            else:
                obs_dict["feet_distance"] = np.array([0.0], dtype=np.float32)
        except:
            obs_dict["feet_distance"] = np.array([0.0], dtype=np.float32)
        return obs_dict

    def get_reward_dict(self, obs_dict):
        rwd_dict = collections.OrderedDict()
        for k in self.rwd_keys_wt.keys(): rwd_dict[k] = 0.0

        try:
            com_vel = self._get_com_velocity()
            v_x = com_vel[0]


            r_forward_vel = float(np.clip(v_x, 0.0, 2.0))


            pelvis_v_z = self.sim.data.qvel[2]
            r_upward_vel = float(np.clip(pelvis_v_z, 0.0, 2.0))

            R_pelvis = quat2mat(self.sim.data.qpos[3:7])
            y_fwd = R_pelvis[1, 0]
            pelvis_z_up = R_pelvis[2, 2]
            r_yaw_penalty = float(abs(y_fwd))


            r_torso_leaning_penalty = 0.0
            spine_z_up = 1.0
            if self.head_id != -1 and self.pelvis_id != -1:
                head_pos = self.sim.data.body_xpos[self.head_id]
                pelvis_pos = self.sim.data.body_xpos[self.pelvis_id]
                spine_vec = head_pos - pelvis_pos
                spine_z_up = spine_vec[2] / (np.linalg.norm(spine_vec) + 1e-6)


            leaning_err = max(0.0, (1.0 - pelvis_z_up) - 0.04) + max(0.0, (1.0 - spine_z_up) - 0.04)
            r_torso_leaning_penalty = float(leaning_err)


            lumbar_err = 0.0
            for jnt in ["lumbar_extension", "lumbar_bending", "lumbar_rotation"]:
                try:
                    adr = self.sim.model.jnt_qposadr[self.sim.model.joint_name2id(jnt)]

                    lumbar_err += max(0.0, abs(self.sim.data.qpos[adr]) - 0.1)
                except: pass
            r_lumbar_penalty = float(lumbar_err)

            r_feet_separation_penalty = 0.0
            r_flight_height = 0.0
            r_desync_penalty = 0.0

            if self.foot_l_id != -1 and self.foot_r_id != -1:
                f_l = self.sim.data.body_xpos[self.foot_l_id]
                f_r = self.sim.data.body_xpos[self.foot_r_id]

                dist_xy = np.linalg.norm(f_l[:2] - f_r[:2])
                r_feet_separation_penalty = float(dist_xy)


                feet_mean_z = (f_l[2] + f_r[2]) / 2.0

                r_flight_height = float(np.clip(feet_mean_z / 0.1, 0.0, 1.0))

                try:
                    knee_l = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id("knee_angle_l")]]
                    knee_r = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id("knee_angle_r")]]
                    hip_l = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id("hip_flexion_l")]]
                    hip_r = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id("hip_flexion_r")]]
                    r_desync_penalty = float(abs(knee_l - knee_r) + abs(hip_l - hip_r))
                except: pass

            act_mag = np.mean(np.square(obs_dict.get('act', np.zeros(1))))
            is_done = self._get_done()

            rwd_dict["forward_vel"] = r_forward_vel
            rwd_dict["flight_height"] = r_flight_height
            rwd_dict["upward_vel"] = r_upward_vel

            rwd_dict["torso_leaning_penalty"] = r_torso_leaning_penalty
            rwd_dict["feet_separation_penalty"] = r_feet_separation_penalty
            rwd_dict["lumbar_penalty"] = r_lumbar_penalty
            rwd_dict["desync_penalty"] = r_desync_penalty

            rwd_dict["yaw_penalty"] = r_yaw_penalty
            rwd_dict["act_reg"] = float(act_mag)
            rwd_dict["alive"] = 0.0
            rwd_dict["done"] = float(1.0 if is_done else 0.0)

            total_rwd = sum([self.rwd_keys_wt.get(k, 0.0) * rwd_dict[k] for k in self.rwd_keys_wt.keys()])
            rwd_dict["dense"] = float(total_rwd)

        except Exception: pass
        return rwd_dict

    def _get_done(self):
        if getattr(self, '_in_setup', True): return False

        try:
            if not np.isfinite(self.sim.data.qpos).all() or not np.isfinite(self.sim.data.qvel).all():
                return True

            pelvis_pos = self.sim.data.qpos[:3]
            if pelvis_pos[2] < 0.45:
                return True


            if self.head_id != -1 and self.pelvis_id != -1:
                head_pos = self.sim.data.body_xpos[self.head_id]
                spine_vec = head_pos - pelvis_pos
                spine_z_up = spine_vec[2] / (np.linalg.norm(spine_vec) + 1e-6)
                if spine_z_up < 0.7:
                    return True
            else:
                R_pelvis = quat2mat(self.sim.data.qpos[3:7])
                if R_pelvis[2, 2] < 0.5:
                    return True

            if self.foot_l_id != -1 and self.foot_r_id != -1:
                f_l = self.sim.data.body_xpos[self.foot_l_id]
                f_r = self.sim.data.body_xpos[self.foot_r_id]
                if np.linalg.norm(f_l[:2] - f_r[:2]) > 0.4:
                    return True

        except Exception:
            return True

        return False

    def _get_com_velocity(self):
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        cvel = self.sim.data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]


class MSKBenchWalkTurnEnvV0(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "height", "muscle_length",
        "muscle_velocity", "muscle_force",
        "target_rel_pos",
        "next_target_id",
        "facing_target",
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {

        "target_vel": 200.0,
        "approach_target": 50.0,

        "facing_reward": 50.0,
        "torso_posture": 20.0,

        "reach_bonus": 500.0,
        "collision_penalty": -200.0,

        "act_reg": -1.0,
        "alive": 5.0,
        "done": 0.0,
    }

    def __init__(self, model_path, **kwargs):
        self._in_setup = True
        self.steps = 0
        super().__init__(model_path=model_path, **kwargs)
        print("MSK-Bench task message.")

    def _setup(self, **kwargs):

        self._in_setup = True

        self.targets = [
            self.sim.model.site_pos[self.sim.model.site_name2id("target_a")],
            self.sim.model.site_pos[self.sim.model.site_name2id("target_b")],
            self.sim.model.site_pos[self.sim.model.site_name2id("target_c")],
        ]
        self.current_target_idx = 0


        self.target_threshold = 0.5
        self.max_episode_steps = 2000

        super()._setup(obs_keys=self.DEFAULT_OBS_KEYS,
                       weighted_reward_keys=self.DEFAULT_RWD_KEYS_AND_WEIGHTS,
                       **kwargs)


        self._in_setup = False

    def reset(self, **kwargs):
        self.current_target_idx = 0
        self.steps = 0
        return super().reset(**kwargs)

    def step(self, a):
        self.steps += 1
        return super().step(a)

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)

        pelvis_pos = sim.data.qpos[:3]
        target_pos = self.targets[self.current_target_idx]

        target_rel_pos = target_pos - pelvis_pos
        obs_dict["target_rel_pos"] = target_rel_pos
        obs_dict["next_target_id"] = np.array([self.current_target_idx])

        try:
            world_quat = sim.data.body_xquat[self.sim.model.body_name2id("pelvis")]
            pelvis_rot_mat = quat2mat(world_quat)
            forward_vec = pelvis_rot_mat[:2, 0]
            forward_vec = forward_vec / (np.linalg.norm(forward_vec) + 1e-6)

            target_dir = target_rel_pos[:2]
            target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-6)

            facing_alignment = np.dot(forward_vec, target_dir)
            obs_dict["facing_target"] = np.array([facing_alignment])
        except:
            obs_dict["facing_target"] = np.array([0.0])

        return obs_dict

    def get_reward_dict(self, obs_dict):
        pelvis_pos = self.sim.data.qpos[:3]
        com_vel = self.sim.data.qvel[:2].copy()

        target_pos = self.targets[self.current_target_idx]
        target_rel_pos = target_pos[:2] - pelvis_pos[:2]
        dist = np.linalg.norm(target_rel_pos)

        r_approach = np.exp(-1.0 * dist)

        target_dir = target_rel_pos / (dist + 1e-6)
        v_target = np.dot(com_vel, target_dir)
        r_target_vel = np.clip(v_target / 3.0, -1.0, 1.0)


        facing_alignment = obs_dict.get("facing_target", np.array([0.0]))
        if isinstance(facing_alignment, np.ndarray) and facing_alignment.size > 0:
            facing_alignment = facing_alignment.flatten()[0]
        r_facing = np.clip(facing_alignment, 0.0, 1.0)

        is_at_target = dist < self.target_threshold
        r_reach = 1.0 if is_at_target else 0.0

        if is_at_target and self.current_target_idx < len(self.targets) - 1:
            self.current_target_idx += 1

        try:
            torso_quat = self.sim.data.body_xquat[self.sim.model.body_name2id("torso")]
            R_torso = quat2mat(torso_quat)
            z_up = R_torso[2, 2]
        except:
            z_up = 1.0

        r_upright = np.clip((z_up - 0.2) / 0.8, 0.0, 1.0)

        r_collision = 0.0
        for i in range(self.sim.data.ncon):
            con = self.sim.data.contact[i]
            try:
                name1 = self.sim.model.geom(con.geom1).name
                name2 = self.sim.model.geom(con.geom2).name
                if (name1 and "wall" in name1) or (name2 and "wall" in name2):
                    r_collision = 1.0
                    break
            except Exception:
                continue

        act = obs_dict.get("act", np.zeros(self.sim.model.nu))
        r_act_reg = -1.0 * np.mean(np.square(act))

        rwd_dict = collections.OrderedDict()
        rwd_dict["approach_target"] = float(np.asarray(r_approach).flatten()[0])
        rwd_dict["target_vel"] = float(np.asarray(r_target_vel).flatten()[0])

        rwd_dict["facing_reward"] = float(np.asarray(r_facing).flatten()[0])
        rwd_dict["torso_posture"] = float(np.asarray(r_upright).flatten()[0])

        rwd_dict["reach_bonus"] = float(r_reach)
        rwd_dict["collision_penalty"] = float(r_collision)
        rwd_dict["act_reg"] = float(np.asarray(r_act_reg).flatten()[0])
        rwd_dict["alive"] = 1.0

        is_done = self._get_done()
        rwd_dict["done"] = 1.0 if is_done else 0.0
        rwd_dict["sparse"] = float(r_reach)
        rwd_dict["solved"] = 1.0 if (self.current_target_idx == len(self.targets) - 1 and dist < self.target_threshold) else 0.0

        dense_reward = 0.0
        for key, weight in self.rwd_keys_wt.items():
            if key != "dense" and key in rwd_dict:
                dense_reward += weight * rwd_dict[key]

        rwd_dict["dense"] = float(np.asarray(dense_reward).flatten()[0])
        self.rwd_dict = rwd_dict
        return rwd_dict

    def _get_done(self):

        if getattr(self, '_in_setup', True) or getattr(self, 'steps', 0) < 20:
            return 0

        pelvis_pos = self.sim.data.qpos[:3]


        if pelvis_pos[2] < 0.55:
            return 1


        try:
            torso_quat = self.sim.data.body_xquat[self.sim.model.body_name2id("torso")]
            R_torso = quat2mat(torso_quat)
            z_up = R_torso[2, 2]


            if z_up < 0.2:
                return 1
        except:
            pass


        try:
            head_id = self.sim.model.body_name2id("head")
            head_z = self.sim.data.body_xpos[head_id][2]
            if head_z < 0.8:
                return 1
        except:

            try:
                cervical_id = self.sim.model.body_name2id("cervical")
                head_z = self.sim.data.body_xpos[cervical_id][2]
                if head_z < 0.8:
                    return 1
            except: pass


        final_dist = np.linalg.norm(self.targets[-1][:2] - pelvis_pos[:2])
        if self.current_target_idx == len(self.targets) - 1 and final_dist < self.target_threshold:
            return 1


        return 0


class MSKBenchSidestepEnvV0(WalkEnvV0):
    """MSK-Bench task environment."""

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "height", "muscle_length",
        "muscle_velocity", "muscle_force",
        "phase_clock",
        "feet_rel_y_dist"
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {

        "sideway_vel": 400.0,
        "gait_step_sync": 200.0,


        "com_offset_penalty": -500.0,
        "feet_spacing_penalty": -400.0,


        "torso_leaning_penalty": -500.0,
        "lumbar_penalty": -500.0,
        "pelvis_bounce_penalty": -100.0,

        "forward_vel_penalty": -200.0,
        "yaw_penalty": -150.0,
        "legs_crossed_penalty": -300.0,

        "act_reg": -1.0,
        "alive": 5.0,
        "done": -300.0,
        "sparse": 0.0,
        "solved": 0.0,
    }

    def _setup(self, **kwargs):
        self._in_setup = True

        self.target_y_vel = 1.0
        self.phase = 0.0
        self.step_freq = 1.5

        def _get_id_safe(names):
            for name in names:
                try: return self.sim.model.body_name2id(name)
                except: pass
            return -1

        self.pelvis_id = _get_id_safe(["pelvis"])
        self.head_id = _get_id_safe(["head", "cervical"])
        self.foot_l_id = _get_id_safe(["talus_l", "calcn_l", "foot_l"])
        self.foot_r_id = _get_id_safe(["talus_r", "calcn_r", "foot_r"])

        super()._setup(
            obs_keys=self.DEFAULT_OBS_KEYS,
            weighted_reward_keys=self.DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs
        )
        self._in_setup = False
        print("MSK-Bench task message.")

    def reset(self, **kwargs):
        self.steps = 0
        self.phase = 0.0

        if 'seed' in kwargs: del kwargs['seed']
        if 'options' in kwargs: del kwargs['options']

        qpos = self.sim.model.qpos0.copy()
        qvel = np.zeros(self.sim.model.nv)

        stance_angles = {
            "hip_flexion_l": 0.2, "hip_flexion_r": 0.2,
            "knee_angle_l": 0.3, "knee_angle_r": 0.3,
            "hip_adduction_l": -0.1, "hip_adduction_r": -0.1,
            "ankle_angle_l": -0.1, "ankle_angle_r": -0.1,
            "lumbar_extension": 0.0, "lumbar_bending": 0.0, "lumbar_rotation": 0.0
        }

        for joint, angle in stance_angles.items():
            try:
                jnt_id = self.sim.model.joint_name2id(joint)
                qpos[self.sim.model.jnt_qposadr[jnt_id]] = angle
            except: pass

        qpos[7:] += self.np_random.normal(0, 0.01, size=qpos[7:].shape)
        qvel[:] += self.np_random.normal(0, 0.01, size=qvel.shape)

        qpos[2] = 0.90

        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()

        import mujoco
        for _ in range(10):
            mujoco.mj_step(self.sim.model.ptr, self.sim.data.ptr)

        if hasattr(self, 'robot') and hasattr(self.robot, 'sync_sims'):
            self.robot.sync_sims(self.sim, self.sim_obsd)

        kwargs.pop('reset_qpos', None)
        kwargs.pop('reset_qvel', None)
        ret = super(WalkEnvV0, self).reset(reset_qpos=self.sim.data.qpos.copy(), reset_qvel=self.sim.data.qvel.copy(), **kwargs)
        return ret if isinstance(ret, tuple) else (ret, {})

    def step(self, a):
        if getattr(self, '_in_setup', True):
            try: obs = self.get_obs()
            except: obs = np.zeros(self.observation_space.shape[0] if hasattr(self, 'observation_space') else 1, dtype=np.float32)
            return obs, 0.0, False, False, {}

        self.steps += 1
        self.phase += self.dt * 2 * np.pi * self.step_freq
        self.phase = self.phase % (2 * np.pi)

        a_clipped = np.clip(np.nan_to_num(a, nan=0.0), -1.0, 1.0)

        try:
            ret = super().step(a_clipped)
            if len(ret) == 5: obs_vec, reward, done, truncated, info = ret
            else: obs_vec, reward, done, info = ret; truncated = False
        except Exception:
            obs_shape = self.observation_space.shape[0] if hasattr(self, 'observation_space') else 1
            safe_info = {f"rwd_{k}": 0.0 for k in self.DEFAULT_RWD_KEYS_AND_WEIGHTS.keys()}
            safe_info.update({"rwd_dense": -200.0, "error_flag": 1.0})
            return np.zeros(obs_shape, dtype=np.float32), -200.0, True, False, safe_info

        self.obs_dict = self.get_obs_dict(self.sim)
        rwd_dict = self.get_reward_dict(self.obs_dict)
        for k, v in rwd_dict.items(): info['rwd_' + k] = v

        final_obs = self.get_obs()
        safe_reward = float(np.clip(rwd_dict['dense'], -1500.0, 5000.0))
        final_done = bool(self._get_done())

        return final_obs, safe_reward, final_done, bool(truncated), info

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        try:
            obs_dict["phase_clock"] = np.array([np.sin(self.phase), np.cos(self.phase)], dtype=np.float32)
            if self.foot_l_id != -1 and self.foot_r_id != -1:
                y_l = sim.data.body_xpos[self.foot_l_id][1]
                y_r = sim.data.body_xpos[self.foot_r_id][1]
                obs_dict["feet_rel_y_dist"] = np.array([y_l - y_r], dtype=np.float32)
            else:
                obs_dict["feet_rel_y_dist"] = np.array([0.0], dtype=np.float32)
        except:
            obs_dict["phase_clock"] = np.zeros(2, dtype=np.float32)
            obs_dict["feet_rel_y_dist"] = np.zeros(1, dtype=np.float32)
        return obs_dict

    def get_reward_dict(self, obs_dict):
        rwd_dict = collections.OrderedDict()
        for k in self.rwd_keys_wt.keys(): rwd_dict[k] = 0.0

        try:
            com_vel = self._get_com_velocity()
            v_x = com_vel[0]
            v_y = com_vel[1]

            r_sideway_vel = float(np.clip(v_y / self.target_y_vel, -0.2, 1.0))
            r_forward_vel_penalty = float(abs(v_x))

            R_pelvis = quat2mat(self.sim.data.qpos[3:7])
            y_fwd = R_pelvis[1, 0]
            pelvis_z_up = R_pelvis[2, 2]
            r_yaw_penalty = float(abs(y_fwd) * 2.0)

            spine_z_up = 1.0
            if self.head_id != -1 and self.pelvis_id != -1:
                head_pos = self.sim.data.body_xpos[self.head_id]
                pelvis_pos = self.sim.data.body_xpos[self.pelvis_id]
                spine_vec = head_pos - pelvis_pos
                spine_z_up = spine_vec[2] / (np.linalg.norm(spine_vec) + 1e-6)

            r_torso_leaning_penalty = float((1.0 - pelvis_z_up) + (1.0 - spine_z_up))

            lumbar_err = 0.0
            for jnt in ["lumbar_extension", "lumbar_bending", "lumbar_rotation"]:
                try:
                    adr = self.sim.model.jnt_qposadr[self.sim.model.joint_name2id(jnt)]
                    lumbar_err += abs(self.sim.data.qpos[adr])
                except: pass
            r_lumbar_penalty = float(lumbar_err)

            v_z = self.sim.data.qvel[2]
            r_pelvis_bounce_penalty = float(abs(v_z))

            r_legs_crossed_penalty = 0.0
            r_gait_step_sync = 0.0
            r_com_offset_penalty = 0.0
            r_feet_spacing_penalty = 0.0

            if self.foot_l_id != -1 and self.foot_r_id != -1:
                f_l = self.sim.data.body_xpos[self.foot_l_id]
                f_r = self.sim.data.body_xpos[self.foot_r_id]

                y_l = f_l[1]
                y_r = f_r[1]


                feet_y_mid = (y_l + y_r) / 2.0
                pelvis_y = self.sim.data.body_xpos[self.pelvis_id][1]
                r_com_offset_penalty = float(abs(pelvis_y - feet_y_mid))



                dist_xy = np.linalg.norm(f_l[:2] - f_r[:2])

                spacing_err = max(0.0, dist_xy - 0.3)
                r_feet_spacing_penalty = float(spacing_err)

                if y_r > y_l - 0.05:
                    r_legs_crossed_penalty = min(2.0, (y_r - (y_l - 0.05)) * 10.0)

                z_l = f_l[2]
                z_r = f_r[2]

                target_z_l = 0.02 + 0.08 * max(0, np.sin(self.phase))
                target_z_r = 0.02 + 0.08 * max(0, np.sin(self.phase + np.pi))

                err_l = abs(z_l - target_z_l)
                err_r = abs(z_r - target_z_r)
                r_gait_step_sync = np.exp(-15.0 * (err_l + err_r))

            act_mag = np.mean(np.square(obs_dict.get('act', np.zeros(1))))
            is_done = self._get_done()

            rwd_dict["sideway_vel"] = r_sideway_vel
            rwd_dict["gait_step_sync"] = float(r_gait_step_sync)

            rwd_dict["com_offset_penalty"] = r_com_offset_penalty
            rwd_dict["feet_spacing_penalty"] = r_feet_spacing_penalty

            rwd_dict["torso_leaning_penalty"] = r_torso_leaning_penalty
            rwd_dict["lumbar_penalty"] = r_lumbar_penalty
            rwd_dict["pelvis_bounce_penalty"] = r_pelvis_bounce_penalty

            rwd_dict["forward_vel_penalty"] = r_forward_vel_penalty
            rwd_dict["yaw_penalty"] = r_yaw_penalty
            rwd_dict["legs_crossed_penalty"] = float(r_legs_crossed_penalty)
            rwd_dict["act_reg"] = float(act_mag)
            rwd_dict["alive"] = 1.0
            rwd_dict["done"] = float(1.0 if is_done else 0.0)

            total_rwd = sum([self.rwd_keys_wt.get(k, 0.0) * rwd_dict[k] for k in self.rwd_keys_wt.keys()])
            rwd_dict["dense"] = float(total_rwd)

        except Exception: pass
        return rwd_dict

    def _get_done(self):
        if getattr(self, '_in_setup', True): return False

        try:
            if not np.isfinite(self.sim.data.qpos).all() or not np.isfinite(self.sim.data.qvel).all():
                return True

            pelvis_pos = self.sim.data.qpos[:3]
            if pelvis_pos[2] < 0.45:
                return True

            if self.head_id != -1 and self.pelvis_id != -1:
                head_pos = self.sim.data.body_xpos[self.head_id]
                spine_vec = head_pos - pelvis_pos
                spine_z_up = spine_vec[2] / (np.linalg.norm(spine_vec) + 1e-6)
                if spine_z_up < 0.85:
                    return True

            R_pelvis = quat2mat(self.sim.data.qpos[3:7])
            if R_pelvis[2, 2] < 0.75:
                return True


            if self.foot_l_id != -1 and self.foot_r_id != -1:
                f_l = self.sim.data.body_xpos[self.foot_l_id]
                f_r = self.sim.data.body_xpos[self.foot_r_id]
                dist_xy = np.linalg.norm(f_l[:2] - f_r[:2])
                if dist_xy > 0.5:
                    return True

        except Exception:
            return True

        return False

    def _get_com_velocity(self):
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        cvel = -self.sim.data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]


class MSKBenchStairsEnvV0(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "height", "muscle_length",
        "muscle_velocity", "muscle_force",
        "phase_clock",
        "heading_error",
        "terrain_obs"
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "forward_vel": 50.0,
        "push_off": 30.0,
        "step_contact": 40.0,
        "climb_up": 30.0,
        "y_centering": 20.0,
        "corridor_penalty": -50.0,
        "phase_matching": 40.0,
        "approach_lift": 40.0,
        "foot_clearance": 20.0,
        "heading_lock": 30.0,
        "torso_upright": 20.0,
        "static_penalty": 30.0,
        "act_reg": 0.005,
        "done": -100.0,
        "alive": 5.0,
        "sparse": 0.0,
        "solved": 0.0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, normalize_act=True, reset_type='none', render_mode=None, **kwargs):
        self.render_mode = render_mode
        if 'render_mode' in kwargs: del kwargs['render_mode']

        self.stair_start = 0.5
        self.step_depth = 0.25
        self.step_height = 0.18

        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, normalize_act=normalize_act, **kwargs)
        self.reset_type = reset_type

    def _setup(self, obs_keys=DEFAULT_OBS_KEYS, weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS, min_height=0.75, target_x_vel=1.5, max_episode_steps=1000, reset_type="init", **kwargs):
        self.max_episode_steps = max_episode_steps
        self.min_height = min_height
        self.reset_type = reset_type
        self.target_x_vel = target_x_vel
        self.steps = 0
        self.phase = 0.0
        self.gait_freq = 1.0

        self.ref_qpos = self.sim.model.qpos0.copy()




        self.scan_dots = []
        for x in [0.3, 0.6, 0.9]:
            for y in [-0.3, 0.0, 0.3]:
                self.scan_dots.append([x, y])
        self.scan_dots = np.array(self.scan_dots)

        super(WalkEnvV0, self)._setup(obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, sites=[], **kwargs)
        self.init_qpos[:] = self.sim.model.qpos0.copy()
        self.original_mass = self.sim.model.body_mass.copy()

    def reset(self, **kwargs):
        if 'seed' in kwargs: del kwargs['seed']
        if 'options' in kwargs: del kwargs['options']
        self.steps = 0
        self.phase = 0.0
        qpos = self.sim.model.qpos0.copy()
        qvel = np.zeros(self.sim.model.nv)
        qpos[2] = 0.95
        qpos[0] = 0.3
        qpos[3:7] = [1, 0, 0, 0]

        random_scale = self.np_random.uniform(0.95, 1.05, size=self.original_mass.shape)
        self.sim.model.body_mass[:] = self.original_mass * random_scale

        self._set_joint_angle(qpos, "knee_angle_l", 0.3)
        self._set_joint_angle(qpos, "knee_angle_r", 0.3)
        self._set_joint_angle(qpos, "hip_flexion_l", 0.3)
        self._set_joint_angle(qpos, "hip_flexion_r", 0.3)
        self._set_joint_angle(qpos, "ankle_angle_l", 0.1)
        self._set_joint_angle(qpos, "ankle_angle_r", 0.1)

        qvel[0] = 0.5

        qpos[:] += self.np_random.normal(0, 0.005, size=qpos.shape)
        qvel[:] += self.np_random.normal(0, 0.005, size=qvel.shape)
        self.robot.sync_sims(self.sim, self.sim_obsd)

        kwargs.pop('reset_qpos', None)
        kwargs.pop('reset_qvel', None)
        ret = super(WalkEnvV0, self).reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)

        if isinstance(ret, tuple) and len(ret) == 2: obs, info = ret
        else: obs = ret; info = {}
        return obs if self.render_mode is None else (obs, info)

    def step(self, a):
        self.steps += 1
        self.phase += self.dt * 2 * np.pi * self.gait_freq
        step_result = super().step(a)

        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False

        rwd_dict = self.get_reward_dict(self.get_obs_dict(self.sim))
        reward = sum(rwd_dict.values())
        info['rwd_dict'] = rwd_dict

        if len(step_result) == 5:
            return obs, reward, terminated, truncated, info
        else:
            return obs, reward, done, info

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        obs_dict["phase_clock"] = np.array([np.sin(self.phase), np.cos(self.phase)])

        quat = sim.data.qpos[3:7]
        R = quat2mat(quat)
        forward_vec = R @ np.array([1, 0, 0])
        obs_dict["heading_error"] = np.array([forward_vec[1]])

        pelvis_pos = sim.data.body("pelvis").xpos



        #    World_Point = Pelvis_Pos + Rotation * Relative_Point


        scan_heights = []


        def get_ground_height(x):
            if x < self.stair_start: return 0.0
            idx = int((x - self.stair_start) / self.step_depth)

            idx = max(0, min(idx, 20))
            return (idx + 1) * self.step_height

        for dot in self.scan_dots:

            dot_3d = np.array([dot[0], dot[1], 0])

            world_offset = R @ dot_3d

            check_x = pelvis_pos[0] + world_offset[0]
            check_y = pelvis_pos[1] + world_offset[1]


            g_height = get_ground_height(check_x)








            rel_h = g_height - pelvis_pos[2]
            scan_heights.append(rel_h)

        obs_dict["terrain_obs"] = np.array(scan_heights)
        return obs_dict

    def get_reward_dict(self, obs_dict):
        com_vel = self._get_com_velocity()
        x_vel = com_vel[0]
        z_vel = com_vel[2]

        forward_vel = np.clip(x_vel, -0.2, 2.0)

        static_penalty = 0.0
        if x_vel < 0.2: static_penalty = -1.0

        climb_up = np.clip(z_vel, -0.5, 3.0)

        quat = self.sim.data.qpos[3:7].copy()
        R = quat2mat(quat)
        torso_vec = R @ np.array([0, 0, 1])
        forward_vec = R @ np.array([1, 0, 0])

        heading_lock = np.exp(-10.0 * (1.0 - forward_vec[0]))
        torso_upright = np.exp(-5.0 * (0.98 - torso_vec[2])**2)

        pelvis_y = self.sim.data.qpos[1]
        corridor_penalty = 0.0
        if abs(pelvis_y) > 0.6: corridor_penalty = abs(pelvis_y) - 0.6
        y_centering = np.exp(-5.0 * pelvis_y**2)

        clock = np.sin(self.phase)
        step_contact = 0.0



        terrain_obs = obs_dict["terrain_obs"]

        avg_front_height = np.mean(terrain_obs[:3])



        is_facing_stair = (avg_front_height > -0.85)

        approach_lift = 0.0
        push_off = 0.0
        phase_matching = 0.0
        foot_clearance = 0.0

        try:
            f_l_id = self.sim.model.body_name2id("talus_l")
            f_r_id = self.sim.model.body_name2id("talus_r")

            vel_l = self.sim.data.cvel[f_l_id][:3]
            vel_r = self.sim.data.cvel[f_r_id][:3]
            h_l = self.sim.data.body_xpos[f_l_id][2]
            h_r = self.sim.data.body_xpos[f_r_id][2]


            if h_l > 0.1: step_contact += 0.1

            if clock > 0:
                phase_matching += np.clip(vel_l[2], -1, 3)
                push_off += 1.0 if x_vel > 0.2 else 0.0


                if is_facing_stair and x_vel > 0.1:
                    approach_lift += 5.0 * np.clip(vel_l[2], 0, 3)

                    if h_l > 0.3: approach_lift += 5.0
            else:
                phase_matching += np.clip(vel_r[2], -1, 3)
                push_off += 1.0 if x_vel > 0.2 else 0.0

                if is_facing_stair and x_vel > 0.1:
                    approach_lift += 5.0 * np.clip(vel_r[2], 0, 3)
                    if h_r > 0.3: approach_lift += 5.0

            if h_l > 0.05: foot_clearance += 1.0
            if h_r > 0.05: foot_clearance += 1.0

        except: pass

        if "act" in obs_dict:
            act_mag = np.linalg.norm(obs_dict["act"]) / self.sim.model.na if self.sim.model.na != 0 else 0
        else: act_mag = 0.0

        done = self._get_done()
        is_solved = (self._get_height() > 2.0)

        rwd_dict = collections.OrderedDict((
            ("forward_vel", forward_vel),
            ("push_off", push_off),
            ("phase_matching", phase_matching),
            ("approach_lift", approach_lift),
            ("static_penalty", static_penalty),
            ("climb_up", climb_up),
            ("heading_lock", heading_lock),
            ("torso_upright", torso_upright),
            ("foot_clearance", foot_clearance),
            ("corridor_penalty", -50.0 * corridor_penalty),
            ("y_centering", y_centering),
            ("step_contact", step_contact),
            ("act_reg", -1.0 * act_mag),
            ("alive", 1.0),
            ("done", 1.0 if done else 0.0),
            ("sparse", 0.0),
            ("solved", is_solved),
        ))
        rwd_dict["dense"] = np.sum([wt * rwd_dict.get(key, 0.0) for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def _get_done(self):
        if self.steps >= self.max_episode_steps: return 1
        if self.sim.data.qpos[2] < self.min_height: return 1

        if abs(self.sim.data.qpos[1]) > 1.0: return 1

        try:
            head_z = self.sim.data.body("head").xpos[2]
            if head_z < 1.0: return 1
        except: pass

        quat = self.sim.data.qpos[3:7]
        R = quat2mat(quat)
        z_up = (R @ np.array([0, 0, 1]))[2]
        if z_up < 0.5: return 1

        forward_vec = R @ np.array([1, 0, 0])
        if forward_vec[0] < 0.5: return 1

        return 0

    def muscle_lengths(self): return self.sim.data.actuator_length
    def muscle_forces(self): return np.clip(self.sim.data.actuator_force / 1000, -100, 100)
    def muscle_velocities(self): return np.clip(self.sim.data.actuator_velocity, -100, 100)
    def _get_torso_angle(self): return self.sim.data.body_xquat[self.sim.model.body_name2id("torso")]
    def _get_height(self): return self.sim.data.qpos[2]
    def _get_com_velocity(self): return self.sim.data.qvel[:3]
    def _get_joint_angle(self, name):
        try:
            id = self.sim.model.joint_name2id(name)
            addr = self.sim.model.jnt_qposadr[id]
            return self.sim.data.qpos[addr]
        except: return 0.0
    def _set_joint_angle(self, qpos_arr, name, val):
        try:
            id = self.sim.model.joint_name2id(name)
            addr = self.sim.model.jnt_qposadr[id]
            qpos_arr[addr] = val
        except: pass


class MSKBenchHurdleEnvV0(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "height", "muscle_length",
        "muscle_velocity", "muscle_force",
        "nearest_hurdle_rel",
        "phase_clock"
    ]


    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "forward_vel": 300.0,
        "idle_penalty": -100.0,


        "hip_lift_bonus": 100.0,
        "foot_clearance_bonus": 150.0,
        "hurdle_flight_phase": 100.0,


        "crane_penalty": -150.0,

        "heading_penalty": -50.0,
        "yaw_penalty": -150.0,
        "knee_overbend_penalty": -200.0,

        "hurdle_crash_penalty": 0.0,
        "jump_ramp_penalty": 0.0,
        "landing_ramp_penalty": 0.0,

        "hurdle_clearance": 3000.0,

        "torso_upright": 30.0,
        "side_lean_penalty": -30.0,

        "act_reg": -2.0,
        "act_rate_penalty": -2.0,

        "done": -100.0,
        "alive": 5.0,

        "sparse": 0.0,
        "solved": 0.0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, normalize_act=True, reset_type='none', render_mode=None, max_episode_steps=1000, **kwargs):
        self._in_setup = True
        self.steps = 0
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.last_action = None
        self.hurdles_cleared = 0

        if 'render_mode' in kwargs: del kwargs['render_mode']

        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, normalize_act=normalize_act, **kwargs)
        self.reset_type = reset_type
        self._in_setup = False
        print("MSK-Bench task message.")

    def _setup(self, obs_keys=DEFAULT_OBS_KEYS, weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS, **kwargs):
        self._in_setup = True

        def _get_id_safe(names, type='body'):
            for name in names:
                try:
                    if type == 'body': return self.sim.model.body_name2id(name)
                    if type == 'site': return self.sim.model.site_name2id(name)
                    if type == 'joint': return self.sim.model.joint_name2id(name)
                except: pass
            return -1

        self.head_id = _get_id_safe(["head", "cervical"])
        self.pelvis_id = _get_id_safe(["pelvis"])
        self.foot_l_id = _get_id_safe(["talus_l", "calcn_l", "foot_l"])
        self.foot_r_id = _get_id_safe(["talus_r", "calcn_r", "foot_r"])

        hurdle_list = []
        for i in range(1, 10):
            body_name = f"h{i}"
            try:
                body_id = self.sim.model.body_name2id(body_name)
                geom_id = self.sim.model.body_geomadr[body_id]
                geom_size = self.sim.model.geom_size[geom_id]
                body_pos = self.sim.model.body_pos[body_id]

                x_pos = body_pos[0]
                top_h = body_pos[2] + geom_size[2]

                hurdle_list.append([x_pos, top_h])
            except Exception:
                pass

        self.hurdles = np.array(hurdle_list)
        if len(self.hurdles) == 0:
            print("MSK-Bench task message.")
            self.hurdles = np.array([[2.0, 0.2], [4.0, 0.3], [6.0, 0.4], [8.0, 0.5], [10.0, 0.5]])
        else:
            print(f"Obstacle scan completed. Detected hurdles:\n{self.hurdles}")

        self.phase = 0.0
        self.gait_freq = 1.0

        if 'min_height' not in kwargs: kwargs['min_height'] = 0.8
        if 'target_x_vel' not in kwargs: kwargs['target_x_vel'] = 1.2

        super()._setup(obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, sites=[], **kwargs)
        self._in_setup = False

    def reset(self, *, seed=None, options=None, **kwargs):
        """
        Compatible reset for SB3 2.x / Gymnasium and MSK-Bench old-style envs.

        SB3 2.x SubprocVecEnv expects:
            obs, info = env.reset(seed=..., options=...)

        But many MSK-Bench envs return only:
            obs = env.reset()

        This wrapper normalizes both cases to:
            return obs, info
        """

        # 1. Reset your custom episode variables
        self.steps = 0
        self.phase = 0.0
        self.last_action = None
        self.hurdles_cleared = 0

        # 2. Prepare kwargs for parent reset
        reset_kwargs = dict(kwargs)

        if seed is not None:
            reset_kwargs["seed"] = seed

        if options is not None:
            reset_kwargs["options"] = options

        # 3. Try parent reset with Gymnasium-style arguments first
        try:
            ret = super().reset(**reset_kwargs)

        except TypeError:
            # 4. If MSK-Bench / old Gym does not accept seed/options, remove them
            reset_kwargs.pop("seed", None)
            reset_kwargs.pop("options", None)

            # Try to seed manually if possible
            if seed is not None:
                try:
                    self.seed(seed)
                except Exception:
                    pass

            ret = super().reset(**reset_kwargs)

        # 5. Normalize return format
        # Case A: parent already returns (obs, info)
        if isinstance(ret, tuple):
            if len(ret) == 2 and isinstance(ret[1], dict):
                obs, info = ret
                return obs, info

            # Case B: weird tuple, take first element as obs
            obs = ret[0]
            info = {}
            return obs, info

        # Case C: old Gym / MSK-Bench returns only obs
        obs = ret
        info = {}
        return obs, info

    def step(self, a):
        if getattr(self, '_in_setup', True):
            try:
                obs = self.get_obs()
            except:
                obs = np.zeros(self.observation_space.shape if hasattr(self, 'observation_space') else 1, dtype=np.float32)
            return obs, 0.0, False, False, {}

        self.steps += 1
        self.phase += self.dt * 2 * np.pi * self.gait_freq

        a = np.nan_to_num(a, nan=0.0)
        a_clipped = np.clip(a, -1.0, 1.0)

        try:
            ret = super().step(a_clipped)
            if len(ret) == 4:
                obs_vec, reward, done, info = ret
                truncated = False
            else:
                obs_vec, reward, done, truncated, info = ret
        except Exception as e:
            if hasattr(self, 'observation_space'):
                dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            else:
                try: dummy_obs = self.get_obs()
                except: dummy_obs = np.zeros(1, dtype=np.float32)
            safe_info = {f"rwd_{k}": 0.0 for k in self.DEFAULT_RWD_KEYS_AND_WEIGHTS.keys()}
            safe_info.update({"rwd_dense": -200.0, "error_flag": 1.0})
            return dummy_obs, -200.0, True, False, safe_info

        self.obs_dict = self.get_obs_dict(self.sim)
        self.obs_dict['current_action'] = a_clipped

        rwd_dict = self.get_reward_dict(self.obs_dict)
        self.last_action = a_clipped

        for k, v in rwd_dict.items():
            info['rwd_' + k] = v

        final_obs = self.get_obs()
        safe_reward = float(np.clip(rwd_dict['dense'], -1000.0, 5000.0))
        final_done = bool(self._get_done())

        return final_obs, safe_reward, final_done, bool(truncated), info

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        try:
            pelvis_pos = sim.data.body_xpos[self.pelvis_id] if self.pelvis_id != -1 else sim.data.qpos[:3]
            if self.hurdles_cleared < len(self.hurdles):
                target_hurdle = self.hurdles[self.hurdles_cleared]
                obs_dict["nearest_hurdle_rel"] = np.array([
                    target_hurdle[0] - pelvis_pos[0],
                    target_hurdle[1],
                    0.0
                ])
            else:
                obs_dict["nearest_hurdle_rel"] = np.zeros(3)

            obs_dict["phase_clock"] = np.array([np.sin(self.phase), np.cos(self.phase)])
        except:
            obs_dict["nearest_hurdle_rel"] = np.zeros(3)
            obs_dict["phase_clock"] = np.zeros(2)
        return obs_dict

    def _get_spine_vector(self):
        try:
            p_pos = self.sim.data.body_xpos[self.pelvis_id] if self.pelvis_id != -1 else self.sim.data.qpos[:3]
            h_pos = self.sim.data.body_xpos[self.head_id]
            vec = h_pos - p_pos
            norm = np.linalg.norm(vec)
            if norm > 1e-6: return vec / norm
        except: pass
        return np.array([0.0, 0.0, 1.0])

    def get_reward_dict(self, obs_dict):
        com_vel = self._get_com_velocity()

        is_moving = 1.0 if com_vel[0] > 0.5 else 0.0
        forward_vel = np.clip(com_vel[0], -0.1, 4.0)
        idle_penalty = 1.0 if com_vel[0] < 0.5 else 0.0


        v_x_multiplier = np.clip(com_vel[0], 0.0, 3.0)

        hurdle_clearance = 0.0
        hurdle_crash_penalty = 0.0
        jump_ramp_penalty = 0.0
        landing_ramp_penalty = 0.0
        hurdle_flight_phase = 0.0
        hip_lift_bonus = 0.0
        foot_clearance_bonus = 0.0
        crane_penalty = 0.0

        try:
            pelvis_pos = self.sim.data.body_xpos[self.pelvis_id] if self.pelvis_id != -1 else self.sim.data.qpos[:3]
            f_l = self.sim.data.body_xpos[self.foot_l_id] if self.foot_l_id != -1 else pelvis_pos
            f_r = self.sim.data.body_xpos[self.foot_r_id] if self.foot_r_id != -1 else pelvis_pos

            y_deviation = abs(pelvis_pos[1])
            y_velocity = abs(com_vel[1])
            heading_penalty = y_velocity + y_deviation * 2.0

            spine_vec = self._get_spine_vector()
            z_up = spine_vec[2]

            hip_fl_val, hip_fr_val = 0.0, 0.0
            knee_overbend_penalty = 0.0
            try:
                hip_fl_val = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id("hip_flexion_l")]]
                hip_fr_val = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id("hip_flexion_r")]]
                knee_l_val = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id("knee_angle_l")]]
                knee_r_val = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id("knee_angle_r")]]

                if knee_l_val > 1.2 and knee_r_val > 1.2:
                    knee_overbend_penalty = np.clip(max(knee_l_val, knee_r_val) - 1.2, 0.0, 1.0)
            except: pass

            rel_info = obs_dict["nearest_hurdle_rel"]
            dist_x = rel_info[0]
            hurdle_h = rel_info[1]

            target_hurdle_x = pelvis_pos[0] + dist_x
            foot_l_dist = target_hurdle_x - f_l[0]
            foot_r_dist = target_hurdle_x - f_r[0]
            lead_foot_dist = min(foot_l_dist, foot_r_dist)


            if 0.0 < lead_foot_dist < 0.8 and com_vel[0] > 0.3:
                max_hip_flexion = max(hip_fl_val, hip_fr_val)
                if max_hip_flexion > 0.3:
                    hip_lift_bonus = np.clip((max_hip_flexion - 0.3) / 1.0, 0.0, 1.0) * v_x_multiplier


            if abs(dist_x) < 0.4:
                max_foot_z = max(f_l[2], f_r[2])
                if max_foot_z > hurdle_h - 0.05:
                    foot_clearance_bonus = np.clip((max_foot_z - (hurdle_h - 0.05)) / 0.2, 0.0, 1.0) * v_x_multiplier


            if abs(dist_x) < 0.3 and is_moving:
                if max(f_l[2], f_r[2]) > hurdle_h:
                    hurdle_flight_phase = 1.0 * v_x_multiplier




            if dist_x > 0.6:
                foot_z_diff = abs(f_l[2] - f_r[2])
                if foot_z_diff > 0.2:
                    crane_penalty = np.clip((foot_z_diff - 0.2) / 0.5, 0.0, 1.0)


            if dist_x < 0.0:
                if y_deviation < 0.5:
                    hurdle_clearance = 1.0
                self.hurdles_cleared += 1

        except:
            heading_penalty = 0.0
            spine_vec = np.array([0,0,1])
            z_up = 1.0

        torso_upright = np.exp(-20.0 * (1.0 - z_up)**2) * is_moving

        try:
            pelvis_quat = self.sim.data.qpos[3:7]
            mat = quat2mat(pelvis_quat)
            y_fwd = (mat @ np.array([1, 0, 0]))[1]
            yaw_penalty = np.abs(y_fwd)
        except:
            yaw_penalty = 0.0

        side_lean_penalty = 1.0 if abs(spine_vec[1]) > 0.15 else 0.0

        curr_act = obs_dict.get('current_action', np.zeros(self.sim.model.nu))
        act_mag = np.mean(np.square(curr_act))
        act_rate = np.mean(np.square(curr_act - self.last_action)) if self.last_action is not None else 0.0

        done = self._get_done()

        rwd_dict = collections.OrderedDict((
            ("forward_vel", float(forward_vel)),
            ("idle_penalty", float(idle_penalty)),
            ("hip_lift_bonus", float(hip_lift_bonus)),
            ("foot_clearance_bonus", float(foot_clearance_bonus)),
            ("hurdle_flight_phase", float(hurdle_flight_phase)),
            ("crane_penalty", float(crane_penalty)),
            ("heading_penalty", float(heading_penalty)),
            ("yaw_penalty", float(yaw_penalty)),
            ("knee_overbend_penalty", float(knee_overbend_penalty)),
            ("hurdle_crash_penalty", float(hurdle_crash_penalty)),
            ("jump_ramp_penalty", float(jump_ramp_penalty)),
            ("landing_ramp_penalty", float(landing_ramp_penalty)),
            ("hurdle_clearance", float(hurdle_clearance)),
            ("torso_upright", float(torso_upright)),
            ("side_lean_penalty", float(side_lean_penalty)),
            ("act_reg", float(act_mag)),
            ("act_rate_penalty", float(act_rate)),
            ("alive", 1.0),
            ("done", float(1.0 if done else 0.0)),
            ("sparse", 0.0),
            ("solved", 0.0),
        ))

        rwd_dict["dense"] = np.sum([self.rwd_keys_wt.get(key, 0.0) * rwd_dict.get(key, 0.0) for key in self.rwd_keys_wt.keys()])
        return rwd_dict

    def _get_done(self):
        if getattr(self, '_in_setup', True): return 0
        try:
            if not np.isfinite(self.sim.data.qpos).all() or not np.isfinite(self.sim.data.qvel).all(): return 1
            if self.steps >= self.max_episode_steps: return 1

            if self.sim.data.qpos[2] < 0.55: return 1

            spine_vec = self._get_spine_vector()
            z_up = spine_vec[2]

            if z_up < 0.5: return 1

            pelvis_quat = self.sim.data.qpos[3:7]
            mat = quat2mat(pelvis_quat)
            y_fwd = (mat @ np.array([1, 0, 0]))[1]
            if abs(y_fwd) > 0.8:
                return 1

        except: pass
        return 0

    def _get_torso_angle(self): return self.sim.data.body_xquat[self.sim.model.body_name2id("torso")]
    def _get_com_velocity(self): return self.sim.data.qvel[:2].copy()


class MSKBenchStepStonesEnvV0(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "height", "muscle_length",
        "muscle_velocity", "muscle_force",
        "next_stones_rel",
        "landing_target_rel",
        "phase_clock"
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {

        "forward_vel": 100.0,
        "stone_progression": 800.0,
        "magnet_foot_reach": 150.0,


        "torso_upright": 50.0,
        "stand_tall": 50.0,


        "straight_facing": 100.0,


        "idle_penalty": 50.0,
        "knee_overbend_penalty": -50.0,
        "deviation_penalty": -20.0,


        "yaw_penalty": -100.0,

        "act_reg": -1.0,
        "done": -300.0,
    }

    def _setup(self, obs_keys=DEFAULT_OBS_KEYS, weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS, max_episode_steps=1000, **kwargs):
        self.steps = 0
        self.phase = 0.0
        self.gait_freq = 0.85
        self.current_stone_idx = 0
        self.max_episode_steps = max_episode_steps
        self.base_foot_z = 0.0

        self.stones = np.array([
            [0.00, 0.00], [0.40, 0.05], [0.80, -0.05], [1.25, 0.02], [1.70, -0.02],
            [2.20, 0.05], [2.70, -0.05], [3.20, 0.00], [3.70, 0.00], [4.50, 0.00]
        ])

        self.torso_id = self.sim.model.body_name2id("torso")
        self.pelvis_id = self.sim.model.body_name2id("pelvis")
        self.foot_l_id = self.sim.model.body_name2id("talus_l")
        self.foot_r_id = self.sim.model.body_name2id("talus_r")

        super()._setup(obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, sites=[], max_episode_steps=max_episode_steps, **kwargs)

        self.init_qpos[:] = self.sim.model.qpos0.copy()
        self.init_qvel[:] = 0.0

        print("MSK-Bench task message.")

    def reset(self, *, seed=None, options=None, **kwargs):
        # ==============================

        # ==============================
        if seed is not None:
            try:
                from gymnasium.utils import seeding
            except ImportError:
                from gym.utils import seeding

            self.np_random, _ = seeding.np_random(seed)


        kwargs.pop("seed", None)
        kwargs.pop("options", None)

        # ==============================

        # ==============================
        self.steps = 0
        self.phase = 0.0
        self.current_stone_idx = 0

        # ==============================

        # ==============================
        try:
            ret = super().reset(seed=seed, options=options, **kwargs)
        except TypeError:
            ret = super().reset(**kwargs)

        if isinstance(ret, tuple) and len(ret) == 2:
            _, info = ret
        else:
            info = {}

        # ==============================

        # ==============================
        qpos = self.sim.model.qpos0.copy()
        qvel = np.zeros(self.sim.model.nv)

        qpos[0:2] = self.stones[0]
        qpos[2] = 3.05

        self._set_joint(qpos, "hip_flexion_l", 0.2)
        self._set_joint(qpos, "knee_angle_l", 0.1)
        self._set_joint(qpos, "hip_flexion_r", -0.1)

        qpos[7:] += self.np_random.normal(0, 0.01, size=qpos[7:].shape)
        qvel[:] += self.np_random.normal(0, 0.01, size=qvel.shape)

        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.data.ctrl[:] = 0.0
        self.sim.forward()

        # ==============================

        # ==============================
        if hasattr(self, "robot") and hasattr(self.robot, "sync_sims"):
            self.robot.sync_sims(self.sim, self.sim_obsd)

        # ==============================

        # ==============================
        import mujoco

        for _ in range(15):
            mujoco.mj_step(self.sim.model.ptr, self.sim.data.ptr)

        # ==============================

        # ==============================
        f_l_z = self.sim.data.body_xpos[self.foot_l_id][2]
        f_r_z = self.sim.data.body_xpos[self.foot_r_id][2]
        self.base_foot_z = min(f_l_z, f_r_z)

        # ==============================

        # ==============================
        obs = self.get_obs()


        return obs, info

    def step(self, a):
        self.steps += 1
        self.phase += self.dt * 2 * np.pi * self.gait_freq

        if np.isnan(a).any() or np.isinf(a).any():
            a = np.zeros_like(a)

        return super().step(a)

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        try:
            p_pos = sim.data.body_xpos[self.pelvis_id]
            target_idx = min(self.current_stone_idx + 1, len(self.stones)-1)
            target_stone = self.stones[target_idx]

            rel_1 = target_stone - p_pos[:2]
            rel_2 = self.stones[min(target_idx + 1, len(self.stones)-1)] - p_pos[:2]

            obs_dict["next_stones_rel"] = np.nan_to_num(np.concatenate([rel_1, rel_2]).astype(np.float32))
            obs_dict["landing_target_rel"] = np.nan_to_num(rel_1.astype(np.float32))
            obs_dict["phase_clock"] = np.array([np.sin(self.phase), np.cos(self.phase)], dtype=np.float32)
        except:
            obs_dict["next_stones_rel"] = np.zeros(4, dtype=np.float32)
            obs_dict["landing_target_rel"] = np.zeros(2, dtype=np.float32)
            obs_dict["phase_clock"] = np.zeros(2, dtype=np.float32)

        return obs_dict

    def get_reward_dict(self, obs_dict):
        try:
            f_l = self.sim.data.body_xpos[self.foot_l_id]
            f_r = self.sim.data.body_xpos[self.foot_r_id]
            p_pos = self.sim.data.body_xpos[self.pelvis_id]
            v_f = self.sim.data.qvel[0]

            is_moving = 1.0 if v_f > 0.1 else 0.0
            idle_penalty = 1.0 if v_f < 0.1 else 0.0


            torso_quat = self.sim.data.body_xquat[self.torso_id]
            R_torso = quat2mat(torso_quat)
            z_up = R_torso[2, 2]
            x_fwd = R_torso[0, 0]
            y_fwd = R_torso[1, 0]

            r_torso_upright = np.exp(-5.0 * (1.0 - z_up)**2)


            r_straight_facing = np.exp(-5.0 * (1.0 - x_fwd)**2)
            yaw_penalty = np.abs(y_fwd)

            leg_extension = p_pos[2] - min(f_l[2], f_r[2])
            r_stand_tall = np.clip((leg_extension - 0.6) / 0.3, 0.0, 1.0)

            next_stone_idx = min(self.current_stone_idx + 1, len(self.stones)-1)
            next_stone = self.stones[next_stone_idx]
            prog = 0.0
            is_solved = 0.0
            if p_pos[0] > next_stone[0] and self.current_stone_idx < len(self.stones) - 1:
                self.current_stone_idx += 1
                prog = 1.0
            if self.current_stone_idx >= len(self.stones) - 1:
                is_solved = 1.0

            valid_stones = [s for s in self.stones if s[0] > p_pos[0] + 0.05]
            target_l = next((s for s in valid_stones if s[1] >= -0.01), self.stones[-1])
            target_r = next((s for s in valid_stones if s[1] <= 0.01), self.stones[-1])

            dist_l_2d = np.linalg.norm(f_l[:2] - target_l)
            dist_r_2d = np.linalg.norm(f_r[:2] - target_r)

            lead_dist = dist_l_2d if f_l[0] > f_r[0] else dist_r_2d
            r_magnet_reach = np.exp(-2.0 * lead_dist) * is_moving

            knee_penalty = 0.0
            try:
                knee_l_val = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id("knee_angle_l")]]
                knee_r_val = self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id("knee_angle_r")]]
                if knee_l_val > 1.0: knee_penalty += np.clip(knee_l_val - 1.0, 0.0, 1.0)
                if knee_r_val > 1.0: knee_penalty += np.clip(knee_r_val - 1.0, 0.0, 1.0)
            except: pass

            deviation_penalty = np.clip(abs(p_pos[1]) - 0.2, 0.0, 1.0)

            act_mag = 0.0
            if self.sim.model.na != 0:
                act_val = np.linalg.norm(obs_dict.get("act", np.zeros(1)), axis=-1)
                if isinstance(act_val, np.ndarray): act_val = act_val.item()
                act_mag = act_val / self.sim.model.na

            done = self._get_done()

            rwd_dict = collections.OrderedDict((
                ("forward_vel", float(np.clip(v_f, 0.0, 2.0))),
                ("idle_penalty", float(-idle_penalty)),
                ("stone_progression", float(prog)),

                ("magnet_foot_reach", float(r_magnet_reach)),

                ("torso_upright", float(r_torso_upright)),
                ("stand_tall", float(r_stand_tall)),


                ("straight_facing", float(r_straight_facing)),
                ("yaw_penalty", float(yaw_penalty)),

                ("knee_overbend_penalty", float(knee_penalty)),
                ("deviation_penalty", float(deviation_penalty)),

                ("act_reg", float(act_mag)),
                ("done", 1.0 if done else 0.0),
            ))

            total_rwd = np.sum([wt * rwd_dict.get(key, 0.0) for key, wt in self.rwd_keys_wt.items()], axis=0)
            if np.isnan(total_rwd) or np.isinf(total_rwd): total_rwd = 0.0

            rwd_dict["dense"] = np.float64(total_rwd)
            rwd_dict["sparse"] = np.float64(self.current_stone_idx)
            rwd_dict["solved"] = np.float64(is_solved)

        except Exception as e:
            traceback.print_exc()
            rwd_dict = OrderedDict({k: 0.0 for k in self.rwd_keys_wt.keys()})
            rwd_dict.update({'sparse': 0.0, 'solved': 0.0, 'dense': 0.0, 'done': 0.0})

        for k in rwd_dict.keys():
            rwd_dict[k] = np.float64(rwd_dict[k])

        return rwd_dict

    def _get_done(self):
        if self.steps >= self.max_episode_steps: return True

        try:
            f_l = self.sim.data.body_xpos[self.foot_l_id]
            f_r = self.sim.data.body_xpos[self.foot_r_id]
            f_z_min = min(f_l[2], f_r[2])
            p_z = self.sim.data.body_xpos[self.pelvis_id][2]

            torso_quat = self.sim.data.qpos[3:7]
            R_torso = quat2mat(torso_quat)
            z_up = R_torso[2, 2]
            y_fwd = R_torso[1, 0]

            if z_up < 0.4: return True


            if abs(y_fwd) > 0.8: return True

            if f_z_min < 1.80: return True
            if p_z < 2.50: return True
            if (p_z - f_z_min) < 0.65: return True

        except: pass
        return False

    def _set_joint(self, qpos_arr, names, val):
        if isinstance(names, str): names = [names]
        for name in names:
            try:
                id = self.sim.model.joint_name2id(name)
                addr = self.sim.model.jnt_qposadr[id]
                qpos_arr[addr] = val
            except: pass


class MSKBenchSlideEnvV0(WalkEnvV0):
    """MSK-Bench task environment."""

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "height", "muscle_length",
        "muscle_velocity", "muscle_force",
        "slope_obs",
        "phase_clock"
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {

        "forward_vel": 50.0,
        "climb_up": 30.0,


        "idle_penalty": -20.0,


        "foot_align": 50.0,
        "ankle_flex": 30.0,
        "lean_forward": 40.0,


        "stand_tall": 50.0,
        "foot_contact": 20.0,
        "high_step": 30.0,
        "knee_power": 30.0,


        "illegal_contact": -50.0,
        "drag_penalty": 5.0,
        "act_reg": 0.005,
        "done": -100.0,
        "alive": 5.0,
        "sparse": 0.0,
        "solved": 0.0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, normalize_act=True, reset_type='none', render_mode=None, **kwargs):
        self.steps = 0
        self.max_episode_steps = 1000
        self.render_mode = render_mode


        if 'render_mode' in kwargs:
            del kwargs['render_mode']


        self.torso_id = 0
        self.head_id = 0
        self.foot_l_id = 0
        self.foot_r_id = 0
        self.pelvis_id = 0
        self.geom_foot_l = -1
        self.geom_foot_r = -1

        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, normalize_act=normalize_act, **kwargs)
        self.reset_type = reset_type

    def _setup(self, obs_keys=DEFAULT_OBS_KEYS, weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS, **kwargs):
        self.min_height = kwargs.get('min_height', 0.65)
        self.target_x_vel = kwargs.get('target_x_vel', 1.0)
        self.steps = 0
        self.phase = 0.0
        self.gait_freq = 1.0

        try:
            self.torso_id = self.sim.model.body_name2id("torso")
            self.head_id = self.sim.model.body_name2id("head")
            self.foot_l_id = self.sim.model.body_name2id("talus_l")
            self.foot_r_id = self.sim.model.body_name2id("talus_r")
            self.pelvis_id = self.sim.model.body_name2id("pelvis")

            names = self.sim.model.geom_names
            self.geom_foot_l = self.sim.model.geom_name2id("foot_l_geom") if "foot_l_geom" in names else -1
            self.geom_foot_r = self.sim.model.geom_name2id("foot_r_geom") if "foot_r_geom" in names else -1
        except Exception as e:
            print(f"Warning: Model IDs not found: {e}")

        super(WalkEnvV0, self)._setup(obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, sites=[], **kwargs)
        self.init_qpos[:] = self.sim.model.qpos0.copy()

    def reset(self, **kwargs):
        self.steps = 0
        self.phase = 0.0
        qpos = self.sim.model.qpos0.copy()
        qvel = np.zeros(self.sim.model.nv)
        qpos[2] = 0.95


        self._set_joint_angle(qpos, "hip_flexion_l", 0.6)
        self._set_joint_angle(qpos, "hip_flexion_r", 0.3)
        self._set_joint_angle(qpos, "knee_angle_l", 0.5)
        self._set_joint_angle(qpos, "knee_angle_r", 0.3)
        self._set_joint_angle(qpos, "ankle_angle_l", 0.2)
        self._set_joint_angle(qpos, "ankle_angle_r", 0.2)


        qvel[0] = 0.8


        qpos_noise = self.np_random.normal(0, 0.005, size=qpos.shape)
        qpos_noise[3:7] = 0.0
        qpos[:] += qpos_noise


        qvel[:] += self.np_random.normal(0, 0.005, size=qvel.shape)

        self.robot.sync_sims(self.sim, self.sim_obsd)

        kwargs.pop('reset_qpos', None)
        kwargs.pop('reset_qvel', None)



        return super(WalkEnvV0, self).reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)

    def step(self, a):
        self.steps += 1
        self.phase += self.dt * 2 * np.pi * self.gait_freq
        return super().step(a)

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)


        torso_quat = sim.data.qpos[3:7]
        R = quat2mat(torso_quat)
        forward_vec = R @ np.array([1, 0, 0])

        obs_dict["slope_obs"] = np.array([forward_vec[2]])
        obs_dict["phase_clock"] = np.array([np.sin(self.phase), np.cos(self.phase)])
        return obs_dict

    def get_reward_dict(self, obs_dict):
        try:
            com_vel = self._get_com_velocity()
            x_vel = com_vel[0]
            z_vel = com_vel[2]


            forward_vel_reward = np.clip(x_vel, -0.2, 2.0)
            climb_up_reward = np.clip(z_vel, -0.2, 2.0)


            idle_penalty = 1.0 if x_vel < 0.2 else 0.0


            quat = self.sim.data.qpos[3:7]
            R = quat2mat(quat)
            torso_vec = R @ np.array([0, 0, 1])
            lean_forward = np.exp(-30.0 * (0.15 - torso_vec[0])**2)


            ankle_l = self._get_joint_angle("ankle_angle_l")
            ankle_r = self._get_joint_angle("ankle_angle_r")
            target_ankle = 0.25
            ankle_flex = np.exp(-10.0 * (target_ankle - ankle_l)**2) + \
                         np.exp(-10.0 * (target_ankle - ankle_r)**2)

            foot_align = 1.0 if (ankle_l > 0.05 and ankle_r > 0.05) else 0.0


            pelvis_z = self.sim.data.body_xpos[self.pelvis_id][2]
            f_l_z = self.sim.data.body_xpos[self.foot_l_id][2]
            f_r_z = self.sim.data.body_xpos[self.foot_r_id][2]
            avg_feet_z = (f_l_z + f_r_z) / 2.0
            stand_height = pelvis_z - avg_feet_z
            stand_tall = np.exp(-10.0 * (0.90 - stand_height)**2) if stand_height < 0.90 else 1.0


            illegal_contact = 0.0
            foot_contact = 0.0
            if self.sim.data.ncon > 0:
                for i in range(self.sim.data.ncon):
                    contact = self.sim.data.contact[i]
                    g1, g2 = contact.geom1, contact.geom2
                    is_foot = (g1 in [self.geom_foot_l, self.geom_foot_r] or
                               g2 in [self.geom_foot_l, self.geom_foot_r])
                    if is_foot:
                        foot_contact += 1.0
                    else:
                        illegal_contact -= 1.0


            clock = np.sin(self.phase)
            high_step = 0.0
            drag_penalty = 0.0
            knee_power = 0.0
            target_swing_h = 0.15
            knee_l = self._get_joint_angle("knee_angle_l")
            knee_r = self._get_joint_angle("knee_angle_r")

            if clock > 0:
                if f_l_z > target_swing_h: high_step += 1.0
                elif f_l_z < 0.05: drag_penalty -= 1.0
                if 0.1 < knee_r < 0.8: knee_power += 1.0
            else:
                if f_r_z > target_swing_h: high_step += 1.0
                elif f_r_z < 0.05: drag_penalty -= 1.0
                if 0.1 < knee_l < 0.8: knee_power += 1.0


            act = obs_dict.get("act", np.zeros(self.sim.model.nu))
            act_mag = np.mean(np.square(act))
            done = self._get_done()
            is_solved = (self.sim.data.qpos[2] > 2.0)

            rwd_dict = collections.OrderedDict((
                ("forward_vel", float(forward_vel_reward)),
                ("climb_up", float(climb_up_reward)),
                ("stand_tall", float(stand_tall)),
                ("foot_contact", float(foot_contact > 0)),
                ("illegal_contact", float(illegal_contact)),
                ("high_step", float(high_step)),
                ("lean_forward", float(lean_forward)),
                ("foot_align", float(foot_align)),
                ("ankle_flex", float(ankle_flex)),
                ("idle_penalty", float(idle_penalty * -1.0)),
                ("knee_power", float(knee_power)),
                ("drag_penalty", float(drag_penalty)),
                ("act_reg", float(-1.0 * act_mag)),
                ("alive", 1.0),
                ("done", float(1.0 if done else 0.0)),
                ("sparse", 0.0),
                ("solved", float(is_solved)),
            ))


            rwd_dict["dense"] = np.sum([wt * rwd_dict.get(key, 0.0) for key, wt in self.rwd_keys_wt.items()], axis=0)
            return rwd_dict

        except Exception as e:
            return collections.OrderedDict((("forward_vel", 0.0), ("dense", 0.0)))

    def _get_done(self):
        if self.steps < 20: return 0
        if self.steps >= self.max_episode_steps: return 1

        try:
            head_z = self.sim.data.body_xpos[self.head_id][2]
            pelvis_z = self.sim.data.body_xpos[self.pelvis_id][2]
            f_l_z = self.sim.data.body_xpos[self.foot_l_id][2]
            f_r_z = self.sim.data.body_xpos[self.foot_r_id][2]
            avg_feet_z = (f_l_z + f_r_z) / 2.0


            if (pelvis_z - avg_feet_z) < 0.55: return 1

            quat = self.sim.data.qpos[3:7]
            z_axis = (quat2mat(quat) @ np.array([0, 0, 1]))[2]
            if z_axis < 0.4: return 1

            if head_z < pelvis_z - 0.2: return 1
        except:
            pass
        return 0

    def _set_joint_angle(self, qpos_arr, name, val):
        try:
            jnt_id = self.sim.model.joint_name2id(name)
            qpos_addr = self.sim.model.jnt_qposadr[jnt_id]
            qpos_arr[qpos_addr] = val
        except:
            pass

    def _get_joint_angle(self, name):
        try:
            jnt_id = self.sim.model.joint_name2id(name)
            qpos_addr = self.sim.model.jnt_qposadr[jnt_id]
            return self.sim.data.qpos[qpos_addr]
        except:
            return 0.0

    def _get_com_velocity(self):
        return self.sim.data.qvel[:3]


class MSKBenchDoorOpenEnvV0(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "height", "muscle_length", "muscle_velocity", "muscle_force",
        "handle_rel_pos", "door_angle", "post_door_target_rel_pos"
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {

        "forward_vel": 200.0,
        "reach_handle": 400.0,
        "open_door": 500.0,
        "pass_through": 2000.0,


        "torso_upright": 50.0,
        "pelvis_height": 50.0,


        "split_legs_penalty": -400.0,
        "com_fall_penalty": -300.0,
        "door_leaning_penalty": -500.0,
        "stagnation_penalty": -150.0,
        "yaw_penalty": -100.0,

        "act_reg": -1.0,
        "alive": 10.0,
        "done": -100.0,
    }

    def __init__(self, model_path, **kwargs):
        self._in_setup = True
        self.steps = 0
        self.hand_id = -1
        self.head_id = -1
        self.pelvis_id = -1
        self.torso_id = -1
        self.door_hinge_id = -1
        self.handle_site_id = -1
        self.door_body_id = -1

        self.last_action = None
        self._force_done = False

        self.max_door_angle = 0.0
        self.door_opened_flag = False
        self.time_since_opened = 0
        self.max_x_reached = 0.0

        super().__init__(model_path=model_path, **kwargs)
        self._in_setup = False
        print("MSK-Bench task message.")

    def _setup(self, **kwargs):
        self._in_setup = True
        def _get_id_safe(names, type='body'):
            for name in names:
                try:
                    if type == 'body': return self.sim.model.body_name2id(name)
                    if type == 'site': return self.sim.model.site_name2id(name)
                    if type == 'joint': return self.sim.model.joint_name2id(name)
                except: pass
            return -1

        self.door_hinge_id = _get_id_safe(["door_hinge"], 'joint')
        self.handle_site_id = _get_id_safe(["handle_site", "handle"], 'site')
        self.door_body_id = _get_id_safe(["door"], 'body')

        self.hand_id = _get_id_safe(["thirdmc", "capitate", "scaphoid", "lunate", "hand", "radius"])
        self.head_id = _get_id_safe(["head", "cervical"])
        self.pelvis_id = _get_id_safe(["pelvis"])
        self.torso_id = _get_id_safe(["thorax", "torso", "spine"])

        self.foot_l_id = _get_id_safe(["talus_l", "calcn_l", "foot_l"])
        self.foot_r_id = _get_id_safe(["talus_r", "calcn_r", "foot_r"])

        self.post_door_target = np.array([2.5, 0.0, 0.95])

        super()._setup(obs_keys=self.DEFAULT_OBS_KEYS,
                       weighted_reward_keys=self.DEFAULT_RWD_KEYS_AND_WEIGHTS,
                       **kwargs)

    def reset(self, *, seed=None, options=None, **kwargs):
        self.steps = 0
        self.last_action = None
        self._force_done = False

        self.max_door_angle = 0.0
        self.door_opened_flag = False
        self.time_since_opened = 0
        self.max_x_reached = 0.0

        if seed is not None:
            self._np_random, seed = gym.utils.seeding.np_random(seed)

        qpos = self.sim.model.qpos0.copy()
        qvel = np.zeros(self.sim.model.nv)

        if hasattr(self, 'init_qpos'):
            n_init = min(len(self.init_qpos), len(qpos))
            qpos[:n_init] = self.init_qpos[:n_init]

        qpos[7:] += np.random.normal(0, 0.01, size=qpos[7:].shape)
        qvel[:] += np.random.normal(0, 0.01, size=qvel.shape)

        if self.door_hinge_id != -1:
            door_qpos_adr = self.sim.model.jnt_qposadr[self.door_hinge_id]
            door_qvel_adr = self.sim.model.jnt_dofadr[self.door_hinge_id]
            qpos[door_qpos_adr] = 0.0
            qvel[door_qvel_adr] = 0.0

        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()

        if hasattr(self, 'robot') and hasattr(self.robot, 'sync_sims'):
            self.robot.sync_sims(self.sim, self.sim_obsd)

        settled_qpos = self.sim.data.qpos.copy()
        settled_qvel = self.sim.data.qvel.copy()

        ret = super(WalkEnvV0, self).reset(reset_qpos=settled_qpos, reset_qvel=settled_qvel, **kwargs)
        return ret if isinstance(ret, tuple) else (ret, {})

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        obs_dict["handle_rel_pos"] = np.zeros(3, dtype=np.float32)
        obs_dict["door_angle"] = np.array([0.0], dtype=np.float32)
        obs_dict["post_door_target_rel_pos"] = np.zeros(3, dtype=np.float32)

        try:
            if self.hand_id != -1 and self.handle_site_id != -1:
                handle_pos = sim.data.site_xpos[self.handle_site_id]
                hand_pos = sim.data.xpos[self.hand_id]
                obs_dict["handle_rel_pos"] = (handle_pos - hand_pos).astype(np.float32)

            pelvis_pos = sim.data.qpos[:3]
            obs_dict["post_door_target_rel_pos"] = (self.post_door_target - pelvis_pos).astype(np.float32)

            if self.door_hinge_id != -1:
                door_qpos_adr = sim.model.jnt_qposadr[self.door_hinge_id]
                obs_dict["door_angle"] = np.array([sim.data.qpos[door_qpos_adr]], dtype=np.float32)
        except: pass
        return obs_dict

    def step(self, a):
        if getattr(self, '_in_setup', True):
            try: obs = self.get_obs()
            except: obs = np.zeros(self.observation_space.shape if hasattr(self, 'observation_space') else 1, dtype=np.float32)
            return obs, 0.0, False, False, {}

        self.steps += 1
        if np.isnan(a).any() or np.isinf(a).any():
            a = np.zeros_like(a)

        a_clipped = np.clip(a, -1.0, 1.0)
        self.last_action = a_clipped

        try:
            ret = super().step(a_clipped)
            if len(ret) == 5: obs_vec, reward, done, truncated, info = ret
            else: obs_vec, reward, done, info = ret; truncated = False
        except Exception as e:
            if hasattr(self, 'observation_space'): dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            else:
                try: dummy_obs = self.get_obs()
                except: dummy_obs = np.zeros(1, dtype=np.float32)
            safe_info = {f"rwd_{k}": 0.0 for k in self.DEFAULT_RWD_KEYS_AND_WEIGHTS.keys()}
            safe_info.update({"rwd_dense": -200.0, "error_flag": 1.0})
            return dummy_obs, -200.0, True, False, safe_info

        self.obs_dict = self.get_obs_dict(self.sim)
        self.obs_dict['current_action'] = a_clipped

        rwd_dict = self.get_reward_dict(self.obs_dict)
        for k, v in rwd_dict.items(): info['rwd_' + k] = v

        final_obs = self.get_obs()
        safe_reward = float(np.clip(rwd_dict['dense'], -1500.0, 5000.0))
        final_done = bool(self._get_done())

        return final_obs, safe_reward, final_done, bool(truncated), info

    def get_reward_dict(self, obs_dict):
        rwd_dict = collections.OrderedDict()
        for k in self.rwd_keys_wt.keys(): rwd_dict[k] = 0.0
        rwd_dict["dense"] = 0.0; rwd_dict["sparse"] = 0.0; rwd_dict["solved"] = 0.0

        try:
            door_angle = obs_dict.get("door_angle", [0.0])[0]
            handle_dist = np.linalg.norm(obs_dict.get("handle_rel_pos", [1.0, 1.0, 1.0]))

            pelvis_pos = self.sim.data.qpos[:3]
            pelvis_x, pelvis_z = pelvis_pos[0], pelvis_pos[2]

            if pelvis_x > self.max_x_reached:
                self.max_x_reached = pelvis_x

            v_x = self.sim.data.qvel[0]


            yaw_penalty = abs(self.sim.data.qvel[1]) * 0.5
            rwd_dict["yaw_penalty"] = float(yaw_penalty)


            r_pelvis_height = np.exp(-10.0 * (0.9 - pelvis_z)**2)


            split_penalty = 0.0
            if self.foot_l_id != -1 and self.foot_r_id != -1:
                f_l_x = self.sim.data.body_xpos[self.foot_l_id][0]
                f_r_x = self.sim.data.body_xpos[self.foot_r_id][0]
                leg_x_dist = abs(f_l_x - f_r_x)
                if leg_x_dist > 0.6:
                    split_penalty = min(1.0, (leg_x_dist - 0.6) * 2.0)


            hand_mult = np.clip((0.3 - handle_dist) / 0.15, 0.0, 1.0)

            self.max_door_angle = max(getattr(self, 'max_door_angle', 0.0), door_angle)
            if self.max_door_angle > 1.0:
                self.door_opened_flag = True

            r_reach = 0.0
            r_open = 0.0
            door_leaning_penalty = 0.0
            stagnation_penalty = 0.0


            r_reach = np.exp(-1.5 * handle_dist)

            if getattr(self, 'door_opened_flag', False):

                self.time_since_opened += 1
                if self.torso_id != -1 and self.door_body_id != -1:
                    torso_y = self.sim.data.body_xpos[self.torso_id][1]
                    door_y = self.sim.data.body_xpos[self.door_body_id][1]
                    if abs(torso_y - door_y) < 0.3: door_leaning_penalty = 1.0


                if self.time_since_opened > 50 and pelvis_x < 1.5 and abs(v_x) < 0.1:
                    stagnation_penalty = np.clip(self.time_since_opened / 200.0, 0.0, 1.0)
            else:

                r_open = np.clip(door_angle / 1.2, 0, 1.0) * hand_mult


                if v_x < 0.05:
                    stagnation_penalty = 1.0

            r_pass = 0.0
            if getattr(self, 'door_opened_flag', False):
                if pelvis_x > 1.0:
                    r_pass = np.clip((pelvis_x - 1.0) / 1.5, 0.0, 1.0)
                if pelvis_x > 2.0:
                    r_pass += 1.0


            torso_quat = self.sim.data.qpos[3:7]
            z_up = (quat2mat(torso_quat) @ np.array([0, 0, 1]))[2]
            r_torso_upright = np.exp(-5.0 * (1.0 - z_up)**2)

            com_fall_penalty = 0.0
            is_falling = False
            try:
                if self.foot_l_id != -1 and self.foot_r_id != -1:
                    f_l_x = self.sim.data.body_xpos[self.foot_l_id][0]
                    f_r_x = self.sim.data.body_xpos[self.foot_r_id][0]
                    front_foot_x = max(f_l_x, f_r_x)
                    if pelvis_x > (front_foot_x + 0.35):
                        is_falling = True
                        com_fall_penalty = np.clip((pelvis_x - front_foot_x - 0.35) / 0.2, 0.0, 1.0)
            except: pass

            act_mag = np.mean(np.square(obs_dict.get('act', np.zeros(1))))
            is_done = self._get_done()
            sparse_rwd = float(1.0 if (pelvis_x > 2.0) else 0.0)


            v_x_reward = 0.0 if is_falling else float(np.clip(v_x, -0.2, 1.0))

            rwd_dict["forward_vel"] = v_x_reward
            rwd_dict["pelvis_height"] = float(r_pelvis_height)
            rwd_dict["torso_upright"] = float(r_torso_upright)
            rwd_dict["split_legs_penalty"] = float(split_penalty)

            rwd_dict["com_fall_penalty"] = float(com_fall_penalty)
            rwd_dict["door_leaning_penalty"] = float(door_leaning_penalty)
            rwd_dict["stagnation_penalty"] = float(stagnation_penalty)
            rwd_dict["reach_handle"] = float(r_reach)
            rwd_dict["open_door"] = float(r_open)
            rwd_dict["pass_through"] = float(r_pass)
            rwd_dict["act_reg"] = float(act_mag)
            rwd_dict["alive"] = 1.0
            rwd_dict["done"] = float(1.0 if is_done else 0.0)
            rwd_dict["sparse"] = sparse_rwd
            rwd_dict["solved"] = sparse_rwd

            total_rwd = sum([self.rwd_keys_wt.get(k, 0.0) * rwd_dict[k] for k in self.rwd_keys_wt.keys()])
            rwd_dict["dense"] = float(np.clip(total_rwd, -1000.0, 3000.0))

        except Exception: pass
        return rwd_dict

    def _get_done(self):
        if getattr(self, '_in_setup', True): return False
        if getattr(self, '_force_done', False): return True

        try:
            if not np.isfinite(self.sim.data.qpos).all() or not np.isfinite(self.sim.data.qvel).all():
                return True

            pelvis_pos = self.sim.data.qpos[:3]


            if pelvis_pos[2] < 0.65:
                return True


            if self.head_id != -1:
                head_z = self.sim.data.body_xpos[self.head_id][2]
                if head_z < 1.1:
                    return True

            if self.door_hinge_id != -1:
                door_qvel_adr = self.sim.model.jnt_dofadr[self.door_hinge_id]
                door_vel = self.sim.data.qvel[door_qvel_adr]
                if abs(door_vel) > 20.0: return True

        except Exception as e:
            print(f"Low-level environment error intercepted: {e}")
            traceback.print_exc()
            return True

        return False


class MSKBenchReachEnvV0(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "height", "muscle_length",
        "muscle_velocity", "muscle_force",
        "hand_target_rel",
        "phase_clock"
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {

        "reach_dist": 50.0,
        "reach_vel": 20.0,
        "reach_bonus": 200.0,


        "torso_upright": 20.0,


        "act_reg": 0.5,
        "done": -100.0,
        "alive": 1.0,
        "solved": 0.0,
        "sparse": 0.0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, normalize_act=True, reset_type='none', render_mode=None, max_episode_steps=1000, **kwargs):
        self.steps = 0
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        if 'render_mode' in kwargs: del kwargs['render_mode']

        self.touch_count = 0
        self.knee_l_id = -1; self.knee_r_id = -1
        self.lunate_id = -1; self.target_id = -1

        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, normalize_act=normalize_act, **kwargs)
        self.reset_type = reset_type

    def _setup(self, obs_keys=DEFAULT_OBS_KEYS, weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS, **kwargs):
        self.steps = 0
        self.phase = 0.0
        self.gait_freq = 1.0

        if 'min_height' not in kwargs: kwargs['min_height'] = 0.8
        if 'target_x_vel' not in kwargs: kwargs['target_x_vel'] = 0.0

        super()._setup(obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, sites=[], **kwargs)
        self.init_qpos[:] = self.sim.model.qpos0.copy()


        def get_id(name):
            try: return self.sim.model.body_name2id(name)
            except: return -1

        self.knee_l_id = get_id("patella_l")
        self.knee_r_id = get_id("patella_r")
        self.lunate_id = get_id("lunate")
        self.target_id = get_id("target")
        self.head_id = get_id("head")

        self.target_range_x = [0.4, 0.8]
        self.target_range_y = [-0.5, 0.5]
        self.target_range_z = [1.2, 1.6]

    def _sample_target(self):
        pelvis_pos = self.sim.data.body("pelvis").xpos
        difficulty = min(self.touch_count, 5)


        min_z = 1.2 - (difficulty * 0.15)
        max_dist = 0.8 + (difficulty * 0.1)

        while True:
            new_x = self.np_random.uniform(0.3, max_dist)
            new_y = self.np_random.uniform(-0.6, 0.6)
            new_z = self.np_random.uniform(max(0.6, min_z), 1.7)

            dist = np.linalg.norm(np.array([new_x, new_y, new_z]) - pelvis_pos)
            if dist > 0.4: break

        if self.target_id != -1:
            self.sim.model.body_pos[self.target_id] = np.array([new_x, new_y, new_z])
            self.sim.forward()

    def reset(self, **kwargs):
        if 'reset_qpos' in kwargs: del kwargs['reset_qpos']
        if 'reset_qvel' in kwargs: del kwargs['reset_qvel']

        self.steps = 0
        self.phase = 0.0

        qpos = self.sim.model.qpos0.copy()
        qvel = np.zeros(self.sim.model.nv)


        qpos[0] = 0.0
        qpos[1] = 0.0
        qpos[2] = 0.93
        qpos[3:7] = euler2quat(np.array([0, 0, 0]))


        self._set_joint(qpos, "knee_angle_l", 0.4)
        self._set_joint(qpos, "knee_angle_r", 0.4)
        self._set_joint(qpos, "hip_flexion_l", 0.25)
        self._set_joint(qpos, "hip_flexion_r", 0.25)
        self._set_joint(qpos, "ankle_angle_l", 0.15)
        self._set_joint(qpos, "ankle_angle_r", 0.15)
        self._set_joint(qpos, "hip_adduction_l", 0.05)
        self._set_joint(qpos, "hip_adduction_r", -0.05)





        self._set_joint(qpos, "shoulder1_l2", 1.2)
        self._set_joint(qpos, "shoulder_elv_l", 0.2)
        self._set_joint(qpos, "shoulder_lot", 0.0)
        self._set_joint(qpos, "elbow_flexion_l", 1.6)
        self._set_joint(qpos, "pro_sup_l", 1.5)
        self._set_joint(qpos, "deviation_l", 0.2)


        self._set_joint(qpos, "shoulder1_r2", 1.2)
        self._set_joint(qpos, "shoulder_elv", 0.2)
        self._set_joint(qpos, "shoulder_rot", 0.0)
        self._set_joint(qpos, "elbow_flexion", 1.6)
        self._set_joint(qpos, "pro_sup", -1.5)
        self._set_joint(qpos, "deviation", -0.2)


        for finger in ["2", "3", "4", "5"]:
            self._set_joint(qpos, f"mcp{finger}_flexion_l", 0.5)
            self._set_joint(qpos, f"mcp{finger}_flexion", 0.5)


        self._set_joint(qpos, "lumbar_extension", -0.1)


        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()


        for _ in range(10):
            mujoco.mj_step(self.sim.model.ptr, self.sim.data.ptr)


        self.touch_count = 0
        self._sample_target()

        ret = BaseV0.reset(self, reset_qpos=self.sim.data.qpos.copy(), reset_qvel=self.sim.data.qvel.copy(), **kwargs)
        if isinstance(ret, tuple): return ret
        else: return ret, {}

    def step(self, a):
        self.steps += 1

        return super().step(a)

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)

        hand_target_rel = np.zeros(3)
        if self.lunate_id != -1 and self.target_id != -1:
            hand_pos = sim.data.body_xpos[self.lunate_id]
            target_pos = sim.data.body_xpos[self.target_id]
            hand_target_rel = target_pos - hand_pos

        obs_dict["hand_target_rel"] = hand_target_rel
        obs_dict["phase_clock"] = np.array([np.sin(self.phase), np.cos(self.phase)])


        if sim.model.na > 0:
            obs_dict['act'] = sim.data.act.copy()
        else:
            obs_dict['act'] = np.zeros(0)

        # Flatten
        for k, v in obs_dict.items():
            obs_dict[k] = np.array(v, dtype=np.float32).ravel()

        return obs_dict

    def get_reward_dict(self, obs_dict):

        reach_dist = 0.0; reach_vel = 0.0; reach_bonus = 0.0
        torso_upright = 0.0

        try:
            if self.lunate_id != -1 and self.target_id != -1:
                hand_pos = self.sim.data.body_xpos[self.lunate_id]
                target_pos = self.sim.data.body_xpos[self.target_id]
                dist = np.linalg.norm(hand_pos - target_pos)


                reach_dist = np.exp(-4.0 * dist)



                hand_vel = self.sim.data.cvel[self.lunate_id][:3]
                target_dir = (target_pos - hand_pos) / (dist + 1e-6)
                vel_towards_target = np.dot(hand_vel, target_dir)
                reach_vel = np.clip(vel_towards_target, 0, 2.0)



                if dist < 0.15:
                    reach_bonus = 1.0
                    self.touch_count += 1
                    self._sample_target()

            if self.head_id != -1:
                head_z = self.sim.data.body_xpos[self.head_id][2]
                torso_upright = np.exp(-10.0 * (1.7 - head_z)**2) if head_z < 1.7 else 1.0

        except: pass

        act = obs_dict.get('act', np.zeros(self.sim.model.nu))
        act_mag = np.mean(np.square(act))
        done = self._get_done()

        rwd_dict = collections.OrderedDict((
            ("reach_dist", float(reach_dist * 50.0)),
            ("reach_vel", float(reach_vel * 20.0)),
            ("reach_bonus", float(reach_bonus * 200.0)),
            ("torso_upright", float(torso_upright * 20.0)),
            ("act_reg", float(-1.0 * act_mag)),
            ("alive", 1.0),
            ("done", float(1.0 if done else 0.0)),
            ("sparse", float(self.touch_count)),
            ("solved", float(self.touch_count > 5)),
        ))


        dense_score = 0.0
        for key, wt in self.rwd_keys_wt.items():
            dense_score += wt * rwd_dict.get(key, 0.0)
        rwd_dict["dense"] = np.array(dense_score).reshape(())

        return rwd_dict

    def _get_done(self):
        if self.steps >= self.max_episode_steps: return 1


        if self.sim.data.qpos[2] < 0.6: return 1


        ground_threshold = 0.15
        if self.knee_l_id != -1 and self.knee_r_id != -1:
            k_l = self.sim.data.body_xpos[self.knee_l_id][2]
            k_r = self.sim.data.body_xpos[self.knee_r_id][2]
            if k_l < ground_threshold or k_r < ground_threshold:
                return 1

        return 0

    def _set_joint(self, qpos_arr, name, val):
        try:
            id = self.sim.model.joint_name2id(name)
            addr = self.sim.model.jnt_qposadr[id]
            qpos_arr[addr] = val
        except: pass


class MSKBenchWalkAndSitEnvV0(WalkEnvV0):
    """MSK-Bench task environment."""

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "height", "muscle_length",
        "muscle_velocity", "muscle_force",
        "feet_rel_positions",
        "chair_rel_pos",
        "phase_indicator",
        "waypoint_rel_pos"
    ]


    DEFAULT_RWD_KEYS_AND_WEIGHTS = {

        "walk_backwards": 20.0,
        "sit_contact": 50.0,
        "sit_precision": 100.0,
        "controlled_descent": 40.0,


        "sit_upright": 150.0,
        "sitting_pose": 100.0,
        "lumbar_hold": 80.0,
        "sit_still": 50.0,


        "head_height": 20.0,
        "shoulders_stable": 15.0,
        "feet_width": 20.0,
        "feet_ground": 30.0,


        "relaxed_posture": 200.0,
        "quick_drop": 50.0,
        "act_reg": 0.005,
        "alive": 5.0,
        "done": -100.0,
        "sparse": 0.0,
        "solved": 0.0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, normalize_act=True, reset_type='none', render_mode=None, **kwargs):
        self.render_mode = render_mode
        if 'render_mode' in kwargs:
            del kwargs['render_mode']


        self.steps = 0
        self.has_sat_down = False
        self.chair_pos = np.array([-0.5, 0.0, 0.45])

        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, normalize_act=normalize_act, **kwargs)
        self.reset_type = reset_type

    def _setup(self, obs_keys=DEFAULT_OBS_KEYS, weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS, chair_pos=None, **kwargs):
        self.max_episode_steps = kwargs.get('max_episode_steps', 1000)
        self.min_height = kwargs.get('min_height', 0.35)

        if chair_pos is not None:
            self.chair_pos = np.array(chair_pos)


        self.target_seat_center = self.chair_pos + np.array([-0.05, 0.0, 0.0])
        self.waypoint_pos = self.target_seat_center.copy()


        self.torso_id = self.sim.model.body_name2id("torso")
        self.pelvis_id = self.sim.model.body_name2id("pelvis")
        try:
            self.head_id = self.sim.model.body_name2id("head")
        except:
            self.head_id = None


        self.forbidden_contact_bodies = []
        for part in ["patella_l", "patella_r", "tibia_l", "tibia_r", "head"]:
            try:
                self.forbidden_contact_bodies.append(self.sim.model.body_name2id(part))
            except:
                pass

        super(WalkEnvV0, self)._setup(obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, sites=[], **kwargs)
        self.init_qpos[:] = self.sim.model.qpos0.copy()

        print(f"馃獞 Walk-and-Sit V32 Loaded | Chair at {self.chair_pos}")

    def reset(self, **kwargs):
        self.steps = 0
        self.has_sat_down = False

        qpos = self.sim.model.qpos0.copy()
        qvel = np.zeros(self.sim.model.nv)


        dist = self.np_random.uniform(1.0, 1.3)
        qpos[0] = self.chair_pos[0] + dist
        qpos[1] = self.chair_pos[1] + self.np_random.uniform(-0.02, 0.02)
        qpos[2] = 0.89


        yaw = self.np_random.uniform(-0.05, 0.05)
        qpos[3:7] = euler2quat(np.array([0, 0, yaw]))


        self._set_joint_angle(qpos, ["hip_adduction_l", "hip_adduction_r"], -0.1)
        self._set_joint_angle(qpos, ["hip_flexion_l", "hip_flexion_r"], 0.2)
        self._set_joint_angle(qpos, ["knee_angle_l", "knee_angle_r"], 0.2)
        self._set_joint_angle(qpos, ["lumbar_extension"], 0.1)


        qvel[0] = self.np_random.uniform(-0.2, -0.05)


        qpos[:] += self.np_random.normal(0, 0.005, size=qpos.shape)
        qvel[:] += self.np_random.normal(0, 0.005, size=qvel.shape)

        self.robot.sync_sims(self.sim, self.sim_obsd)


        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()
        for _ in range(10):
            mujoco.mj_step(self.sim.model.ptr, self.sim.data.ptr)


        kwargs.pop('reset_qpos', None)
        kwargs.pop('reset_qvel', None)
        ret = super(WalkEnvV0, self).reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        return (ret, {}) if not isinstance(ret, tuple) else ret

    def step(self, a):
        self.steps += 1
        obs, reward, terminated, truncated, info = super().step(a)


        rwd_dict = self.get_reward_dict(self.get_obs_dict(self.sim))
        reward = rwd_dict["dense"]
        info['rwd_dict'] = rwd_dict

        return obs, reward, terminated, truncated, info

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        pelvis_pos = sim.data.qpos[:3]


        obs_dict["chair_rel_pos"] = self.target_seat_center - pelvis_pos
        obs_dict["waypoint_rel_pos"] = self.target_seat_center - pelvis_pos
        obs_dict["phase_indicator"] = np.array([1.0 if self.has_sat_down else 0.0])

        return obs_dict

    def get_reward_dict(self, obs_dict):

        pelvis_pos = self.sim.data.qpos[:3]
        com_vel = self._get_com_velocity()
        to_chair_vec = self.target_seat_center - pelvis_pos
        dist_xy = np.linalg.norm(to_chair_vec[:2])

        target_seat_h = self.target_seat_center[2] + 0.05
        dist_z = np.abs(target_seat_h - pelvis_pos[2])


        if dist_z < 0.12 and dist_xy < 0.15:
            self.has_sat_down = True


        r_walk_backwards = 0.0
        r_controlled_descent = 0.0
        r_quick_drop = 0.0

        if not self.has_sat_down:
            torso_quat = self._get_torso_angle()
            R = quat2mat(torso_quat)
            target_dir = to_chair_vec[:2] / (dist_xy + 1e-6)
            torso_fwd = R @ np.array([1.0, 0.0, 0.0])


            face_dot = np.dot(torso_fwd[:2], target_dir)
            if face_dot < -0.8:
                vel_proj = np.dot(com_vel[:2], target_dir)
                r_walk_backwards = np.clip(vel_proj, -0.2, 1.0) * 5.0


            if dist_xy < 0.25:
                vel_z = self.sim.data.qvel[2]
                if vel_z < -1.5:
                    r_quick_drop = 1.0
                elif vel_z < -0.1:
                    r_controlled_descent = abs(vel_z) * 5.0


        r_sit_contact = np.exp(-dist_z**2 / 0.01)
        r_sit_precision = np.exp(-dist_xy**2 / 0.005) if (self.has_sat_down or dist_z < 0.15) else 0.0


        torso_quat = self._get_torso_angle()
        z_up = (quat2mat(torso_quat) @ np.array([0, 0, 1]))[2]

        lumbar = self._get_angle("lumbar_extension")[0]
        hips = self._get_angle(["hip_flexion_l", "hip_flexion_r"])
        knees = self._get_angle(["knee_angle_l", "knee_angle_r"])

        r_sit_upright = 0.0
        r_relaxed_posture = 0.0
        r_sitting_pose = 0.0
        r_lumbar_hold = 0.0
        r_feet_ground = 0.0

        if self.has_sat_down:

            if z_up < 0.8:
                r_relaxed_posture = 1.0
            else:
                r_sit_upright = np.exp(-30.0 * (1.0 - z_up)**2)


            hip_score = np.exp(-5.0 * (hips - 1.5)**2).mean()
            knee_score = np.exp(-5.0 * (knees - 1.6)**2).mean()
            r_sitting_pose = (hip_score + knee_score) / 2.0


            r_lumbar_hold = np.exp(-20.0 * (lumbar - 0.2)**2)
            vel_norm = np.linalg.norm(self.sim.data.qvel)
            r_sit_still = np.exp(-5.0 * vel_norm)


            feet_h = self._get_feet_heights()
            if np.all(feet_h < 0.05): r_feet_ground = 1.0
        else:
            r_sit_upright = np.exp(-10.0 * (1.0 - z_up)**2) * 0.2
            r_sit_still = 0.0


        R = quat2mat(self._get_torso_angle())
        shoulder_vec = R @ np.array([0.0, 1.0, 0.0])
        r_shoulders_stable = np.exp(-10.0 * shoulder_vec[2]**2)

        try:
            head_z = self.sim.data.body("head").xpos[2]
            r_head_height = np.exp(-10.0 * max(0, 1.5 - head_z)**2)
        except:
            r_head_height = 0.0


        act_mag = np.linalg.norm(obs_dict["act"]) / self.sim.model.na if self.sim.model.na != 0 else 0
        done = self._get_done()

        rwd_dict = collections.OrderedDict((
            ("walk_backwards", r_walk_backwards),
            ("sit_contact", r_sit_contact),
            ("sit_precision", r_sit_precision),
            ("controlled_descent", r_controlled_descent),
            ("sit_upright", r_sit_upright),
            ("sitting_pose", r_sitting_pose),
            ("sit_still", r_sit_still),
            ("lumbar_hold", r_lumbar_hold),
            ("head_height", r_head_height),
            ("shoulders_stable", r_shoulders_stable),
            ("feet_ground", r_feet_ground),
            ("quick_drop", r_quick_drop * 50.0),
            ("relaxed_posture", r_relaxed_posture * 100.0),
            ("act_reg", -1.0 * act_mag),
            ("alive", 0.5),
            ("done", 1.0 if done else 0.0),
            ("sparse", 0.0),
            ("solved", float(self.has_sat_down)),
        ))


        rwd_dict["dense"] = np.sum([wt * rwd_dict.get(key, 0.0) for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def _get_done(self):
        if self.steps >= self.max_episode_steps: return 1

        pelvis_pos = self.sim.data.qpos[:3]


        try:
            head_z = self.sim.data.body("head").xpos[2]
            if head_z < 0.8: return 1
        except: pass


        if not self.has_sat_down and pelvis_pos[2] < 0.5: return 1
        if self.has_sat_down and pelvis_pos[2] < 0.35: return 1


        for bid in self.forbidden_contact_bodies:
            if self.sim.data.body_xpos[bid][2] < 0.05: return 1


        z_up = (quat2mat(self.sim.data.qpos[3:7]) @ np.array([0, 0, 1]))[2]
        limit = 0.7 if self.has_sat_down else 0.6
        if z_up < limit: return 1

        return 0


    def _set_joint_angle(self, qpos_arr, names, val):
        if isinstance(names, str): names = [names]
        for name in names:
            try:
                jnt_id = self.sim.model.joint_name2id(name)
                qpos_arr[self.sim.model.jnt_qposadr[jnt_id]] = val
            except: pass

    def _get_angle(self, names):
        if isinstance(names, str): names = [names]
        vals = []
        for name in names:
            try:
                jnt_id = self.sim.model.joint_name2id(name)
                vals.append(self.sim.data.qpos[self.sim.model.jnt_qposadr[jnt_id]])
            except: vals.append(0.0)
        return np.array(vals)

    def _get_torso_angle(self):
        return self.sim.data.body_xquat[self.torso_id]

    def _get_com_velocity(self):
        return self.sim.data.qvel[:2].copy()

    def _get_feet_heights(self):
        try:
            f_l = self.sim.model.body_name2id("talus_l")
            f_r = self.sim.model.body_name2id("talus_r")
            return np.array([self.sim.data.body_xpos[f_l][2], self.sim.data.body_xpos[f_r][2]])
        except: return np.array([0.0, 0.0])


class MSKBenchChinUpEnvV0(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "height", "muscle_length", "muscle_velocity", "muscle_force",
        "chin_bar_dist",
        "phase_clock"
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "chin_up_progress": 50.0,
        "chin_over_bar": 200.0,
        "muscle_force_reward": 5.0,
        "explosive_pull": 40.0,
        "elbow_pull": 30.0,
        "pull_vel": 30.0,
        "swing_penalty": -50.0,
        "leg_wiggle": -20.0,
        "bad_posture": -10.0,
        "act_reg": -0.0001,
        "done": -100.0,
        "alive": 10.0,
        "solved": 0.0,
        "sparse": 0.0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, normalize_act=True, reset_type='none', render_mode=None, max_episode_steps=1000, **kwargs):
        self.steps = 0
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        if 'render_mode' in kwargs: del kwargs['render_mode']

        self.leg_qvel_idxs = []
        self.pull_muscle_ids = []

        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, normalize_act=normalize_act, **kwargs)
        self.reset_type = reset_type

    def _setup(self, obs_keys=DEFAULT_OBS_KEYS, weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS, **kwargs):
        self.bar_height = 2.3
        self.bar_pos = np.array([0.35, 0.0, 2.3])
        self.phase = 0.0

        if 'min_height' not in kwargs: kwargs['min_height'] = 0.5
        if 'target_x_vel' not in kwargs: kwargs['target_x_vel'] = 0.0

        super()._setup(obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, sites=[], **kwargs)
        self.init_qpos[:] = self.sim.model.qpos0.copy()

        leg_joint_names = [
            "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
            "knee_angle_l", "ankle_angle_l",
            "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
            "knee_angle_r", "ankle_angle_r"
        ]
        self.leg_qvel_idxs = []
        for name in leg_joint_names:
            try:
                j_id = self.sim.model.joint_name2id(name)
                dof_addr = self.sim.model.jnt_dofadr[j_id]
                self.leg_qvel_idxs.append(dof_addr)
            except: pass
        self.leg_qvel_idxs = np.array(self.leg_qvel_idxs, dtype=int)

        self.pull_muscle_ids = []
        target_muscles = ['lat', 'bic', 'bra', 'pect', 'trap', 'delt']
        for i in range(self.sim.model.nu):
            try:
                name = self.sim.model.actuator_id2name(i)
                if name and any(s in name.lower() for s in target_muscles):
                    self.pull_muscle_ids.append(i)
            except: pass
        self.pull_muscle_ids = np.array(self.pull_muscle_ids, dtype=int)

    def reset(self, **kwargs):
        self.steps = 0
        self.phase = 0.0

        qpos = self.sim.model.qpos0.copy()
        qvel = np.zeros(self.sim.model.nv)

        qpos[0] = 0.35
        qpos[1] = 0.0
        qpos[2] = 1.20

        self._set_joint(qpos, "shoulder_flex_l", 3.1)
        self._set_joint(qpos, "shoulder_flex_r", 3.1)
        self._set_joint(qpos, "elbow_flex_l", 0.05)
        self._set_joint(qpos, "elbow_flex_r", 0.05)

        self._set_joint(qpos, "hip_flexion_l", 0.0)
        self._set_joint(qpos, "hip_flexion_r", 0.0)
        self._set_joint(qpos, "knee_angle_l", 0.0)
        self._set_joint(qpos, "knee_angle_r", 0.0)

        noise = self.np_random.normal(0, 0.002, size=qpos.shape)
        qpos[7:] += noise[7:]

        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()

        for _ in range(100):
            mujoco.mj_step(self.sim.model.ptr, self.sim.data.ptr)

        if 'reset_qpos' in kwargs: del kwargs['reset_qpos']
        if 'reset_qvel' in kwargs: del kwargs['reset_qvel']



        ret = super(WalkEnvV0, self).reset(
            reset_qpos=self.sim.data.qpos.copy(),
            reset_qvel=self.sim.data.qvel.copy(),
            **kwargs
        )
        if isinstance(ret, tuple): return ret
        else: return ret, {}

    def step(self, a):
        self.steps += 1
        return super().step(a)

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        try:
            head_pos = sim.data.body_xpos[self.sim.model.body_name2id("head")]
            dist_z = self.bar_height - head_pos[2]
            obs_dict["chin_bar_dist"] = np.array([dist_z])
            obs_dict["phase_clock"] = np.zeros(2)
        except:
            obs_dict["chin_bar_dist"] = np.array([1.0])
            obs_dict["phase_clock"] = np.zeros(2)
        return obs_dict

    def get_reward_dict(self, obs_dict):
        try:
            head_z = self.sim.data.body_xpos[self.sim.model.body_name2id("head")][2]
        except: head_z = 1.5

        dist_to_bar = max(0, self.bar_height - head_z)
        chin_up_progress = np.exp(-5.0 * dist_to_bar)

        chin_over_bar = 0.0
        if head_z > self.bar_height:
            chin_over_bar = 1.0

        elbow_l = self._get_joint_val("elbow_flex_l")
        elbow_r = self._get_joint_val("elbow_flex_r")
        elbow_pull = (elbow_l + elbow_r)

        com_vel = self.sim.data.subtree_linvel[self.sim.model.body_name2id("pelvis")]
        pull_vel = max(0, com_vel[2])

        explosive_pull = 0.0
        try:
            root_acc_z = self.sim.data.qacc[2]
            if root_acc_z > 0: explosive_pull = root_acc_z
        except: pass

        horizontal_speed = np.sqrt(com_vel[0]**2 + com_vel[1]**2)
        swing_penalty = horizontal_speed

        leg_wiggle = 0.0
        if len(self.leg_qvel_idxs) > 0:
            leg_qvel = self.sim.data.qvel[self.leg_qvel_idxs]
            leg_wiggle = np.linalg.norm(leg_qvel)

        knee_l = self._get_joint_val("knee_angle_l")
        knee_r = self._get_joint_val("knee_angle_r")
        hip_l = self._get_joint_val("hip_flexion_l")

        bad_posture = 0.0
        if abs(knee_l) > 0.1 or abs(knee_r) > 0.1 or abs(hip_l) > 0.1:
            bad_posture = 1.0

        muscle_force_reward = 0.0
        if len(self.pull_muscle_ids) > 0:
            forces = self.sim.data.actuator_force[self.pull_muscle_ids]
            muscle_force_reward = np.mean(forces) / 500.0

        act = obs_dict.get('act', np.zeros(self.sim.model.nu))
        act_mag = np.mean(np.square(act))
        done = self._get_done()

        rwd_dict = collections.OrderedDict((
            ("chin_up_progress", chin_up_progress),
            ("chin_over_bar", chin_over_bar),
            ("elbow_pull", elbow_pull),
            ("pull_vel", pull_vel),

            ("muscle_force_reward", muscle_force_reward),
            ("explosive_pull", explosive_pull),

            ("swing_penalty", swing_penalty),
            ("leg_wiggle", leg_wiggle),
            ("bad_posture", bad_posture),

            ("act_reg", act_mag),
            ("alive", 1.0),
            ("done", 1.0 if done else 0.0),
            ("sparse", 0.0),
            ("solved", 0.0),
        ))
        rwd_dict["dense"] = np.sum([wt * rwd_dict.get(key, 0.0) for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def _get_done(self):
        if self.steps >= self.max_episode_steps: return 1
        if self.sim.data.qpos[2] < 0.5: return 1
        return 0

    def _set_joint(self, qpos_arr, name, val):
        try:
            id = self.sim.model.joint_name2id(name)
            addr = self.sim.model.jnt_qposadr[id]
            qpos_arr[addr] = val
        except: pass

    def _get_joint_val(self, name):
        try:
            id = self.sim.model.joint_name2id(name)
            addr = self.sim.model.jnt_qposadr[id]
            return self.sim.data.qpos[addr]
        except: return 0.0


class MSKBenchCatchEnvV0(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "muscle_length", "muscle_velocity",
        "cube_pos", "cube_vel", "hands_to_cube_dist"
    ]


    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach_cube": 100.0,
        "catch_bonus": 500.0,
        "drop_penalty": -200.0,
        "torso_upright": 50.0,
        "act_reg": 0.1,
        "alive": 10.0,
        "done": 0.0,
        "sparse": 1.0,
        "solved": 10.0,
    }

    def _find_body_id(self, potential_names):
        """MSK-Bench task environment."""
        for name in potential_names:
            try:
                return self.sim.model.body_name2id(name)
            except Exception:
                continue
        return None

    def _setup(self, **kwargs):

        self.hand_r_id = self._find_body_id(["lunate", "radius", "scaphoid"])
        self.hand_l_id = self._find_body_id(["lunate_l", "radius_l", "scaphoid_l"])


        self.cube_joint_id = self.sim.model.joint_name2id("cube_joint")
        self.cube_site_id = self.sim.model.site_name2id("cube_center")

        super()._setup(obs_keys=self.DEFAULT_OBS_KEYS,
                       weighted_reward_keys=self.DEFAULT_RWD_KEYS_AND_WEIGHTS,
                       **kwargs)

    def reset(self, **kwargs):
        """MSK-Bench task environment."""
        if 'seed' in kwargs: del kwargs['seed']
        if 'options' in kwargs: del kwargs['options']

        qpos = self.sim.model.qpos0.copy()
        qvel = np.zeros(self.sim.model.nv)



        try:
            for jnt in ["shoulder_elv", "shoulder_elv_l"]:
                jnt_id = self.sim.model.joint_name2id(jnt)
                adr = self.sim.model.jnt_qposadr[jnt_id]
                qpos[adr] = 0.5
            for jnt in ["elbow_flexion", "elbow_flexion_l"]:
                jnt_id = self.sim.model.joint_name2id(jnt)
                adr = self.sim.model.jnt_qposadr[jnt_id]
                qpos[adr] = 1.0
        except Exception:
            pass


        cube_qpos_adr = self.sim.model.jnt_qposadr[self.cube_joint_id]
        cube_dof_adr = self.sim.model.jnt_dofadr[self.cube_joint_id]


        qpos[cube_qpos_adr : cube_qpos_adr+3] = [1.0, self.np_random.uniform(-0.1, 0.1), 1.0]



        qvel[cube_dof_adr : cube_dof_adr+3] = [-1.5, self.np_random.uniform(-0.2, 0.2), 2.5]

        self.robot.sync_sims(self.sim, self.sim_obsd)
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()

        ret = super(WalkEnvV0, self).reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        if isinstance(ret, tuple): return ret
        else: return ret, {}

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)


        if getattr(self, 'hand_r_id', None) is None or getattr(self, 'cube_site_id', None) is None:
            obs_dict["cube_pos"] = np.zeros(3)
            obs_dict["cube_vel"] = np.zeros(3)
            obs_dict["hands_to_cube_dist"] = np.zeros(1)
            return obs_dict


        cube_pos = sim.data.site_xpos[self.cube_site_id]
        cube_dof_adr = sim.model.jnt_dofadr[self.cube_joint_id]
        cube_vel = sim.data.qvel[cube_dof_adr : cube_dof_adr+3]


        hand_r_pos = sim.data.xpos[self.hand_r_id]
        hand_l_pos = sim.data.xpos[self.hand_l_id]
        hands_center = (hand_r_pos + hand_l_pos) / 2.0

        obs_dict["cube_pos"] = cube_pos
        obs_dict["cube_vel"] = cube_vel
        obs_dict["hands_to_cube_dist"] = np.array([np.linalg.norm(cube_pos - hands_center)])

        return obs_dict

    def get_reward_dict(self, obs_dict):

        if getattr(self, 'cube_site_id', None) is None or getattr(self, 'hand_r_id', None) is None:
            dummy_dict = collections.OrderedDict((k, 0.0) for k in self.rwd_keys_wt.keys())
            dummy_dict["sparse"] = 0.0
            dummy_dict["solved"] = 0.0
            dummy_dict["done"] = 0.0
            dummy_dict["dense"] = 0.0
            self.rwd_dict = dummy_dict
            return dummy_dict


        cube_pos = self.sim.data.site_xpos[self.cube_site_id]


        hand_r_pos = self.sim.data.xpos[self.hand_r_id]
        hand_l_pos = self.sim.data.xpos[self.hand_l_id]
        hands_center = (hand_r_pos + hand_l_pos) / 2.0
        hands_dist = np.linalg.norm(cube_pos - hands_center)

        pelvis_z = self.sim.data.qpos[2]



        r_reach = np.exp(-3.0 * hands_dist)


        r_catch = 0.0
        r_drop = 0.0


        if hands_dist < 0.15 and cube_pos[2] > 0.5:
            r_catch = 1.0



        if cube_pos[2] < 0.1:
            r_drop = 1.0


        torso_quat = self.sim.data.body_xquat[self.sim.model.body_name2id("torso")]
        z_up = (quat2mat(torso_quat) @ np.array([0, 0, 1]))[2]
        r_upright = np.exp(-5.0 * (1.0 - z_up)**2)


        rwd_dict = collections.OrderedDict()
        rwd_dict["reach_cube"] = r_reach
        rwd_dict["catch_bonus"] = r_catch
        rwd_dict["drop_penalty"] = r_drop
        rwd_dict["torso_upright"] = r_upright

        act = obs_dict.get("act", np.zeros(self.sim.model.nu))
        rwd_dict["act_reg"] = -1.0 * np.mean(np.square(act))
        rwd_dict["alive"] = 1.0

        is_done = self._get_done()

        is_solved = 1.0 if (r_catch > 0.5 and not is_done) else 0.0

        rwd_dict["done"] = float(is_done)
        rwd_dict["sparse"] = is_solved
        rwd_dict["solved"] = is_solved


        dense_reward = 0.0
        for key, weight in self.rwd_keys_wt.items():
            if key != "dense" and key in rwd_dict:
                dense_reward += weight * rwd_dict[key]

        rwd_dict["dense"] = dense_reward
        self.rwd_dict = rwd_dict

        return rwd_dict

    def _get_done(self):
        pelvis_z = self.sim.data.qpos[2]
        cube_z = self.sim.data.site_xpos[self.cube_site_id][2]


        if pelvis_z < 0.5:
            return 1


        if cube_z < 0.05:
            return 1

        return super()._get_done()


class MSKBenchPoleWalkEnvV0(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy", "qvel", "com_vel", "torso_angle",
        "feet_heights", "height", "muscle_length",
        "muscle_velocity", "muscle_force",
        "vision_obs",
        "phase_clock"
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "stand_reward": 400.0,
        "upright_reward": 200.0,
        "gait_reward": 400.0,
        "forward_vel": 600.0,
        "heading_align": 400.0,
        "torso_upright": 300.0,
        "backward_lean_pen": -400.0,
        "static_penalty": -600.0,
        "collision_hit": -1500.0,
        "gap_navigation": 800.0,
        "act_reg": -1.0,
        "alive": 20.0,
        "done": -100.0,
    }

    def __init__(self, model_path, **kwargs):
        self.observation_space = gym.spaces.Box(

            low=-np.inf, high=np.inf, shape=(1941,), dtype=np.float32
        )
        self.steps = 0
        self.pole_geom_ids = []
        self.pole_positions = []
        self.phase = 0.0
        self.gait_freq = 1.1
        self._in_setup = True

        self.torso_id = -1; self.pelvis_id = -1
        self.talus_l_id = -1; self.talus_r_id = -1

        self.rwd_dict = OrderedDict()
        self.rwd_keys_wt = self.DEFAULT_RWD_KEYS_AND_WEIGHTS
        self._init_rwd_dict()

        if 'min_height' not in kwargs:
            kwargs['min_height'] = 0.6

        super().__init__(model_path=model_path, **kwargs)

    def _init_rwd_dict(self):
        for k in self.rwd_keys_wt.keys(): self.rwd_dict[k] = 0.0
        self.rwd_dict.update({'sparse': 0.0, 'solved': 0.0, 'dense': 0.0, 'done': 0.0})

    def _setup(self, **kwargs):
        self._in_setup = True
        try:
            self.torso_id = self.sim.model.body_name2id("torso")
            self.pelvis_id = self.sim.model.body_name2id("pelvis")
            self.talus_l_id = self.sim.model.body_name2id("talus_l")
            self.talus_r_id = self.sim.model.body_name2id("talus_r")
        except: pass

        self.pole_geom_ids = []
        self.pole_positions = []
        for i in range(self.sim.model.ngeom):
            body_id = self.sim.model.geom_bodyid[i]
            body_name = self.sim.model.id2name(body_id, 'body') or ''
            if body_name.startswith('p'):
                self.pole_geom_ids.append(i)
                self.pole_positions.append(self.sim.model.body_pos[body_id][:2].copy())

        self.pole_positions = np.array(self.pole_positions)


        for i in range(self.sim.model.nu):
            name = self.sim.model.id2name(i, 'actuator')
            if name and any(s in name.lower() for s in ['oblique', 'lumbar', 'glute_med']):
                self.sim.model.actuator_gainprm[i, 2] *= 3.0

        super()._setup(
            obs_keys=self.DEFAULT_OBS_KEYS,
            weighted_reward_keys=self.DEFAULT_RWD_KEYS_AND_WEIGHTS,
            sites=[],
            **kwargs
        )
        self._in_setup = False

    def reset(self, *, seed=None, options=None, **kwargs):
        """
        Compatible reset for SB3 2.x / Gymnasium + MSK-Bench.

        SB3 2.x SubprocVecEnv expects:
            obs, info = env.reset(seed=..., options=...)

        This function always returns:
            obs, info
        """

        # 1. Reset custom episode variables
        self.steps = 0
        self.phase = 0.0

        # 2. Call parent reset safely
        reset_kwargs = dict(kwargs)

        if seed is not None:
            reset_kwargs["seed"] = seed

        if options is not None:
            reset_kwargs["options"] = options

        parent_info = {}

        try:
            parent_ret = super().reset(**reset_kwargs)
        except TypeError:
            # MSK-Bench / old Gym may not accept seed/options
            reset_kwargs.pop("seed", None)
            reset_kwargs.pop("options", None)

            if seed is not None:
                try:
                    self.seed(seed)
                except Exception:
                    pass

            parent_ret = super().reset(**reset_kwargs)

        # 3. Extract info if parent reset already returned Gymnasium-style tuple
        if isinstance(parent_ret, tuple):
            if len(parent_ret) == 2 and isinstance(parent_ret[1], dict):
                parent_info = parent_ret[1]

        # 4. Manually set initial qpos/qvel
        qpos = self.sim.model.qpos0.copy()
        qvel = np.zeros(self.sim.model.nv)


        qpos[0:2] = 0.0


        qpos[2] = 1.05

        def set_joint_angle(joint_name, angle):
            try:
                joint_id = self.sim.model.joint_name2id(joint_name)
                qpos_adr = self.sim.model.jnt_qposadr[joint_id]
                qpos[qpos_adr] = angle
            except Exception:
                pass


        set_joint_angle("hip_flexion_l", 0.1)
        set_joint_angle("knee_angle_l", 0.05)
        set_joint_angle("hip_flexion_r", 0.1)
        set_joint_angle("knee_angle_r", 0.05)
        set_joint_angle("hip_adduction_l", 0.0)
        set_joint_angle("hip_adduction_r", 0.0)


        try:
            qpos[7:] += self.np_random.normal(0, 0.01, size=qpos[7:].shape)
            qvel[:] += self.np_random.normal(0, 0.01, size=qvel.shape)
        except Exception:
            qpos[7:] += np.random.normal(0, 0.01, size=qpos[7:].shape)
            qvel[:] += np.random.normal(0, 0.01, size=qvel.shape)

        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.data.ctrl[:] = 0.0
        self.sim.forward()

        if hasattr(self, "robot") and hasattr(self.robot, "sync_sims"):
            try:
                self.robot.sync_sims(self.sim, self.sim_obsd)
            except Exception:
                pass


        try:
            import mujoco
            for _ in range(30):
                mujoco.mj_step(self.sim.model.ptr, self.sim.data.ptr)
        except Exception:
            pass


        try:
            f_l_z = self.sim.data.body_xpos[self.talus_l_id][2]
            f_r_z = self.sim.data.body_xpos[self.talus_r_id][2]
            self.base_foot_z = min(f_l_z, f_r_z)
        except Exception:
            self.base_foot_z = 0.0

        # 7. Get final observation
        obs = self.get_obs()


        try:
            if isinstance(obs, dict):
                obs, _ = self.obsdict2obsvec(obs, self.obs_keys)

            obs = self._align_obs_dim(obs)
        except Exception:
            obs = np.zeros((1941,), dtype=np.float32)


        info = dict(parent_info)
        info["reset_success"] = True

        return obs, info

    def step(self, a):
        if getattr(self, '_in_setup', True):
            return np.zeros((1941,), dtype=np.float32), 0.0, False, False, {}

        self.steps += 1
        self.phase += self.dt * 2 * np.pi * self.gait_freq

        try:
            obs, reward, done, truncated, info = super().step(a)
            if isinstance(obs, dict):
                obs, _ = self.obsdict2obsvec(obs, self.obs_keys)

            obs = self._align_obs_dim(obs)
            self.rwd_dict = self.get_reward_dict(self.obs_dict)

            real_done = self._get_done()
            for k, v in self.rwd_dict.items(): info['rwd_' + k] = v

            return obs, float(self.rwd_dict['dense']), real_done, truncated, info
        except:
            return np.zeros((1941,), dtype=np.float32), -100.0, True, False, {"status": "crash"}

    def get_reward_dict(self, obs_dict):
        rwd_dict = OrderedDict((k, 0.0) for k in self.rwd_keys_wt.keys())
        rwd_dict.update({'sparse': 0.0, 'solved': 0.0, 'done': 0.0})
        try:
            height = self.sim.data.qpos[2]
            stand_r = np.exp(-15.0 * (1.05 - height)**2)

            fe_jid = self.sim.model.joint_name2id("flex_extension")
            fe_angle = self.sim.data.qpos[self.sim.model.jnt_qposadr[fe_jid]]
            upright_r = np.exp(-10.0 * fe_angle**2)

            rot_mat = quat2mat(self.sim.data.body_xquat[self.torso_id])

            v_forward_global = self.sim.data.qvel[0]
            v_side_global = self.sim.data.qvel[1]

            heading_x = (rot_mat @ np.array([1, 0, 0]))[0]
            heading_align = np.exp(-5.0 * (1.0 - heading_x)**2)

            torso_z_world = rot_mat @ np.array([0, 0, 1])
            backward_lean_pen = 1.0 if torso_z_world[0] < -0.05 else 0.0

            v_forward_clip = np.clip(v_forward_global, -0.2, 2.0)
            static_p = 1.0 if v_forward_clip < 0.15 else 0.0

            pelvis_pos = self.sim.data.body_xpos[self.pelvis_id]
            f_l_x = self.sim.data.body_xpos[self.talus_l_id][0]
            f_r_x = self.sim.data.body_xpos[self.talus_r_id][0]

            sin_p = np.sin(self.phase)
            target_l_x = 0.25 * sin_p
            target_r_x = -0.25 * sin_p

            gait_r = np.exp(-10.0 * ((f_l_x - pelvis_pos[0]) - target_l_x)**2) + \
                     np.exp(-10.0 * ((f_r_x - pelvis_pos[0]) - target_r_x)**2)


            gap_nav_score = 0.0
            radar_obs = obs_dict.get("vision_obs", np.zeros(7))

            if np.max(radar_obs) > 0.3:

                safest_idx = np.argmin(radar_obs)

                if safest_idx > 3:

                    gap_nav_score = np.clip(v_side_global * 2.0, 0.0, 1.0)
                elif safest_idx < 3:

                    gap_nav_score = np.clip(-v_side_global * 2.0, 0.0, 1.0)
                else:
                    gap_nav_score = np.clip(v_forward_global * 1.5, 0.0, 1.0)

            rwd_dict.update({
                "stand_reward": float(stand_r),
                "upright_reward": float(upright_r),
                "gait_reward": float(gait_r),
                "forward_vel": float(v_forward_clip),
                "heading_align": float(heading_align),
                "torso_upright": float(np.exp(-10.0 * (1.0 - rot_mat[2, 2])**2)),
                "backward_lean_pen": float(-backward_lean_pen),
                "static_penalty": float(static_p),
                "gap_navigation": float(gap_nav_score),
                "act_reg": float(-np.mean(np.square(self.sim.data.ctrl))),
                "alive": 2.0,
                "done": float(1.0 if self._get_done() else 0.0),
            })
            rwd_dict['dense'] = np.sum([self.rwd_keys_wt.get(k, 0.0) * rwd_dict[k] for k in self.rwd_keys_wt.keys()])
        except:
            rwd_dict['dense'] = 0.0
        return rwd_dict

    def _get_done(self):
        if getattr(self, '_in_setup', True): return False

        p_z = self.sim.data.qpos[2]
        if p_z < 0.70 or p_z > 2.0: return True

        try:
            fe_jid = self.sim.model.joint_name2id("flex_extension")
            if abs(self.sim.data.qpos[self.sim.model.jnt_qposadr[fe_jid]]) > 0.7:
                return True
        except: pass

        return self._check_collision()

    def _align_obs_dim(self, obs):
        obs_flat = obs.astype(np.float32).flatten()
        if obs_flat.shape[0] == 1941: return obs_flat
        aligned = np.zeros(1941, dtype=np.float32)
        aligned[:min(1941, obs_flat.shape[0])] = obs_flat[:min(1941, obs_flat.shape[0])]
        return aligned

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)



        radar_readings = np.zeros(7, dtype=np.float32)
        try:
            if self.torso_id != -1 and len(self.pole_positions) > 0:
                torso_pos = sim.data.body_xpos[self.torso_id][:2]
                rot_mat = quat2mat(sim.data.body_xquat[self.torso_id])
                forward_dir = (rot_mat @ np.array([1, 0, 0]))[:2]
                forward_angle = np.arctan2(forward_dir[1], forward_dir[0])


                radar_angles = np.linspace(-np.pi/2, np.pi/2, 7)
                max_radar_dist = 2.5

                rel_vecs = self.pole_positions - torso_pos
                dists = np.linalg.norm(rel_vecs, axis=1)
                angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0]) - forward_angle
                angles = (angles + np.pi) % (2 * np.pi) - np.pi

                for i, ray_angle in enumerate(radar_angles):

                    angle_diffs = np.abs(angles - ray_angle)
                    valid_poles = (angle_diffs < 0.25) & (dists < max_radar_dist)
                    if np.any(valid_poles):
                        closest_dist = np.min(dists[valid_poles])
                        radar_readings[i] = 1.0 - (closest_dist / max_radar_dist)
        except: pass

        obs_dict["vision_obs"] = radar_readings
        obs_dict["phase_clock"] = np.array([np.sin(self.phase), np.cos(self.phase)], dtype=np.float32)
        return obs_dict

    def _check_collision(self):
        if not self.pole_geom_ids: return False
        for i in range(self.sim.data.ncon):
            con = self.sim.data.contact[i]
            if (con.geom1 in self.pole_geom_ids) or (con.geom2 in self.pole_geom_ids):
                g1_body = self.sim.model.geom_bodyid[con.geom1]
                g2_body = self.sim.model.geom_bodyid[con.geom2]
                names = (str(self.sim.model.id2name(g1_body, 'body')) +
                         str(self.sim.model.id2name(g2_body, 'body'))).lower()
                deadly_parts = ['torso', 'pelvis', 'head', 'skull', 'cervical', 'clavicle', 'humerus']
                if any(n in names for n in deadly_parts):
                    return True
        return False

