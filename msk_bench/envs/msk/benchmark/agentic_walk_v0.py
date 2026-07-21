"""Agentic reward-tuning walk task.

The environment keeps the original MSK-Bench walking dynamics and lets one
coordinator process update reward weights from recent rollout statistics.
API credentials are read only from environment variables and are never stored in
repository files.
"""

from __future__ import annotations

import collections
import json
import os
import re
import time
from pathlib import Path
from typing import Mapping

import numpy as np

from msk_bench.envs.msk.benchmark.walk_v0 import WalkEnvV0
from msk_bench.utils import gym
from msk_bench.utils.quat_math import quat2mat


DEFAULT_AGENTIC_REWARD_WEIGHTS = {
    "vel_dir_bonus": 5.0,
    "tracking_vel": 10.0,
    "tracking_joint": 3.0,
    "upright_reward": 6.0,
    "alive": 1.0,
    "act_reg": 0.0,
    "done": -100.0,
    "face_front": 8.0,
    "anti_hunch": 8.0,
}


class ReferenceMotion:
    def __init__(self, motion_file: str | os.PathLike[str], dt: float = 1.0 / 30.0):
        self.dt = float(dt)
        self.qpos_data = np.load(motion_file)
        self.n_frames = int(self.qpos_data.shape[0])
        self.duration = self.n_frames * self.dt
        root_pos = self.qpos_data[:, 0:3]
        self.root_lin_vel = np.gradient(root_pos, axis=0) / self.dt
        joint_pos = self.qpos_data[:, 7:]
        self.joint_vel = np.gradient(joint_pos, axis=0) / self.dt

    def get_reference_state(self, sim_time: float):
        phase_time = sim_time % self.duration
        frame_idx = min(int(phase_time / self.dt), self.n_frames - 1)
        return self.qpos_data[frame_idx], self.root_lin_vel[frame_idx], self.joint_vel[frame_idx]


class AgenticRewardController:
    def __init__(
        self,
        weight_file: str | os.PathLike[str],
        best_weight_file: str | os.PathLike[str],
        reward_weights: Mapping[str, float],
        update_steps: int,
        enabled: bool = True,
    ):
        self.weight_file = Path(weight_file)
        self.best_weight_file = Path(best_weight_file)
        self.update_steps = int(update_steps)
        self.enabled = bool(enabled)
        self.best_score = -float("inf")
        self.stats: collections.defaultdict[str, list[float]] = collections.defaultdict(list)
        self.weight_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.weight_file.exists():
            self.write_weights(reward_weights)

    def read_weights(self, current_weights: Mapping[str, float]) -> dict[str, float]:
        if not self.weight_file.exists():
            return dict(current_weights)
        try:
            loaded = json.loads(self.weight_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return dict(current_weights)
        weights = dict(current_weights)
        for key, value in loaded.items():
            if key in weights:
                weights[key] = float(value)
        return weights

    def write_weights(self, weights: Mapping[str, float]) -> None:
        self.weight_file.write_text(json.dumps(dict(weights), indent=2, sort_keys=True), encoding="utf-8")

    def record(self, speed: float, pitch: float) -> None:
        self.stats["speed"].append(float(speed))
        self.stats["pitch"].append(float(pitch))

    def maybe_update(self, step_count: int, reward_weights: Mapping[str, float]) -> dict[str, float]:
        if not self.enabled or self.update_steps <= 0 or step_count % self.update_steps != 0:
            return dict(reward_weights)
        if not self.stats["speed"]:
            return dict(reward_weights)

        lock_file = self.weight_file.with_suffix(self.weight_file.suffix + ".lock")
        if lock_file.exists() and time.time() - lock_file.stat().st_mtime < 60:
            self.stats.clear()
            return dict(reward_weights)

        try:
            lock_file.write_text(str(time.time()), encoding="utf-8")
            updated = self._update_from_llm(reward_weights)
            self.write_weights(updated)
            return updated
        finally:
            self.stats.clear()
            try:
                lock_file.unlink()
            except FileNotFoundError:
                pass

    def _update_from_llm(self, reward_weights: Mapping[str, float]) -> dict[str, float]:
        api_key = os.getenv("MSK_BENCH_AGENTIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return dict(reward_weights)

        avg_speed = float(np.mean(self.stats["speed"]))
        avg_pitch = float(np.mean(self.stats["pitch"]))
        current_score = min(avg_speed, 1.0) - (0.0 if avg_pitch > 0 else abs(avg_pitch) * 2.0)
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_weight_file.write_text(json.dumps(dict(reward_weights), indent=2, sort_keys=True), encoding="utf-8")

        prompt = (
            "You are a reinforcement learning reward tuning assistant. "
            f"Current reward weights: {json.dumps(dict(reward_weights), sort_keys=True)}\n"
            f"Average forward speed: {avg_speed:.3f} m/s; target is 1.0 m/s.\n"
            f"Average torso pitch projection: {avg_pitch:.3f}; upright is positive, hunching is negative.\n"
            "Return only a JSON object with the same reward keys. Keep each update conservative."
        )

        try:
            from openai import OpenAI

            client_kwargs = {"api_key": api_key}
            base_url = os.getenv("MSK_BENCH_AGENTIC_BASE_URL") or os.getenv("OPENAI_BASE_URL")
            if base_url:
                client_kwargs["base_url"] = base_url
            client = OpenAI(**client_kwargs)
            response = client.chat.completions.create(
                model=os.getenv("MSK_BENCH_AGENTIC_MODEL", "gpt-5.5"),
                messages=[{"role": "user", "content": prompt}],
                temperature=float(os.getenv("MSK_BENCH_AGENTIC_TEMPERATURE", "0.2")),
                timeout=float(os.getenv("MSK_BENCH_AGENTIC_TIMEOUT", "25")),
            )
            content = response.choices[0].message.content or "{}"
            match = re.search(r"\{.*\}", content, re.DOTALL)
            new_weights = json.loads(match.group(0) if match else content)
        except Exception:
            return dict(reward_weights)

        bounded = dict(reward_weights)
        for key, value in new_weights.items():
            if key not in bounded:
                continue
            old_value = float(reward_weights[key])
            proposed = float(value)
            if old_value == 0.0:
                bounded[key] = proposed
            else:
                max_delta = abs(old_value) * 0.2
                bounded[key] = float(np.clip(proposed, old_value - max_delta, old_value + max_delta))
        return bounded


class MSKBenchAgenticWalkEnvV0(WalkEnvV0):
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
        "qpos_error",
        "act",
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = DEFAULT_AGENTIC_REWARD_WEIGHTS

    def __init__(
        self,
        model_path,
        obsd_model_path=None,
        seed=None,
        reference_motion_path=None,
        agentic_weight_dir=None,
        agentic_update_steps=20000,
        enable_agentic=True,
        **kwargs,
    ):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(
            model_path=model_path,
            obsd_model_path=obsd_model_path,
            seed=seed,
            env_credits=self.MYO_CREDIT,
        )
        curr_dir = Path(__file__).resolve().parent
        motion_path = Path(reference_motion_path) if reference_motion_path else curr_dir / "clean_walk.npy"
        self.ref_motion = ReferenceMotion(motion_path) if motion_path.exists() else None
        weight_dir = Path(agentic_weight_dir or os.getenv("MSK_BENCH_AGENTIC_STATE_DIR", curr_dir / "agentic_state"))
        self.agentic = AgenticRewardController(
            weight_file=weight_dir / "shared_reward_weights.json",
            best_weight_file=weight_dir / "best_reward_weights.json",
            reward_weights=self.DEFAULT_RWD_KEYS_AND_WEIGHTS,
            update_steps=agentic_update_steps,
            enabled=enable_agentic,
        )
        self.step_count = 0
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
        self.torso_id = self.sim.model.body_name2id("torso")
        self.pelvis_id = self.sim.model.body_name2id("pelvis")
        super()._setup(obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, **kwargs)
        self.rwd_keys_wt = dict(self.agentic.read_weights(self.rwd_keys_wt))

    def reset(self, **kwargs):
        self.rwd_keys_wt = dict(self.agentic.read_weights(self.rwd_keys_wt))
        if self.ref_motion is not None:
            self.start_time_offset = self.np_random.uniform(0, self.ref_motion.duration)
            ref_qpos, ref_root_vel, ref_joint_vel = self.ref_motion.get_reference_state(self.start_time_offset)
            init_qpos = ref_qpos.copy()
            init_qpos[0] = 0.0
            init_qvel = np.zeros_like(self.sim.data.qvel)
            init_qvel[0:3] = ref_root_vel
            num_joints = min(len(ref_joint_vel), len(init_qvel) - 6)
            init_qvel[6 : 6 + num_joints] = ref_joint_vel[:num_joints]
        else:
            self.start_time_offset = 0.0
            init_qpos = self.sim.model.key_qpos[0].copy()
            init_qvel = self.sim.model.key_qvel[0].copy()
        self.robot.sync_sims(self.sim, self.sim_obsd)
        return super().reset(reset_qpos=init_qpos, reset_qvel=init_qvel, **kwargs)

    def get_obs_dict(self, sim):
        obs_dict = {
            "t": np.array([sim.data.time]),
            "time": np.array([sim.data.time]),
            "qpos": sim.data.qpos.copy(),
            "qpos_without_xy": sim.data.qpos[2:].copy(),
            "qvel": sim.data.qvel[:].copy(),
            "com_vel": self._get_com_velocity().copy(),
            "torso_angle": self._get_torso_angle().copy(),
            "feet_heights": self._get_feet_heights().copy(),
            "height": np.array([self._get_height()]).copy(),
            "feet_rel_positions": self._get_feet_relative_position().copy(),
        }
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

        duration = self.ref_motion.duration if self.ref_motion else 1.0
        obs_dict["phase_var"] = np.array([(sim.data.time / duration) % 1]).copy()
        if self.ref_motion is not None:
            ref_qpos, _, _ = self.ref_motion.get_reference_state(sim.data.time + getattr(self, "start_time_offset", 0.0))
            current_joints = sim.data.qpos[7:]
            target_joints = ref_qpos[7:]
            min_len = min(len(current_joints), len(target_joints))
            obs_dict["qpos_error"] = target_joints[:min_len] - current_joints[:min_len]
        else:
            obs_dict["qpos_error"] = np.zeros(sim.model.nq - 7)
        return obs_dict

    def get_reward_dict(self, obs_dict):
        current_time = self.sim.data.time + getattr(self, "start_time_offset", 0.0)
        if self.ref_motion:
            ref_qpos, ref_root_vel, _ = self.ref_motion.get_reference_state(current_time)
            target_root_vel_xy = ref_root_vel[:2]
        else:
            ref_qpos = self.sim.data.qpos
            target_root_vel_xy = np.array([1.0, 0.0])

        curr_root_vel = self._get_com_velocity()
        torso_quat = self._get_torso_angle()
        mat = quat2mat(torso_quat)
        head_dir = mat @ np.array([0.0, 0.0, 1.0])
        face_dir = mat @ np.array([1.0, 0.0, 0.0])

        self.step_count += 1
        self.agentic.record(curr_root_vel[0], face_dir[2])
        self.rwd_keys_wt = self.agentic.maybe_update(self.step_count, self.rwd_keys_wt)

        facing_front_score = face_dir[0]
        r_face_front = np.exp(-5.0 * (1.0 - facing_front_score))
        if facing_front_score < 0.7:
            r_face_front *= 0.1

        z_project = head_dir[2]
        r_upright = np.exp(-10.0 * (1.0 - z_project))
        pitch_val = face_dir[2]
        r_pitch = 1.0 if pitch_val >= -0.1 else np.exp(-20.0 * (pitch_val + 0.1) ** 2)
        if self.sim.data.body_xpos[self.torso_id][2] < 1.15:
            r_upright = 0.0

        target_speed = np.linalg.norm(target_root_vel_xy)
        dot_prod = 0.0
        vel_dir_reward = 0.0
        if target_speed > 0.01:
            dot_prod = float(np.dot(curr_root_vel, target_root_vel_xy))
            vel_dir_reward = 1.0 if dot_prod > 0 else -1.0

        r_tracking_vel = np.exp(-2.0 * np.sum((curr_root_vel - target_root_vel_xy) ** 2))
        if dot_prod < 0:
            r_tracking_vel *= 0.1

        curr_joints = self.sim.data.qpos[7:]
        ref_joints = ref_qpos[7:]
        min_j_len = min(len(curr_joints), len(ref_joints))
        r_tracking_joint = np.exp(-2.0 * np.sum((curr_joints[:min_j_len] - ref_joints[:min_j_len]) ** 2))

        act_mag = 0.0
        if self.sim.model.na != 0:
            act_val = np.linalg.norm(obs_dict["act"], axis=-1)
            act_mag = float(act_val.item() if isinstance(act_val, np.ndarray) else act_val) / self.sim.model.na

        done_val = self._get_done()
        if isinstance(done_val, (list, np.ndarray)):
            done_val = float(done_val[0])

        rwd_dict = collections.OrderedDict(
            (
                ("tracking_joint", float(r_tracking_joint)),
                ("tracking_vel", float(r_tracking_vel)),
                ("vel_dir_bonus", float(vel_dir_reward)),
                ("upright_reward", float(r_upright)),
                ("face_front", float(r_face_front)),
                ("anti_hunch", float(r_pitch)),
                ("alive", 1.0),
                ("act_reg", float(-act_mag)),
                ("done", done_val),
                ("sparse", float(r_tracking_joint)),
                ("solved", r_tracking_joint > 0.8),
            )
        )
        rwd_dict["dense"] = np.sum([wt * rwd_dict.get(key, 0.0) for key, wt in self.rwd_keys_wt.items()], axis=0)
        if dot_prod < -0.1:
            rwd_dict["dense"] -= 2.0
        return rwd_dict

    def muscle_lengths(self):
        return self.sim.data.actuator_length

    def muscle_forces(self):
        return np.clip(self.sim.data.actuator_force / 1000, -100, 100)

    def muscle_velocities(self):
        return np.clip(self.sim.data.actuator_velocity, -100, 100)

    def _get_done(self):
        if self._get_height() < self.min_height:
            return 1
        if self._get_rot_condition():
            return 1
        return 0

    def _get_feet_heights(self):
        try:
            f_l = self.sim.model.body_name2id("talus_l")
            f_r = self.sim.model.body_name2id("talus_r")
            return np.array([self.sim.data.body_xpos[f_l][2], self.sim.data.body_xpos[f_r][2]])
        except Exception:
            return np.array([0.0, 0.0])

    def _get_feet_relative_position(self):
        try:
            f_l = self.sim.model.body_name2id("talus_l")
            f_r = self.sim.model.body_name2id("talus_r")
            pelvis = self.pelvis_id
            return np.array(
                [
                    self.sim.data.body_xpos[f_l] - self.sim.data.body_xpos[pelvis],
                    self.sim.data.body_xpos[f_r] - self.sim.data.body_xpos[pelvis],
                ]
            )
        except Exception:
            return np.zeros((2, 3))

    def _get_torso_angle(self):
        return self.sim.data.body_xquat[self.torso_id]

    def _get_com_velocity(self):
        return self.sim.data.qvel[:2].copy()

    def _get_height(self):
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        return (np.sum(mass * self.sim.data.xipos, 0) / np.sum(mass))[2]

    def _get_rot_condition(self):
        return int(abs((quat2mat(self.sim.data.qpos[3:7].copy()) @ [0, 0, 1])[2]) < self.max_rot)