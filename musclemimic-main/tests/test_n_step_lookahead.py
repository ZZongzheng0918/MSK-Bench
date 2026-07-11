"""
Tests to verify that n_step_lookahead is reflected in goal observation content and dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from loco_mujoco.core.utils.math import calculate_relative_site_quantities
from musclemimic.core.goals.bimanual import GoalBimanualTrajMimic
from musclemimic.core.goals.trajectory import GoalTrajMimic, GoalTrajMimicv2


# Keep tests runnable on CPU-only machines even when the project has CUDA deps installed.
jax.config.update("jax_platform_name", "cpu")


@dataclass(frozen=True)
class _TrajFrame:
    qpos: np.ndarray
    qvel: np.ndarray
    site_xpos: np.ndarray
    site_xmat: np.ndarray
    cvel: np.ndarray
    subtree_com: np.ndarray


class _DummyTrajData:
    def __init__(self, frames: Sequence[_TrajFrame]):
        self._frames = list(frames)

    def get(self, traj_no: int, subtraj_step_no: int, backend: Any) -> _TrajFrame:  # noqa: ARG002
        return self._frames[int(subtraj_step_no)]


@dataclass(frozen=True)
class _DummyTrajInfo:
    site_names: list[str]


@dataclass(frozen=True)
class _DummyTraj:
    data: _DummyTrajData
    info: _DummyTrajInfo


class _DummyTrajHandler:
    def __init__(self, frames: Sequence[_TrajFrame], site_names: list[str]):
        self.traj = _DummyTraj(data=_DummyTrajData(frames), info=_DummyTrajInfo(site_names=site_names))
        self._n_steps = len(frames)

    def len_trajectory(self, traj_ind: int) -> int:  # noqa: ARG002
        return self._n_steps

    def get_traj_data_at(self, traj_no: int, subtraj_step_no: int, carry: Any, backend: Any) -> _TrajFrame:  # noqa: ARG002
        return self.traj.data.get(traj_no, subtraj_step_no, backend)


def _make_identity_site_xmat(n_sites: int) -> np.ndarray:
    return np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (n_sites, 1))


def _make_fullbody_model_and_data() -> tuple[mujoco.MjModel, mujoco.MjData]:
    xml = """
    <mujoco model="test_fullbody_goal">
      <option timestep="0.01"/>
      <worldbody>
        <body name="torso" pos="0 0 0">
          <freejoint name="root"/>
          <geom name="torso_geom" type="sphere" size="0.05" mass="1"/>
          <site name="pelvis_mimic" pos="0 0 0" size="0.01"/>
          <body name="head" pos="0 0 1">
            <joint name="hinge" type="hinge" axis="0 1 0" pos="0 0 0" range="-1 1"/>
            <geom name="head_geom" type="sphere" size="0.03" mass="0.5" pos="0 0 0.2"/>
            <site name="head_mimic" pos="0 0 0.2" size="0.01"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def _make_bimanual_model_and_data() -> tuple[mujoco.MjModel, mujoco.MjData]:
    xml = """
    <mujoco model="test_bimanual_goal">
      <option timestep="0.01"/>
      <worldbody>
        <body name="torso" pos="0 0 0">
          <joint name="j0" type="hinge" axis="0 1 0" pos="0 0 0" range="-1 1"/>
          <joint name="j1" type="hinge" axis="1 0 0" pos="0 0 0" range="-1 1"/>
          <geom name="torso_geom" type="sphere" size="0.05" mass="1"/>
          <site name="site_main" pos="0 0 0" size="0.01"/>
          <site name="site_rel" pos="0 0 0.1" size="0.01"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def _build_frames(
    model: mujoco.MjModel,
    traj_site_names: list[str],
    site_positions_for_step: dict[str, callable],
    n_steps: int,
) -> list[_TrajFrame]:
    name_to_idx = {name: idx for idx, name in enumerate(traj_site_names)}
    frames: list[_TrajFrame] = []
    for step in range(n_steps):
        qpos = (np.arange(model.nq, dtype=np.float32) + step * 100.0).copy()
        qvel = (np.arange(model.nv, dtype=np.float32) + step * 1000.0).copy()

        site_xpos = np.zeros((len(traj_site_names), 3), dtype=np.float32)
        for site_name, pos_fn in site_positions_for_step.items():
            site_xpos[name_to_idx[site_name]] = np.asarray(pos_fn(step), dtype=np.float32)

        frames.append(
            _TrajFrame(
                qpos=qpos,
                qvel=qvel,
                site_xpos=site_xpos,
                site_xmat=_make_identity_site_xmat(len(traj_site_names)),
                cvel=np.zeros((model.nbody, 6), dtype=np.float32),
                subtree_com=np.zeros((model.nbody, 3), dtype=np.float32),
            )
        )
    return frames


def _current_site_block(goal: GoalTrajMimic, data: mujoco.MjData) -> np.ndarray:
    rel_site_ids = goal._rel_site_ids
    rel_body_ids = goal._site_bodyid[rel_site_ids]
    site_rpos, site_rangles, site_rvel = calculate_relative_site_quantities(
        data, rel_site_ids, rel_body_ids, goal._body_rootid, np
    )
    return np.concatenate(
        [
            np.ravel(site_rpos),
            np.ravel(site_rangles),
            np.ravel(site_rvel),
        ]
    )


@pytest.mark.parametrize("n_step_lookahead", [1, 3])
def test_goal_traj_mimic_n_step_lookahead_dim_and_content(n_step_lookahead: int):
    model, data = _make_fullbody_model_and_data()

    info_props = {"upper_body_xml_name": "torso", "sites_for_mimic": ["pelvis_mimic", "head_mimic"]}
    traj_site_names = ["head_mimic", "pelvis_mimic"]  # reversed order to exercise site mapping
    frames = _build_frames(
        model=model,
        traj_site_names=traj_site_names,
        site_positions_for_step={
            "pelvis_mimic": lambda _step: [0.0, 0.0, 0.0],
            "head_mimic": lambda step: [float(step), 0.0, 0.0],
        },
        n_steps=8,
    )
    th = _DummyTrajHandler(frames=frames, site_names=traj_site_names)

    class DummyMyoFullBodyEnv:
        root_free_joint_xml_name = "root"
        sites_for_mimic = info_props["sites_for_mimic"]

    env = DummyMyoFullBodyEnv()
    env.th = th

    goal = GoalTrajMimic(info_props, n_step_lookahead=n_step_lookahead, visualize_goal=False)
    goal._init_from_mj(env, model, data, current_obs_size=0)
    goal.init_from_traj(th)

    assert goal.n_step_lookahead == n_step_lookahead

    n_relative_sites = len(info_props["sites_for_mimic"]) - 1
    current_sites_dim = 12 * n_relative_sites
    traj_step_dim = len(goal._qpos_ind) + len(goal._qvel_ind) + current_sites_dim
    motion_phase_dim = 1 if goal.enable_motion_phase else 0
    assert goal.dim == traj_step_dim * n_step_lookahead + current_sites_dim + motion_phase_dim

    carry = SimpleNamespace(traj_state=SimpleNamespace(traj_no=0, subtraj_step_no=0))
    obs, _ = goal.get_obs_and_update_state(env, model, data, carry, backend=np)
    assert obs.shape == (goal.dim,)

    offset = current_sites_dim
    qpos_len = len(goal._qpos_ind) * n_step_lookahead
    qpos_slice = obs[offset : offset + qpos_len]
    offset += qpos_len

    qvel_len = len(goal._qvel_ind) * n_step_lookahead
    qvel_slice = obs[offset : offset + qvel_len]
    offset += qvel_len

    site_rpos_len = 3 * n_relative_sites * n_step_lookahead
    site_rpos_slice = obs[offset : offset + site_rpos_len]
    offset += site_rpos_len

    site_rangles_len = 3 * n_relative_sites * n_step_lookahead
    site_rangles_slice = obs[offset : offset + site_rangles_len]
    offset += site_rangles_len

    site_rvel_len = 6 * n_relative_sites * n_step_lookahead
    site_rvel_slice = obs[offset : offset + site_rvel_len]

    expected_qpos = np.concatenate([frames[i].qpos[goal._qpos_ind] for i in range(n_step_lookahead)])
    expected_qvel = np.concatenate([frames[i].qvel[goal._qvel_ind] for i in range(n_step_lookahead)])
    expected_site_rpos = np.concatenate([[float(i), 0.0, 0.0] for i in range(n_step_lookahead)])
    np.testing.assert_allclose(qpos_slice, expected_qpos)
    np.testing.assert_allclose(qvel_slice, expected_qvel)
    np.testing.assert_allclose(site_rpos_slice, expected_site_rpos)
    np.testing.assert_allclose(site_rangles_slice, np.zeros_like(site_rangles_slice))
    np.testing.assert_allclose(site_rvel_slice, np.zeros_like(site_rvel_slice))


def test_goal_traj_mimic_current_sites_prefix_matches_expected():
    model, data = _make_fullbody_model_and_data()

    info_props = {"upper_body_xml_name": "torso", "sites_for_mimic": ["pelvis_mimic", "head_mimic"]}
    traj_site_names = ["head_mimic", "pelvis_mimic"]
    frames = _build_frames(
        model=model,
        traj_site_names=traj_site_names,
        site_positions_for_step={
            "pelvis_mimic": lambda _step: [0.0, 0.0, 0.0],
            "head_mimic": lambda step: [float(step), 0.0, 0.0],
        },
        n_steps=4,
    )
    th = _DummyTrajHandler(frames=frames, site_names=traj_site_names)

    class DummyMyoFullBodyEnv:
        root_free_joint_xml_name = "root"
        sites_for_mimic = info_props["sites_for_mimic"]

    env = DummyMyoFullBodyEnv()
    env.th = th

    goal = GoalTrajMimic(info_props, n_step_lookahead=1, visualize_goal=False)
    goal._init_from_mj(env, model, data, current_obs_size=0)
    goal.init_from_traj(th)

    carry = SimpleNamespace(traj_state=SimpleNamespace(traj_no=0, subtraj_step_no=0))
    obs, _ = goal.get_obs_and_update_state(env, model, data, carry, backend=np)

    n_relative_sites = len(info_props["sites_for_mimic"]) - 1
    current_sites_dim = 12 * n_relative_sites
    current_block = obs[:current_sites_dim]

    expected = _current_site_block(goal, data)
    np.testing.assert_allclose(current_block, expected, atol=1e-6)


@pytest.mark.parametrize("n_step_lookahead", [1, 3])
def test_goal_traj_mimic_v2_n_step_lookahead_matches_v1(n_step_lookahead: int):
    model, data = _make_fullbody_model_and_data()

    info_props = {"upper_body_xml_name": "torso", "sites_for_mimic": ["pelvis_mimic", "head_mimic"]}
    traj_site_names = ["head_mimic", "pelvis_mimic"]  # reversed order to exercise site mapping
    frames = _build_frames(
        model=model,
        traj_site_names=traj_site_names,
        site_positions_for_step={
            "pelvis_mimic": lambda _step: [0.0, 0.0, 0.0],
            "head_mimic": lambda step: [float(step), 0.0, 0.0],
        },
        n_steps=8,
    )
    th = _DummyTrajHandler(frames=frames, site_names=traj_site_names)

    class DummyMyoFullBodyEnv:
        root_free_joint_xml_name = "root"
        sites_for_mimic = info_props["sites_for_mimic"]
        mjspec = SimpleNamespace(geoms=[])

    env = DummyMyoFullBodyEnv()
    env.th = th

    goal = GoalTrajMimicv2(info_props, n_step_lookahead=n_step_lookahead, visualize_goal=False)
    goal._init_from_mj(env, model, data, current_obs_size=0)
    goal.init_from_traj(th)

    assert goal.n_step_lookahead == n_step_lookahead

    n_relative_sites = len(info_props["sites_for_mimic"]) - 1
    current_sites_dim = 12 * n_relative_sites
    traj_step_dim = len(goal._qpos_ind) + len(goal._qvel_ind) + current_sites_dim
    motion_phase_dim = 1 if goal.enable_motion_phase else 0
    assert goal.dim == traj_step_dim * n_step_lookahead + current_sites_dim + motion_phase_dim

    carry = SimpleNamespace(traj_state=SimpleNamespace(traj_no=0, subtraj_step_no=0))
    obs, _ = goal.get_obs_and_update_state(env, model, data, carry, backend=np)
    assert obs.shape == (goal.dim,)

    offset = current_sites_dim
    qpos_len = len(goal._qpos_ind) * n_step_lookahead
    qpos_slice = obs[offset : offset + qpos_len]
    offset += qpos_len

    qvel_len = len(goal._qvel_ind) * n_step_lookahead
    qvel_slice = obs[offset : offset + qvel_len]
    offset += qvel_len

    site_rpos_len = 3 * n_relative_sites * n_step_lookahead
    site_rpos_slice = obs[offset : offset + site_rpos_len]
    offset += site_rpos_len

    site_rangles_len = 3 * n_relative_sites * n_step_lookahead
    site_rangles_slice = obs[offset : offset + site_rangles_len]
    offset += site_rangles_len

    site_rvel_len = 6 * n_relative_sites * n_step_lookahead
    site_rvel_slice = obs[offset : offset + site_rvel_len]

    expected_qpos = np.concatenate([frames[i].qpos[goal._qpos_ind] for i in range(n_step_lookahead)])
    expected_qvel = np.concatenate([frames[i].qvel[goal._qvel_ind] for i in range(n_step_lookahead)])
    expected_site_rpos = np.concatenate([[float(i), 0.0, 0.0] for i in range(n_step_lookahead)])
    np.testing.assert_allclose(qpos_slice, expected_qpos)
    np.testing.assert_allclose(qvel_slice, expected_qvel)
    np.testing.assert_allclose(site_rpos_slice, expected_site_rpos)
    np.testing.assert_allclose(site_rangles_slice, np.zeros_like(site_rangles_slice))
    np.testing.assert_allclose(site_rvel_slice, np.zeros_like(site_rvel_slice))


def test_goal_traj_mimic_v2_output_matches_v1():
    model, data = _make_fullbody_model_and_data()

    info_props = {"upper_body_xml_name": "torso", "sites_for_mimic": ["pelvis_mimic", "head_mimic"]}
    traj_site_names = ["head_mimic", "pelvis_mimic"]
    frames = _build_frames(
        model=model,
        traj_site_names=traj_site_names,
        site_positions_for_step={
            "pelvis_mimic": lambda _step: [0.0, 0.0, 0.0],
            "head_mimic": lambda step: [float(step), 0.0, 0.0],
        },
        n_steps=6,
    )
    th = _DummyTrajHandler(frames=frames, site_names=traj_site_names)

    class DummyMyoFullBodyEnv:
        root_free_joint_xml_name = "root"
        sites_for_mimic = info_props["sites_for_mimic"]
        mjspec = SimpleNamespace(geoms=[])

    env = DummyMyoFullBodyEnv()
    env.th = th

    goal_v1 = GoalTrajMimic(info_props, n_step_lookahead=2, visualize_goal=False)
    goal_v1._init_from_mj(env, model, data, current_obs_size=0)
    goal_v1.init_from_traj(th)

    goal_v2 = GoalTrajMimicv2(info_props, n_step_lookahead=2, visualize_goal=False)
    goal_v2._init_from_mj(env, model, data, current_obs_size=0)
    goal_v2.init_from_traj(th)

    carry = SimpleNamespace(traj_state=SimpleNamespace(traj_no=0, subtraj_step_no=1))
    obs_v1, _ = goal_v1.get_obs_and_update_state(env, model, data, carry, backend=np)
    obs_v2, _ = goal_v2.get_obs_and_update_state(env, model, data, carry, backend=np)

    np.testing.assert_allclose(obs_v2, obs_v1, atol=1e-6)


def test_goal_traj_mimic_clamps_lookahead_to_last_frame():
    model, data = _make_fullbody_model_and_data()

    info_props = {"upper_body_xml_name": "torso", "sites_for_mimic": ["pelvis_mimic", "head_mimic"]}
    traj_site_names = ["head_mimic", "pelvis_mimic"]
    frames = _build_frames(
        model=model,
        traj_site_names=traj_site_names,
        site_positions_for_step={
            "pelvis_mimic": lambda _step: [0.0, 0.0, 0.0],
            "head_mimic": lambda step: [float(step), 0.0, 0.0],
        },
        n_steps=2,
    )
    th = _DummyTrajHandler(frames=frames, site_names=traj_site_names)

    class DummyMyoFullBodyEnv:
        root_free_joint_xml_name = "root"
        sites_for_mimic = info_props["sites_for_mimic"]

    env = DummyMyoFullBodyEnv()
    env.th = th

    n_step_lookahead = 4
    goal = GoalTrajMimic(info_props, n_step_lookahead=n_step_lookahead, visualize_goal=False)
    goal._init_from_mj(env, model, data, current_obs_size=0)
    goal.init_from_traj(th)

    carry = SimpleNamespace(traj_state=SimpleNamespace(traj_no=0, subtraj_step_no=1))
    obs, _ = goal.get_obs_and_update_state(env, model, data, carry, backend=np)

    n_relative_sites = len(info_props["sites_for_mimic"]) - 1
    current_sites_dim = 12 * n_relative_sites
    offset = current_sites_dim

    qpos_len = len(goal._qpos_ind) * n_step_lookahead
    qpos_slice = obs[offset : offset + qpos_len]
    offset += qpos_len

    qvel_len = len(goal._qvel_ind) * n_step_lookahead
    qvel_slice = obs[offset : offset + qvel_len]

    expected_qpos = np.concatenate([frames[1].qpos[goal._qpos_ind]] * n_step_lookahead)
    expected_qvel = np.concatenate([frames[1].qvel[goal._qvel_ind]] * n_step_lookahead)
    np.testing.assert_allclose(qpos_slice, expected_qpos)
    np.testing.assert_allclose(qvel_slice, expected_qvel)


def test_goal_traj_mimic_jax_backend_matches_numpy():
    model, data = _make_fullbody_model_and_data()

    info_props = {"upper_body_xml_name": "torso", "sites_for_mimic": ["pelvis_mimic", "head_mimic"]}
    traj_site_names = ["head_mimic", "pelvis_mimic"]
    frames = _build_frames(
        model=model,
        traj_site_names=traj_site_names,
        site_positions_for_step={
            "pelvis_mimic": lambda _step: [0.0, 0.0, 0.0],
            "head_mimic": lambda step: [float(step), 0.0, 0.0],
        },
        n_steps=5,
    )
    th = _DummyTrajHandler(frames=frames, site_names=traj_site_names)

    class DummyMyoFullBodyEnv:
        root_free_joint_xml_name = "root"
        sites_for_mimic = info_props["sites_for_mimic"]

    env = DummyMyoFullBodyEnv()
    env.th = th

    goal = GoalTrajMimic(info_props, n_step_lookahead=3, visualize_goal=False)
    goal._init_from_mj(env, model, data, current_obs_size=0)
    goal.init_from_traj(th)

    carry = SimpleNamespace(traj_state=SimpleNamespace(traj_no=0, subtraj_step_no=0))
    obs_np, _ = goal.get_obs_and_update_state(env, model, data, carry, backend=np)
    obs_jnp, _ = goal.get_obs_and_update_state(env, model, data, carry, backend=jnp)

    np.testing.assert_allclose(np.asarray(obs_jnp), obs_np, atol=1e-6)


def test_goal_traj_mimic_concise_lookahead_jax_backend_matches_numpy():
    """Test concise lookahead works with JAX backend (uses root indices for deltas)."""
    model, data = _make_fullbody_model_and_data()

    info_props = {"upper_body_xml_name": "torso", "sites_for_mimic": ["pelvis_mimic", "head_mimic"]}
    traj_site_names = ["head_mimic", "pelvis_mimic"]
    frames = _build_frames(
        model=model,
        traj_site_names=traj_site_names,
        site_positions_for_step={
            "pelvis_mimic": lambda _step: [0.0, 0.0, 0.0],
            "head_mimic": lambda step: [float(step), 0.0, 0.0],
        },
        n_steps=10,
    )
    th = _DummyTrajHandler(frames=frames, site_names=traj_site_names)

    class DummyMyoFullBodyEnv:
        root_free_joint_xml_name = "root"
        sites_for_mimic = info_props["sites_for_mimic"]

    env = DummyMyoFullBodyEnv()
    env.th = th

    # Test with use_concise_lookahead=True which uses root indices
    goal = GoalTrajMimic(info_props, n_step_lookahead=3, use_concise_lookahead=True, visualize_goal=False)
    goal._init_from_mj(env, model, data, current_obs_size=0)
    goal.init_from_traj(th)

    carry = SimpleNamespace(traj_state=SimpleNamespace(traj_no=0, subtraj_step_no=0))

    # This should not raise JAX indexing errors
    obs_np, _ = goal.get_obs_and_update_state(env, model, data, carry, backend=np)
    obs_jnp, _ = goal.get_obs_and_update_state(env, model, data, carry, backend=jnp)

    np.testing.assert_allclose(np.asarray(obs_jnp), obs_np, atol=1e-6)


@pytest.mark.parametrize("n_step_lookahead", [1, 3])
def test_goal_traj_mimic_concise_lookahead_dim_and_content(n_step_lookahead: int):
    """Test concise lookahead observation dimensions and content structure."""
    model, data = _make_fullbody_model_and_data()

    info_props = {"upper_body_xml_name": "torso", "sites_for_mimic": ["pelvis_mimic", "head_mimic"]}
    traj_site_names = ["head_mimic", "pelvis_mimic"]
    frames = _build_frames(
        model=model,
        traj_site_names=traj_site_names,
        site_positions_for_step={
            "pelvis_mimic": lambda _step: [0.0, 0.0, 0.0],
            "head_mimic": lambda step: [float(step), 0.0, 0.0],
        },
        n_steps=10,
    )
    th = _DummyTrajHandler(frames=frames, site_names=traj_site_names)

    class DummyMyoFullBodyEnv:
        root_free_joint_xml_name = "root"
        sites_for_mimic = info_props["sites_for_mimic"]

    env = DummyMyoFullBodyEnv()
    env.th = th

    goal = GoalTrajMimic(info_props, n_step_lookahead=n_step_lookahead, use_concise_lookahead=True, visualize_goal=False)
    goal._init_from_mj(env, model, data, current_obs_size=0)
    goal.init_from_traj(th)

    # Verify dimension calculation for concise lookahead
    n_relative_sites = len(info_props["sites_for_mimic"]) - 1
    current_sites_dim = 12 * n_relative_sites  # rpos(3) + rangles(3) + rvel(6)
    site_rpos_per_step = 3 * n_relative_sites
    root_delta_per_step = 3 + 6  # pos(3) + vel(6)
    motion_phase_dim = 1 if goal.enable_motion_phase else 0

    # Structure: step0_rpos + (root_pos_delta + root_vel_delta + site_rpos) * (n_step_lookahead - 1)
    if n_step_lookahead > 1:
        expected_traj_dim = site_rpos_per_step + (root_delta_per_step + site_rpos_per_step) * (n_step_lookahead - 1)
    else:
        expected_traj_dim = site_rpos_per_step

    expected_dim = current_sites_dim + expected_traj_dim + motion_phase_dim
    assert goal.dim == expected_dim, f"Expected dim {expected_dim}, got {goal.dim}"

    carry = SimpleNamespace(traj_state=SimpleNamespace(traj_no=0, subtraj_step_no=0))
    obs, _ = goal.get_obs_and_update_state(env, model, data, carry, backend=np)
    assert obs.shape == (goal.dim,), f"Expected shape ({goal.dim},), got {obs.shape}"


def test_goal_traj_mimic_stride_applies_to_targets():
    model, data = _make_fullbody_model_and_data()

    info_props = {"upper_body_xml_name": "torso", "sites_for_mimic": ["pelvis_mimic", "head_mimic"]}
    traj_site_names = ["head_mimic", "pelvis_mimic"]
    frames = _build_frames(
        model=model,
        traj_site_names=traj_site_names,
        site_positions_for_step={
            "pelvis_mimic": lambda _step: [0.0, 0.0, 0.0],
            "head_mimic": lambda step: [float(step), 0.0, 0.0],
        },
        n_steps=120,
    )
    th = _DummyTrajHandler(frames=frames, site_names=traj_site_names)

    class DummyMyoFullBodyEnv:
        root_free_joint_xml_name = "root"
        sites_for_mimic = info_props["sites_for_mimic"]

    env = DummyMyoFullBodyEnv()
    env.th = th

    goal = GoalTrajMimic(info_props, n_step_lookahead=2, n_step_stride=100, visualize_goal=False)
    goal._init_from_mj(env, model, data, current_obs_size=0)
    goal.init_from_traj(th)

    assert goal.n_step_stride == 100

    carry = SimpleNamespace(traj_state=SimpleNamespace(traj_no=0, subtraj_step_no=0))
    obs, _ = goal.get_obs_and_update_state(env, model, data, carry, backend=np)

    n_relative_sites = len(info_props["sites_for_mimic"]) - 1
    current_sites_dim = 12 * n_relative_sites
    offset = current_sites_dim

    qpos_len = len(goal._qpos_ind) * 2
    qpos_slice = obs[offset : offset + qpos_len]
    offset += qpos_len

    qvel_len = len(goal._qvel_ind) * 2
    qvel_slice = obs[offset : offset + qvel_len]
    offset += qvel_len

    site_rpos_len = 3 * n_relative_sites * 2
    site_rpos_slice = obs[offset : offset + site_rpos_len]
    offset += site_rpos_len

    site_rangles_len = 3 * n_relative_sites * 2
    site_rangles_slice = obs[offset : offset + site_rangles_len]
    offset += site_rangles_len

    site_rvel_len = 6 * n_relative_sites * 2
    site_rvel_slice = obs[offset : offset + site_rvel_len]

    expected_qpos = np.concatenate([frames[0].qpos[goal._qpos_ind], frames[100].qpos[goal._qpos_ind]])
    expected_qvel = np.concatenate([frames[0].qvel[goal._qvel_ind], frames[100].qvel[goal._qvel_ind]])
    expected_site_rpos = np.concatenate([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    np.testing.assert_allclose(qpos_slice, expected_qpos)
    np.testing.assert_allclose(qvel_slice, expected_qvel)
    np.testing.assert_allclose(site_rpos_slice, expected_site_rpos)
    np.testing.assert_allclose(site_rangles_slice, np.zeros_like(site_rangles_slice))
    np.testing.assert_allclose(site_rvel_slice, np.zeros_like(site_rvel_slice))


def test_goal_traj_mimic_invalid_n_step_lookahead_raises():
    info_props = {"upper_body_xml_name": "torso", "sites_for_mimic": ["pelvis_mimic", "head_mimic"]}
    with pytest.raises(ValueError):
        GoalTrajMimic(info_props, n_step_lookahead=0, visualize_goal=False)
    with pytest.raises(ValueError):
        GoalTrajMimic(info_props, n_step_lookahead=-2, visualize_goal=False)


def test_goal_traj_mimic_invalid_n_step_stride_raises():
    info_props = {"upper_body_xml_name": "torso", "sites_for_mimic": ["pelvis_mimic", "head_mimic"]}
    with pytest.raises(ValueError):
        GoalTrajMimic(info_props, n_step_stride=0, visualize_goal=False)
    with pytest.raises(ValueError):
        GoalTrajMimic(info_props, n_step_stride=-4, visualize_goal=False)


@pytest.mark.parametrize("n_step_lookahead", [1, 3])
def test_goal_bimanual_traj_mimic_n_step_lookahead_dim_and_content(n_step_lookahead: int):
    model, data = _make_bimanual_model_and_data()

    info_props = {"upper_body_xml_name": "torso", "sites_for_mimic": ["site_main", "site_rel"]}
    traj_site_names = ["site_rel", "site_main"]  # reversed order to exercise site mapping
    frames = _build_frames(
        model=model,
        traj_site_names=traj_site_names,
        site_positions_for_step={
            "site_main": lambda _step: [0.0, 0.0, 0.0],
            "site_rel": lambda step: [0.0, float(step), 0.0],
        },
        n_steps=8,
    )
    th = _DummyTrajHandler(frames=frames, site_names=traj_site_names)

    class DummyMyoBimanualArmEnv:
        sites_for_mimic = info_props["sites_for_mimic"]

    env = DummyMyoBimanualArmEnv()
    env.th = th

    goal = GoalBimanualTrajMimic(info_props, n_step_lookahead=n_step_lookahead, visualize_goal=False)
    goal._init_from_mj(env, model, data, current_obs_size=0)
    goal.init_from_traj(th)

    assert goal.n_step_lookahead == n_step_lookahead

    n_relative_sites = len(info_props["sites_for_mimic"]) - 1
    current_sites_dim = 12 * n_relative_sites
    traj_step_dim = len(goal._qpos_ind) + len(goal._qvel_ind) + current_sites_dim
    motion_phase_dim = 1 if goal.enable_motion_phase else 0
    assert goal.dim == traj_step_dim * n_step_lookahead + current_sites_dim + motion_phase_dim

    carry = SimpleNamespace(traj_state=SimpleNamespace(traj_no=0, subtraj_step_no=0))
    obs, _ = goal.get_obs_and_update_state(env, model, data, carry, backend=np)
    assert obs.shape == (goal.dim,)

    offset = current_sites_dim
    qpos_len = len(goal._qpos_ind) * n_step_lookahead
    qpos_slice = obs[offset : offset + qpos_len]
    offset += qpos_len

    qvel_len = len(goal._qvel_ind) * n_step_lookahead
    qvel_slice = obs[offset : offset + qvel_len]
    offset += qvel_len

    site_rpos_len = 3 * n_relative_sites * n_step_lookahead
    site_rpos_slice = obs[offset : offset + site_rpos_len]
    offset += site_rpos_len

    site_rangles_len = 3 * n_relative_sites * n_step_lookahead
    site_rangles_slice = obs[offset : offset + site_rangles_len]
    offset += site_rangles_len

    site_rvel_len = 6 * n_relative_sites * n_step_lookahead
    site_rvel_slice = obs[offset : offset + site_rvel_len]

    expected_qpos = np.concatenate([frames[i].qpos[goal._qpos_ind] for i in range(n_step_lookahead)])
    expected_qvel = np.concatenate([frames[i].qvel[goal._qvel_ind] for i in range(n_step_lookahead)])
    expected_site_rpos = np.concatenate([[0.0, float(i), 0.0] for i in range(n_step_lookahead)])
    np.testing.assert_allclose(qpos_slice, expected_qpos)
    np.testing.assert_allclose(qvel_slice, expected_qvel)
    np.testing.assert_allclose(site_rpos_slice, expected_site_rpos)
    np.testing.assert_allclose(site_rangles_slice, np.zeros_like(site_rangles_slice))
    np.testing.assert_allclose(site_rvel_slice, np.zeros_like(site_rvel_slice))


def test_goal_bimanual_traj_mimic_stride_applies_to_targets():
    model, data = _make_bimanual_model_and_data()

    info_props = {"upper_body_xml_name": "torso", "sites_for_mimic": ["site_main", "site_rel"]}
    traj_site_names = ["site_rel", "site_main"]
    frames = _build_frames(
        model=model,
        traj_site_names=traj_site_names,
        site_positions_for_step={
            "site_main": lambda _step: [0.0, 0.0, 0.0],
            "site_rel": lambda step: [0.0, float(step), 0.0],
        },
        n_steps=6,
    )
    th = _DummyTrajHandler(frames=frames, site_names=traj_site_names)

    class DummyMyoBimanualArmEnv:
        sites_for_mimic = info_props["sites_for_mimic"]

    env = DummyMyoBimanualArmEnv()
    env.th = th

    goal = GoalBimanualTrajMimic(
        info_props, n_step_lookahead=3, n_step_stride=2, visualize_goal=False
    )
    goal._init_from_mj(env, model, data, current_obs_size=0)
    goal.init_from_traj(th)

    carry = SimpleNamespace(traj_state=SimpleNamespace(traj_no=0, subtraj_step_no=0))
    obs, _ = goal.get_obs_and_update_state(env, model, data, carry, backend=np)

    n_relative_sites = len(info_props["sites_for_mimic"]) - 1
    current_sites_dim = 12 * n_relative_sites
    offset = current_sites_dim

    qpos_len = len(goal._qpos_ind) * 3
    qpos_slice = obs[offset : offset + qpos_len]
    offset += qpos_len

    qvel_len = len(goal._qvel_ind) * 3
    qvel_slice = obs[offset : offset + qvel_len]
    offset += qvel_len

    site_rpos_len = 3 * n_relative_sites * 3
    site_rpos_slice = obs[offset : offset + site_rpos_len]
    offset += site_rpos_len

    site_rangles_len = 3 * n_relative_sites * 3
    site_rangles_slice = obs[offset : offset + site_rangles_len]
    offset += site_rangles_len

    site_rvel_len = 6 * n_relative_sites * 3
    site_rvel_slice = obs[offset : offset + site_rvel_len]

    expected_qpos = np.concatenate([frames[i].qpos[goal._qpos_ind] for i in (0, 2, 4)])
    expected_qvel = np.concatenate([frames[i].qvel[goal._qvel_ind] for i in (0, 2, 4)])
    expected_site_rpos = np.concatenate([[0.0, float(i), 0.0] for i in (0, 2, 4)])
    np.testing.assert_allclose(qpos_slice, expected_qpos)
    np.testing.assert_allclose(qvel_slice, expected_qvel)
    np.testing.assert_allclose(site_rpos_slice, expected_site_rpos)
    np.testing.assert_allclose(site_rangles_slice, np.zeros_like(site_rangles_slice))
    np.testing.assert_allclose(site_rvel_slice, np.zeros_like(site_rvel_slice))
