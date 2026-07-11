"""
Unit tests for the control mapping implemented by `DefaultControl`.

These tests validate the exact action -> ctrl transformation used by the environment:
- Agent outputs live in [-1, 1] (controller's action space)
- `DefaultControl` rescales to actuator ctrlrange and clamps to limits
- Incremental mode integrates deltas in actuator space and clamps
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from flax import struct

from loco_mujoco.core.control_functions.default import DefaultControl


# Keep tests runnable on CPU-only machines even when the project has CUDA deps installed.
jax.config.update("jax_platform_name", "cpu")


def _make_model_and_data() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Minimal model with 2 *muscle* actuators and distinct ctrl ranges.

    Mirrors the two common conventions in our model assets:
    - Some muscle actuators use ctrlrange [0, 1] (e.g., legs/torso defaults)
    - Some use ctrlrange [-1, 1] (e.g., arms defaults)
    """
    xml = """
    <mujoco model="test_default_control">
      <option timestep="0.01"/>
      <worldbody>
        <body name="body" pos="0 0 0">
          <joint name="j0" type="hinge" axis="0 1 0" range="-1 1"/>
          <joint name="j1" type="hinge" axis="1 0 0" range="-1 1"/>
          <geom type="sphere" size="0.05" mass="1"/>
          <site name="s0" pos="0 0 0" size="0.01"/>
          <site name="s1" pos="0 0 0.1" size="0.01"/>
        </body>
      </worldbody>
      <tendon>
        <spatial name="t0" width="0.001">
          <site site="s0"/>
          <site site="s1"/>
        </spatial>
        <spatial name="t1" width="0.001">
          <site site="s0"/>
          <site site="s1"/>
        </spatial>
      </tendon>
      <actuator>
        <!-- Parameters copied from the project model defaults (muscle actuator style). -->
        <general
          name="a0" tendon="t0"
          ctrllimited="true" ctrlrange="0 1"
          dyntype="muscle" gaintype="muscle" biastype="muscle"
          dynprm="0.01 0.04 0 0 0 0 0 0 0 0"
          gainprm="0.75 1.05 -1 400 0.5 1.6 1.5 1.3 1.2 0"
          biasprm="0.75 1.05 -1 400 0.5 1.6 1.5 1.3 1.2 0"
          lengthrange="0.05 0.2"
        />
        <general
          name="a1" tendon="t1"
          ctrllimited="true" ctrlrange="-1 1"
          dyntype="muscle" gaintype="muscle" biastype="muscle"
          dynprm="0.01 0.04 0 0 0 0 0 0 0 0"
          gainprm="0.75 1.05 -1 200 0.5 1.6 1.5 1.3 1.2 0"
          biasprm="0.75 1.05 -1 200 0.5 1.6 1.5 1.3 1.2 0"
          lengthrange="0.05 0.2"
        />
      </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data


@dataclass(frozen=True)
class _DummyEnv:
    _model: mujoco.MjModel
    _action_indices: np.ndarray
    _n_intermediate_steps: int = 1

    @property
    def simulation_dt(self) -> float:
        return float(self._model.opt.timestep)


@struct.dataclass
class _Carry:
    control_func_state: object


def _expected_direct_ctrl(
    action: np.ndarray,
    actuator_low: np.ndarray,
    actuator_high: np.ndarray,
) -> np.ndarray:
    mean = (actuator_high + actuator_low) / 2.0
    delta = (actuator_high - actuator_low) / 2.0
    return np.clip((action * delta) + mean, actuator_low, actuator_high)


def test_default_control_direct_matches_rescale_and_clip():
    model, data = _make_model_and_data()
    env = _DummyEnv(_model=model, _action_indices=np.arange(model.nu))
    control = DefaultControl(env, apply_mode="direct")

    low, high = control.action_limits
    np.testing.assert_allclose(low, -np.ones(model.nu))
    np.testing.assert_allclose(high, np.ones(model.nu))

    actuator_low = np.asarray(model.actuator_ctrlrange[:, 0])
    actuator_high = np.asarray(model.actuator_ctrlrange[:, 1])

    # Unbounded policy actions should saturate to actuator limits.
    action = np.array([100.0, -100.0], dtype=np.float32)
    ctrl, _ = control.generate_action(env, action, model, data, carry=None, backend=np)
    expected = _expected_direct_ctrl(action, actuator_low, actuator_high)
    np.testing.assert_allclose(ctrl, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(ctrl, np.array([1.0, -1.0], dtype=np.float32), rtol=0.0, atol=0.0)

    # Sanity: endpoints map exactly.
    action = np.array([-1.0, 1.0], dtype=np.float32)
    ctrl, _ = control.generate_action(env, action, model, data, carry=None, backend=np)
    np.testing.assert_allclose(ctrl, np.array([0.0, 1.0], dtype=np.float32), rtol=0.0, atol=0.0)


def test_default_control_incremental_integrates_prev_ctrl_and_clips():
    model, data = _make_model_and_data()
    env = _DummyEnv(_model=model, _action_indices=np.arange(model.nu))
    control = DefaultControl(env, apply_mode="incremental")

    state0 = control.init_state(env, key=None, model=model, data=data, backend=np)
    carry = _Carry(control_func_state=state0)
    np.testing.assert_allclose(np.asarray(carry.control_func_state.prev_ctrl), np.array([0.5, 0.0]))

    # 1) Move toward high end.
    ctrl, carry = control.generate_action(env, np.array([1.0, 1.0]), model, data, carry, backend=np)
    np.testing.assert_allclose(ctrl, np.array([1.0, 1.0]))
    np.testing.assert_allclose(np.asarray(carry.control_func_state.prev_ctrl), np.array([1.0, 1.0]))

    # 2) Saturate at limits.
    ctrl, carry = control.generate_action(env, np.array([1.0, 1.0]), model, data, carry, backend=np)
    np.testing.assert_allclose(ctrl, np.array([1.0, 1.0]))
    np.testing.assert_allclose(np.asarray(carry.control_func_state.prev_ctrl), np.array([1.0, 1.0]))

    # 3) Step back by one delta.
    ctrl, carry = control.generate_action(env, np.array([-1.0, -1.0]), model, data, carry, backend=np)
    np.testing.assert_allclose(ctrl, np.array([0.5, 0.0]))
    np.testing.assert_allclose(np.asarray(carry.control_func_state.prev_ctrl), np.array([0.5, 0.0]))

    # 4) Large magnitude actions are clipped to [-1, 1] before applying delta.
    ctrl, carry = control.generate_action(env, np.array([-100.0, -100.0]), model, data, carry, backend=np)
    np.testing.assert_allclose(ctrl, np.array([0.0, -1.0]))
    np.testing.assert_allclose(np.asarray(carry.control_func_state.prev_ctrl), np.array([0.0, -1.0]))


def test_default_control_direct_supports_jax_backend():
    model, data = _make_model_and_data()
    env = _DummyEnv(_model=model, _action_indices=np.arange(model.nu))
    control = DefaultControl(env, apply_mode="direct")

    action = jnp.array([100.0, -100.0], dtype=jnp.float32)
    ctrl, _ = control.generate_action(env, action, model, data, carry=None, backend=jnp)
    np.testing.assert_allclose(np.asarray(ctrl), np.array([1.0, -1.0], dtype=np.float32), rtol=0.0, atol=0.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
