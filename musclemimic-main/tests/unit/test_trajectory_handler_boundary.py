import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from loco_mujoco.trajectory.handler import TrajState, TrajectoryHandler


@struct.dataclass
class _FakeCarry:
    traj_state: TrajState


def _make_handler(lengths):
    handler = object.__new__(TrajectoryHandler)
    handler.len_trajectory = lambda traj_no: lengths[int(np.asarray(traj_no))]
    return handler


@pytest.mark.parametrize("backend", [np, jnp])
def test_reached_trajectory_end_detects_last_step(backend):
    handler = _make_handler([3])
    last_step = backend.asarray(2, dtype=backend.int32)
    prev_step = backend.asarray(1, dtype=backend.int32)

    at_end = handler.reached_trajectory_end(TrajState(0, last_step, 0), backend)
    before_end = handler.reached_trajectory_end(TrajState(0, prev_step, 0), backend)

    assert bool(np.asarray(at_end))
    assert not bool(np.asarray(before_end))


@pytest.mark.parametrize("backend", [np, jnp])
def test_update_state_advances_within_current_trajectory_without_wrap(backend):
    handler = _make_handler([4, 6])
    carry = _FakeCarry(
        traj_state=TrajState(
            traj_no=backend.asarray(0, dtype=backend.int32),
            subtraj_step_no=backend.asarray(2, dtype=backend.int32),
            subtraj_step_no_init=backend.asarray(1, dtype=backend.int32),
        )
    )

    next_carry = handler.update_state(None, None, None, carry, backend)

    assert int(np.asarray(next_carry.traj_state.traj_no)) == 0
    assert int(np.asarray(next_carry.traj_state.subtraj_step_no)) == 3
    assert int(np.asarray(next_carry.traj_state.subtraj_step_no_init)) == 1


@pytest.mark.parametrize("backend", [np, jnp])
def test_update_state_clamps_at_terminal_step_instead_of_switching_trajectory(backend):
    handler = _make_handler([4, 6])
    carry = _FakeCarry(
        traj_state=TrajState(
            traj_no=backend.asarray(0, dtype=backend.int32),
            subtraj_step_no=backend.asarray(3, dtype=backend.int32),
            subtraj_step_no_init=backend.asarray(0, dtype=backend.int32),
        )
    )

    next_carry = handler.update_state(None, None, None, carry, backend)

    assert int(np.asarray(next_carry.traj_state.traj_no)) == 0
    assert int(np.asarray(next_carry.traj_state.subtraj_step_no)) == 3
    assert int(np.asarray(next_carry.traj_state.subtraj_step_no_init)) == 0
