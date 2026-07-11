"""Tests for NStepWrapper with split_goal functionality."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from musclemimic.core.wrappers.mjx import NStepWrapper, NStepWrapperState
from loco_mujoco.core.utils import Box


# =============================================================================
# Mock classes for testing
# =============================================================================

class MockGoal:
    """Mock goal with configurable dimension."""
    def __init__(self, dim: int):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim


class MockObsContainer:
    """Mock obs_container for testing."""
    def __init__(self, state_dim: int, goal_dim: int):
        self._goal_indices = np.arange(state_dim, state_dim + goal_dim)

    def get_obs_ind_by_group(self, group_name: str) -> np.ndarray:
        if group_name == "goal":
            return self._goal_indices
        return np.array([])


class MockMDPInfo:
    """Mock MDPInfo with observation_space."""
    def __init__(self, obs_dim: int):
        self.observation_space = Box(
            low=np.full(obs_dim, -np.inf),
            high=np.full(obs_dim, np.inf)
        )


@struct.dataclass
class MockEnvState:
    """Mock environment state."""
    step_count: int = 0


class MockEnv:
    """Mock environment for testing NStepWrapper."""

    def __init__(self, state_dim: int = 10, goal_dim: int = 5):
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.obs_dim = state_dim + goal_dim
        self._goal = MockGoal(goal_dim)
        self.info = MockMDPInfo(self.obs_dim)
        self.obs_container = MockObsContainer(state_dim, goal_dim)
        self.mjx_env = False  # Not an MJX env for simple testing

    def reset(self, rng_key):
        """Return obs where state=[1,2,3,...] and goal=[100,101,...]."""
        state_obs = jnp.arange(1, self.state_dim + 1, dtype=jnp.float32)
        goal_obs = jnp.arange(100, 100 + self.goal_dim, dtype=jnp.float32)
        obs = jnp.concatenate([state_obs, goal_obs])
        env_state = MockEnvState(step_count=0)
        return obs, env_state

    def step(self, state: MockEnvState, action):
        """Increment obs values each step."""
        step = state.step_count + 1
        state_obs = jnp.arange(1, self.state_dim + 1, dtype=jnp.float32) + step
        goal_obs = jnp.arange(100, 100 + self.goal_dim, dtype=jnp.float32) + step * 10
        obs = jnp.concatenate([state_obs, goal_obs])
        new_state = MockEnvState(step_count=step)
        reward = 0.0
        absorbing = False
        done = False
        info = {}
        return obs, reward, absorbing, done, info, new_state


# =============================================================================
# Tests for original NStepWrapper behavior (split_goal=False)
# =============================================================================

class TestNStepWrapperOriginal:
    """Tests for NStepWrapper without split_goal."""

    def test_init_without_split_goal(self):
        """Test initialization with split_goal=False."""
        env = MockEnv(state_dim=10, goal_dim=5)
        wrapper = NStepWrapper(env, n_steps=3, split_goal=False)

        assert wrapper.n_steps == 3
        assert wrapper.split_goal == False
        assert wrapper.goal_dim == 0
        assert wrapper.state_dim == 0

    def test_reset_stacks_full_obs(self):
        """Test that reset stacks the full observation."""
        env = MockEnv(state_dim=10, goal_dim=5)
        wrapper = NStepWrapper(env, n_steps=3, split_goal=False)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        # Output should be n_steps * obs_dim
        expected_dim = 3 * 15  # 3 steps * (10 state + 5 goal)
        assert obs.shape == (expected_dim,)

        # Buffer should have shape (n_steps, obs_dim)
        assert state.observation_buffer.shape == (3, 15)

    def test_step_rolls_buffer(self):
        """Test that step rolls the buffer correctly."""
        env = MockEnv(state_dim=10, goal_dim=5)
        wrapper = NStepWrapper(env, n_steps=3, split_goal=False)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        # Take a step
        action = jnp.zeros(1)
        next_obs, _, _, _, _, next_state = wrapper.step(state, action)

        # Buffer should have rolled
        assert next_obs.shape == (45,)  # 3 * 15


# =============================================================================
# Tests for split_goal functionality
# =============================================================================

class TestNStepWrapperSplitGoal:
    """Tests for NStepWrapper with split_goal=True."""

    def test_init_with_split_goal(self):
        """Test initialization with split_goal=True."""
        env = MockEnv(state_dim=10, goal_dim=5)
        wrapper = NStepWrapper(env, n_steps=3, split_goal=True)

        assert wrapper.n_steps == 3
        assert wrapper.split_goal == True
        assert wrapper.goal_dim == 5
        assert wrapper.state_dim == 10
        assert wrapper.raw_obs_dim == 15

    def test_reset_output_shape(self):
        """Test that reset returns correct shape with split_goal."""
        env = MockEnv(state_dim=10, goal_dim=5)
        wrapper = NStepWrapper(env, n_steps=3, split_goal=True)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        # Output: n_steps * state_dim + goal_dim
        expected_dim = 3 * 10 + 5  # 35
        assert obs.shape == (expected_dim,)

        # Buffer should only store state: (n_steps, state_dim)
        assert state.observation_buffer.shape == (3, 10)

    def test_reset_state_history_structure(self):
        """Test that reset correctly structures state history."""
        env = MockEnv(state_dim=4, goal_dim=2)
        wrapper = NStepWrapper(env, n_steps=3, split_goal=True)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        # State obs from env: [1, 2, 3, 4]
        # Goal obs from env: [100, 101]
        # Buffer after reset: [[0,0,0,0], [0,0,0,0], [1,2,3,4]]
        # Output: [0,0,0,0, 0,0,0,0, 1,2,3,4, 100,101]

        expected_state_hist = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4], dtype=np.float32)
        expected_goal = np.array([100, 101], dtype=np.float32)
        expected_obs = np.concatenate([expected_state_hist, expected_goal])

        np.testing.assert_allclose(obs, expected_obs)

    def test_step_rolls_only_state(self):
        """Test that step only rolls state, keeps current goal."""
        env = MockEnv(state_dim=4, goal_dim=2)
        wrapper = NStepWrapper(env, n_steps=3, split_goal=True)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        # Take first step
        action = jnp.zeros(1)
        obs1, _, _, _, _, state1 = wrapper.step(state, action)

        # After step 1:
        # - State from env: [2, 3, 4, 5] (original + 1)
        # - Goal from env: [110, 111] (original + 10)
        # - Buffer: [[0,0,0,0], [1,2,3,4], [2,3,4,5]]
        # - Output: [0,0,0,0, 1,2,3,4, 2,3,4,5, 110,111]

        expected_state_hist = np.array([0, 0, 0, 0, 1, 2, 3, 4, 2, 3, 4, 5], dtype=np.float32)
        expected_goal = np.array([110, 111], dtype=np.float32)
        expected_obs = np.concatenate([expected_state_hist, expected_goal])

        np.testing.assert_allclose(obs1, expected_obs)

    def test_step_updates_goal_each_step(self):
        """Test that goal is updated to current timestep each step."""
        env = MockEnv(state_dim=4, goal_dim=2)
        wrapper = NStepWrapper(env, n_steps=2, split_goal=True)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        # Take two steps
        action = jnp.zeros(1)
        obs1, _, _, _, _, state1 = wrapper.step(state, action)
        obs2, _, _, _, _, state2 = wrapper.step(state1, action)

        # Goal should be from step 2: [120, 121]
        goal_from_obs2 = obs2[-2:]
        np.testing.assert_allclose(goal_from_obs2, np.array([120, 121], dtype=np.float32))

    def test_buffer_does_not_contain_goal(self):
        """Test that observation_buffer only contains state observations."""
        env = MockEnv(state_dim=4, goal_dim=2)
        wrapper = NStepWrapper(env, n_steps=3, split_goal=True)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        # Buffer should be (3, 4) not (3, 6)
        assert state.observation_buffer.shape == (3, 4)

        # Take a step
        action = jnp.zeros(1)
        _, _, _, _, _, next_state = wrapper.step(state, action)

        # Buffer shape should remain the same
        assert next_state.observation_buffer.shape == (3, 4)

    def test_observation_space_updated_correctly(self):
        """Test that info.observation_space is updated for split_goal."""
        env = MockEnv(state_dim=10, goal_dim=5)
        wrapper = NStepWrapper(env, n_steps=3, split_goal=True)

        # Expected: n_steps * state_dim + goal_dim = 3 * 10 + 5 = 35
        expected_dim = 35
        assert wrapper.info.observation_space.shape == (expected_dim,)


# =============================================================================
# Tests for JAX compatibility
# =============================================================================

class TestNStepWrapperJAXCompatibility:
    """Tests for JAX scan/vmap compatibility."""

    def test_jit_compatible_reset(self):
        """Test that reset can be JIT compiled."""
        env = MockEnv(state_dim=4, goal_dim=2)
        wrapper = NStepWrapper(env, n_steps=3, split_goal=True)

        @jax.jit
        def jit_reset(rng):
            return wrapper.reset(rng)

        rng = jax.random.PRNGKey(0)
        obs, state = jit_reset(rng)

        assert obs.shape == (3 * 4 + 2,)

    def test_jit_compatible_step(self):
        """Test that step can be JIT compiled."""
        env = MockEnv(state_dim=4, goal_dim=2)
        wrapper = NStepWrapper(env, n_steps=3, split_goal=True)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        @jax.jit
        def jit_step(state, action):
            return wrapper.step(state, action)

        action = jnp.zeros(1)
        next_obs, _, _, _, _, next_state = jit_step(state, action)

        assert next_obs.shape == (3 * 4 + 2,)

    def test_scan_compatible(self):
        """Test that wrapper works with jax.lax.scan."""
        env = MockEnv(state_dim=4, goal_dim=2)
        wrapper = NStepWrapper(env, n_steps=3, split_goal=True)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        def scan_step(carry, _):
            state = carry
            action = jnp.zeros(1)
            next_obs, reward, _, _, _, next_state = wrapper.step(state, action)
            return next_state, next_obs

        # Run 5 steps with scan
        final_state, obs_trajectory = jax.lax.scan(scan_step, state, None, length=5)

        assert obs_trajectory.shape == (5, 3 * 4 + 2)


# =============================================================================
# Edge case tests
# =============================================================================

class TestNStepWrapperEdgeCases:
    """Tests for edge cases."""

    def test_n_steps_one_with_split_goal(self):
        """Test with n_steps=1 (no actual history stacking)."""
        env = MockEnv(state_dim=4, goal_dim=2)
        wrapper = NStepWrapper(env, n_steps=1, split_goal=True)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        # Output: 1 * state_dim + goal_dim = 4 + 2 = 6
        assert obs.shape == (6,)

        # Should be equivalent to original obs
        expected = np.array([1, 2, 3, 4, 100, 101], dtype=np.float32)
        np.testing.assert_allclose(obs, expected)

    def test_large_n_steps(self):
        """Test with larger n_steps value."""
        env = MockEnv(state_dim=10, goal_dim=5)
        wrapper = NStepWrapper(env, n_steps=20, split_goal=True)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        # Output: 20 * 10 + 5 = 205
        assert obs.shape == (205,)
        assert state.observation_buffer.shape == (20, 10)

    def test_goal_dim_zero(self):
        """Test when goal_dim is 0 (NoGoal case) - split_goal should raise error."""
        env = MockEnv(state_dim=10, goal_dim=0)
        # split_goal=True with no goal indices should raise an error
        with pytest.raises(ValueError, match="split_goal=True requires goal observations"):
            NStepWrapper(env, n_steps=3, split_goal=True)

    def test_goal_dim_zero_without_split_goal(self):
        """Test when goal_dim is 0 without split_goal - should work normally."""
        env = MockEnv(state_dim=10, goal_dim=0)
        wrapper = NStepWrapper(env, n_steps=3, split_goal=False)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        # Output: 3 * 10 = 30 (full obs stacking)
        assert obs.shape == (30,)
