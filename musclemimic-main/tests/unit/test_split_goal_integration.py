"""Integration tests for split_goal feature.

Tests the full pipeline: env -> NStepWrapper -> expand_obs_indices_for_history
to ensure state observations are stacked while goal observations stay current.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from omegaconf import OmegaConf

from loco_mujoco.core.utils import Box
from musclemimic.algorithms.common.env_utils import expand_obs_indices_for_history, wrap_env
from musclemimic.core.wrappers.mjx import NStepWrapper


# =============================================================================
# Mock classes that simulate real environment structure
# =============================================================================


class MockObsContainer:
    """Mock obs_container with multiple observation groups."""

    def __init__(self, obs_groups: dict[str, np.ndarray]):
        """
        Args:
            obs_groups: Dict mapping group name to indices, e.g.
                {"state": [0,1,2], "goal": [3,4,5]}
        """
        self._obs_groups = {k: np.asarray(v) for k, v in obs_groups.items()}

    def get_obs_ind_by_group(self, group_name: str) -> np.ndarray:
        return self._obs_groups.get(group_name, np.array([]))


class MockMDPInfo:
    """Mock MDPInfo with observation_space."""

    def __init__(self, obs_dim: int):
        self.observation_space = Box(
            low=np.full(obs_dim, -np.inf),
            high=np.full(obs_dim, np.inf),
        )


@struct.dataclass
class MockEnvState:
    """Mock environment state."""

    step_count: int = 0


class MockEnvWithObsGroups:
    """Mock environment with configurable observation groups.

    Simulates a real environment with:
    - Joint observations (state)
    - Muscle observations (state)
    - Goal observations (goal)
    """

    def __init__(
        self,
        joint_dim: int = 10,
        muscle_dim: int = 5,
        goal_dim: int = 8,
    ):
        self.joint_dim = joint_dim
        self.muscle_dim = muscle_dim
        self.goal_dim = goal_dim
        self.state_dim = joint_dim + muscle_dim
        self.obs_dim = self.state_dim + goal_dim

        # Define observation groups
        joint_indices = np.arange(0, joint_dim)
        muscle_indices = np.arange(joint_dim, joint_dim + muscle_dim)
        goal_indices = np.arange(joint_dim + muscle_dim, self.obs_dim)

        self.obs_container = MockObsContainer({
            "joint": joint_indices,
            "muscle": muscle_indices,
            "state": np.concatenate([joint_indices, muscle_indices]),
            "goal": goal_indices,
        })

        self.info = MockMDPInfo(self.obs_dim)
        self.mjx_env = False

    def reset(self, rng_key):
        """Return structured observation."""
        joint_obs = jnp.arange(1, self.joint_dim + 1, dtype=jnp.float32)
        muscle_obs = jnp.arange(100, 100 + self.muscle_dim, dtype=jnp.float32)
        goal_obs = jnp.arange(1000, 1000 + self.goal_dim, dtype=jnp.float32)
        obs = jnp.concatenate([joint_obs, muscle_obs, goal_obs])
        return obs, MockEnvState(step_count=0)

    def step(self, state: MockEnvState, action):
        """Increment observations each step."""
        step = state.step_count + 1
        joint_obs = jnp.arange(1, self.joint_dim + 1, dtype=jnp.float32) + step
        muscle_obs = jnp.arange(100, 100 + self.muscle_dim, dtype=jnp.float32) + step
        goal_obs = jnp.arange(1000, 1000 + self.goal_dim, dtype=jnp.float32) + step * 100
        obs = jnp.concatenate([joint_obs, muscle_obs, goal_obs])
        return obs, 0.0, False, False, {}, MockEnvState(step_count=step)


# =============================================================================
# Integration Tests
# =============================================================================


class TestSplitGoalIntegration:
    """Integration tests for split_goal feature."""

    def test_nstep_wrapper_splits_correctly(self):
        """Test NStepWrapper correctly separates state and goal observations."""
        env = MockEnvWithObsGroups(joint_dim=4, muscle_dim=2, goal_dim=3)
        wrapper = NStepWrapper(env, n_steps=3, split_goal=True)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        # Expected output shape: 3 * (4 + 2) + 3 = 21
        state_dim = env.joint_dim + env.muscle_dim  # 6
        expected_shape = 3 * state_dim + env.goal_dim  # 21
        assert obs.shape == (expected_shape,), f"Expected {expected_shape}, got {obs.shape}"

        # Verify state history structure (last 6 elements of first 18 should be state)
        state_hist = obs[:3 * state_dim]
        goal_part = obs[3 * state_dim:]

        # First 2 frames should be zeros, last frame should have values
        assert jnp.allclose(state_hist[:2 * state_dim], 0.0)
        # Last frame state: joint=[1,2,3,4], muscle=[100,101]
        expected_last_state = jnp.array([1, 2, 3, 4, 100, 101], dtype=jnp.float32)
        assert jnp.allclose(state_hist[2 * state_dim:], expected_last_state)

        # Goal should be from current timestep: [1000, 1001, 1002]
        expected_goal = jnp.array([1000, 1001, 1002], dtype=jnp.float32)
        assert jnp.allclose(goal_part, expected_goal)

    def test_nstep_wrapper_step_updates_correctly(self):
        """Test that step() rolls state history but updates goal from current obs."""
        env = MockEnvWithObsGroups(joint_dim=2, muscle_dim=1, goal_dim=2)
        wrapper = NStepWrapper(env, n_steps=2, split_goal=True)

        rng = jax.random.PRNGKey(0)
        obs0, state = wrapper.reset(rng)

        # Take a step
        action = jnp.zeros(1)
        obs1, _, _, _, _, state = wrapper.step(state, action)

        state_dim = 3  # 2 joint + 1 muscle
        goal_dim = 2

        # After step 1:
        # State history should be: [frame0_state, frame1_state]
        # frame0_state = [1, 2, 100] (from reset)
        # frame1_state = [2, 3, 101] (from step 1)
        expected_state_hist = jnp.array([1, 2, 100, 2, 3, 101], dtype=jnp.float32)
        assert jnp.allclose(obs1[:2 * state_dim], expected_state_hist)

        # Goal should be from step 1: [1000 + 100, 1001 + 100] = [1100, 1101]
        expected_goal = jnp.array([1100, 1101], dtype=jnp.float32)
        assert jnp.allclose(obs1[2 * state_dim:], expected_goal)

    def test_expand_obs_indices_with_split_goal(self):
        """Test expand_obs_indices_for_history correctly maps indices."""
        env = MockEnvWithObsGroups(joint_dim=4, muscle_dim=2, goal_dim=3)
        config = OmegaConf.create({
            "len_obs_history": 3,
            "split_goal": True,
        })

        # Test with full observation indices
        full_obs_ind = jnp.arange(env.obs_dim)
        expanded = expand_obs_indices_for_history(full_obs_ind, env, config)

        # Should have: 3 * 6 state indices + 3 goal indices = 21
        state_dim = env.joint_dim + env.muscle_dim
        expected_len = 3 * state_dim + env.goal_dim
        assert len(expanded) == expected_len

        # First 18 indices should be state (0-5, 6-11, 12-17)
        state_indices = expanded[:3 * state_dim]
        for i in range(3):
            frame_indices = state_indices[i * state_dim:(i + 1) * state_dim]
            expected = jnp.arange(i * state_dim, (i + 1) * state_dim)
            assert jnp.allclose(frame_indices, expected)

        # Last 3 indices should be goal (18, 19, 20)
        goal_indices = expanded[3 * state_dim:]
        expected_goal_indices = jnp.arange(3 * state_dim, 3 * state_dim + env.goal_dim)
        assert jnp.allclose(goal_indices, expected_goal_indices)

    def test_expand_obs_indices_with_subset(self):
        """Test expand_obs_indices_for_history with actor/critic subset."""
        env = MockEnvWithObsGroups(joint_dim=4, muscle_dim=2, goal_dim=3)
        config = OmegaConf.create({
            "len_obs_history": 2,
            "split_goal": True,
        })

        # Actor only uses joint (0-3) and goal (6-8)
        actor_obs_ind = jnp.array([0, 1, 2, 3, 6, 7, 8])
        expanded = expand_obs_indices_for_history(actor_obs_ind, env, config)

        state_dim = env.joint_dim + env.muscle_dim  # 6
        n_steps = 2

        # State indices in actor: [0,1,2,3] (joint only, no muscle)
        # These should be expanded to: [0,1,2,3] + [4,5,6,7] in wrapped space
        # But wait - the state_dim in wrapped space is still 6 (full state)
        # So indices 0,1,2,3 map to: frame0=[0,1,2,3], frame1=[6,7,8,9]

        # Actually let me think about this more carefully:
        # Original obs: [joint0-3, muscle0-1, goal0-2] = indices [0-3, 4-5, 6-8]
        # Wrapped obs: [state_hist(2 frames * 6), goal] = total 15 elements
        # state_hist: [frame0_state(6), frame1_state(6)]
        # In wrapped space:
        #   frame0: indices 0-5 correspond to original state indices 0-5
        #   frame1: indices 6-11 correspond to original state indices 0-5
        #   goal: indices 12-14 correspond to original goal indices 6-8

        # Actor wants original indices [0,1,2,3,6,7,8]
        # State part [0,1,2,3] -> wrapped indices [0,1,2,3, 6,7,8,9] (2 frames)
        # Goal part [6,7,8] -> wrapped indices [12,13,14]
        # Total: 4*2 + 3 = 11 elements

        assert len(expanded) == 4 * 2 + 3  # 11

    def test_wrap_env_applies_split_goal(self):
        """Test that wrap_env correctly applies split_goal configuration."""
        env = MockEnvWithObsGroups(joint_dim=4, muscle_dim=2, goal_dim=3)
        config = OmegaConf.create({
            "len_obs_history": 2,
            "split_goal": True,
            "normalize_env": False,
            "gamma": 0.99,
        })

        wrapped = wrap_env(env, config)

        # The wrapped env should have NStepWrapper applied
        # Check observation space shape
        state_dim = env.joint_dim + env.muscle_dim
        expected_obs_dim = 2 * state_dim + env.goal_dim  # 2*6 + 3 = 15
        assert wrapped.info.observation_space.shape[0] == expected_obs_dim

    def test_split_goal_vs_no_split_goal(self):
        """Compare output shapes with and without split_goal."""
        env = MockEnvWithObsGroups(joint_dim=4, muscle_dim=2, goal_dim=3)
        n_steps = 3

        # Without split_goal: stacks everything
        wrapper_no_split = NStepWrapper(env, n_steps=n_steps, split_goal=False)
        rng = jax.random.PRNGKey(0)
        obs_no_split, _ = wrapper_no_split.reset(rng)

        # With split_goal: only stacks state
        wrapper_split = NStepWrapper(env, n_steps=n_steps, split_goal=True)
        obs_split, _ = wrapper_split.reset(rng)

        # No split: 3 * 9 = 27
        assert obs_no_split.shape == (n_steps * env.obs_dim,)

        # Split: 3 * 6 + 3 = 21
        state_dim = env.joint_dim + env.muscle_dim
        assert obs_split.shape == (n_steps * state_dim + env.goal_dim,)

        # Split should be smaller (saves redundant goal stacking)
        assert obs_split.shape[0] < obs_no_split.shape[0]

    def test_jit_compatible(self):
        """Test that the full pipeline is JIT-compatible."""
        env = MockEnvWithObsGroups(joint_dim=4, muscle_dim=2, goal_dim=3)
        wrapper = NStepWrapper(env, n_steps=2, split_goal=True)

        @jax.jit
        def run_episode(rng):
            obs, state = wrapper.reset(rng)
            action = jnp.zeros(1)

            def step_fn(carry, _):
                obs, state = carry
                obs, _, _, _, _, state = wrapper.step(state, action)
                return (obs, state), obs

            (final_obs, final_state), obs_history = jax.lax.scan(
                step_fn, (obs, state), None, length=5
            )
            return final_obs, obs_history

        rng = jax.random.PRNGKey(42)
        final_obs, obs_history = run_episode(rng)

        state_dim = env.joint_dim + env.muscle_dim
        expected_shape = 2 * state_dim + env.goal_dim

        assert final_obs.shape == (expected_shape,)
        assert obs_history.shape == (5, expected_shape)


class TestSplitGoalEdgeCases:
    """Edge case tests for split_goal feature."""

    def test_non_contiguous_goal_indices(self):
        """Test with goal indices that are not contiguous."""
        # Create env where goal is interleaved with state
        # This tests that index-based approach works regardless of layout

        class NonContiguousEnv:
            def __init__(self):
                # Obs layout: [state0, goal0, state1, goal1, state2]
                # indices:      0       1       2       3       4
                self.obs_dim = 5
                self.obs_container = MockObsContainer({
                    "state": np.array([0, 2, 4]),
                    "goal": np.array([1, 3]),
                })
                self.info = MockMDPInfo(self.obs_dim)
                self.mjx_env = False

            def reset(self, rng_key):
                obs = jnp.array([10, 100, 20, 200, 30], dtype=jnp.float32)
                return obs, MockEnvState()

            def step(self, state, action):
                obs = jnp.array([11, 101, 21, 201, 31], dtype=jnp.float32)
                return obs, 0.0, False, False, {}, MockEnvState(step_count=1)

        env = NonContiguousEnv()
        wrapper = NStepWrapper(env, n_steps=2, split_goal=True)

        rng = jax.random.PRNGKey(0)
        obs, _ = wrapper.reset(rng)

        # state_dim = 3, goal_dim = 2
        # Output: 2 * 3 + 2 = 8
        assert obs.shape == (8,)

        # State history: [zeros(3), state_from_reset]
        # state_from_reset extracted from indices [0,2,4] = [10, 20, 30]
        expected_state = jnp.array([0, 0, 0, 10, 20, 30], dtype=jnp.float32)
        assert jnp.allclose(obs[:6], expected_state)

        # Goal from indices [1,3] = [100, 200]
        expected_goal = jnp.array([100, 200], dtype=jnp.float32)
        assert jnp.allclose(obs[6:], expected_goal)

    def test_single_frame_history(self):
        """Test with n_steps=1 (no actual history stacking)."""
        env = MockEnvWithObsGroups(joint_dim=3, muscle_dim=2, goal_dim=2)
        wrapper = NStepWrapper(env, n_steps=1, split_goal=True)

        rng = jax.random.PRNGKey(0)
        obs, _ = wrapper.reset(rng)

        # Output: 1 * 5 + 2 = 7 (same as original minus nothing)
        state_dim = env.joint_dim + env.muscle_dim
        assert obs.shape == (state_dim + env.goal_dim,)

        # Should be equivalent to original obs with reordering
        # [joint(3), muscle(2), goal(2)] -> [state(5), goal(2)]
        expected = jnp.array([1, 2, 3, 100, 101, 1000, 1001], dtype=jnp.float32)
        assert jnp.allclose(obs, expected)
