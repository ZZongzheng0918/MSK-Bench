"""Integration tests for fullbody terminal state handlers.

These tests use real MyoFullBody environments with AMASS trajectory data.

Run with: pytest tests/unit/test_enhanced_fullbody_terminal_handler_integration.py -v
"""

import mujoco
import numpy as np
import pytest

from loco_mujoco.task_factories import AMASSDatasetConf, ImitationFactory
from musclemimic.core.terminal_state_handler.enhanced_fullbody import (
    EnhancedFullBodyTerminalStateHandler,
    MeanRelativeSiteDeviationTerminalStateHandler,
    MeanSiteDeviationTerminalStateHandler,
)

pytestmark = pytest.mark.integration


def set_threshold(carry, value: float):
    """Return a new carry with the updated termination_threshold."""
    import jax.numpy as jnp

    return carry.replace(termination_threshold=jnp.asarray(value, dtype=jnp.float32))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def env_with_trajectory():
    """Create MyoFullBody environment with actual trajectory data."""
    env_params = {
        "env_name": "MyoFullBody",
        "headless": True,
        "goal_type": "GoalTrajMimic",
        "goal_params": {
            "sites_for_mimic": [
                "pelvis_mimic",
                "upper_body_mimic",
                "head_mimic",
                "left_shoulder_mimic",
                "left_elbow_mimic",
                "left_hand_mimic",
                "right_shoulder_mimic",
                "right_elbow_mimic",
                "right_hand_mimic",
                "left_hip_mimic",
                "left_knee_mimic",
                "left_ankle_mimic",
                "left_toes_mimic",
                "right_hip_mimic",
                "right_knee_mimic",
                "right_ankle_mimic",
                "right_toes_mimic",
            ],
            "visualize_goal": False,
        },
        "terminal_state_type": "NoTerminalStateHandler",
    }

    task_params = {"amass_dataset_conf": AMASSDatasetConf(["KIT/3/walking_medium04_poses"])}

    try:
        env = ImitationFactory.make(**env_params, **task_params)
    except ImportError as exc:
        if "Optional smpl dependencies not installed" in str(exc):
            pytest.skip(str(exc))
        raise
    yield env


# =============================================================================
# EnhancedFullBodyTerminalStateHandler Integration Tests
# =============================================================================


class TestEnhancedHandlerIntegration:
    """Integration tests for EnhancedFullBodyTerminalStateHandler."""

    def test_perfect_tracking_no_termination(self, env_with_trajectory):
        """Perfect tracking should not trigger termination."""
        env = env_with_trajectory

        handler = EnhancedFullBodyTerminalStateHandler(
            env,
            root_height_healthy_range=[0.3, 2.5],
            ankle_deviation=0.5,
            root_deviation=0.5,
            ankle_sites=["left_ankle_mimic", "right_ankle_mimic"],
            root_site="pelvis_mimic",
            enable_site_check=True,
            site_deviation_mode="mean",
        )

        obs = env.reset()
        carry = env._additional_carry
        actual_step = carry.traj_state.subtraj_step_no

        # Set perfect tracking
        traj_data_ref = env.th.traj.data.get(0, actual_step, np)
        env._data.qpos[:] = traj_data_ref.qpos.copy()
        env._data.qvel[:] = traj_data_ref.qvel.copy()
        mujoco.mj_forward(env._model, env._data)

        is_terminal, _ = handler.is_absorbing(env, obs, {}, env._data, carry)
        assert not is_terminal, "Perfect tracking should not trigger termination"

    def test_height_violation_terminates(self, env_with_trajectory):
        """Height below threshold should trigger termination."""
        env = env_with_trajectory

        handler = EnhancedFullBodyTerminalStateHandler(
            env,
            root_height_healthy_range=[0.6, 2.5],
            enable_site_check=False,
        )

        obs = env.reset()
        carry = env._additional_carry
        actual_step = carry.traj_state.subtraj_step_no

        traj_data_ref = env.th.traj.data.get(0, actual_step, np)
        env._data.qpos[:] = traj_data_ref.qpos.copy()
        env._data.qpos[2] = 0.4  # Below 0.6m threshold
        mujoco.mj_forward(env._model, env._data)

        is_terminal, _ = handler.is_absorbing(env, obs, {}, env._data, carry)
        assert is_terminal, "Low height should trigger termination"

    def test_site_deviation_terminates(self, env_with_trajectory):
        """Large site deviation should trigger termination."""
        env = env_with_trajectory

        handler = EnhancedFullBodyTerminalStateHandler(
            env,
            root_height_healthy_range=[0.3, 2.5],
            ankle_deviation=0.2,
            root_deviation=0.2,
            ankle_sites=["left_ankle_mimic", "right_ankle_mimic"],
            root_site="pelvis_mimic",
            enable_site_check=True,
            site_deviation_mode="mean",
        )

        obs = env.reset()
        carry = env._additional_carry
        actual_step = carry.traj_state.subtraj_step_no

        traj_data_ref = env.th.traj.data.get(0, actual_step, np)
        env._data.qpos[:] = traj_data_ref.qpos.copy()
        env._data.qpos[0] += 0.5  # Large displacement
        mujoco.mj_forward(env._model, env._data)

        is_terminal, _ = handler.is_absorbing(env, obs, {}, env._data, carry)
        assert is_terminal, "Large site deviation should trigger termination"


# =============================================================================
# MeanSiteDeviationTerminalStateHandler Integration Tests
# =============================================================================


class TestMeanAbsoluteHandlerIntegration:
    """Integration tests for MeanSiteDeviationTerminalStateHandler."""

    def test_perfect_tracking_no_termination(self, env_with_trajectory):
        """Perfect tracking should not trigger termination."""
        env = env_with_trajectory

        handler = MeanSiteDeviationTerminalStateHandler(env, mean_site_deviation_threshold=0.3, enable_site_check=True)

        obs = env.reset()
        carry = env._additional_carry
        carry = set_threshold(carry, 0.3)
        actual_step = carry.traj_state.subtraj_step_no

        traj_data_ref = env.th.traj.data.get(0, actual_step, np)
        env._data.qpos[:] = traj_data_ref.qpos.copy()
        env._data.qvel[:] = traj_data_ref.qvel.copy()

        # Apply root XY offset to center the trajectory (matching handler behavior)
        init_traj_data = env.th.get_init_traj_data(carry, np)
        env._data.qpos[0] -= init_traj_data.qpos[0]
        env._data.qpos[1] -= init_traj_data.qpos[1]

        mujoco.mj_forward(env._model, env._data)

        is_terminal, _ = handler.is_absorbing(env, obs, {}, env._data, carry)
        assert not is_terminal, "Perfect tracking should not trigger termination"

    def test_large_deviation_terminates(self, env_with_trajectory):
        """Large mean deviation should trigger termination."""
        env = env_with_trajectory

        handler = MeanSiteDeviationTerminalStateHandler(env, mean_site_deviation_threshold=0.2, enable_site_check=True)

        obs = env.reset()
        carry = env._additional_carry
        carry = set_threshold(carry, 0.2)
        actual_step = carry.traj_state.subtraj_step_no

        traj_data_ref = env.th.traj.data.get(0, actual_step, np)
        env._data.qpos[:] = traj_data_ref.qpos.copy()
        env._data.qpos[0] += 0.5
        mujoco.mj_forward(env._model, env._data)

        is_terminal, _ = handler.is_absorbing(env, obs, {}, env._data, carry)
        assert is_terminal, "Large mean deviation should trigger termination"

    def test_at_threshold_no_termination(self, env_with_trajectory):
        """Deviation exactly at threshold should NOT terminate."""
        env = env_with_trajectory

        handler = MeanSiteDeviationTerminalStateHandler(env, mean_site_deviation_threshold=0.2, enable_site_check=True)

        obs = env.reset()
        carry = env._additional_carry
        carry = set_threshold(carry, 0.2)
        actual_step = carry.traj_state.subtraj_step_no

        traj_data_ref = env.th.traj.data.get(0, actual_step, np)
        env._data.qpos[:] = traj_data_ref.qpos.copy()
        env._data.qvel[:] = traj_data_ref.qvel.copy()

        # Apply root XY offset to center the trajectory (matching handler behavior)
        init_traj_data = env.th.get_init_traj_data(carry, np)
        env._data.qpos[0] -= init_traj_data.qpos[0]
        env._data.qpos[1] -= init_traj_data.qpos[1]

        # Add a small deviation
        env._data.qpos[0] += 0.2
        mujoco.mj_forward(env._model, env._data)

        # Compute actual deviation and set threshold to match
        site_mapping = env._goal._rel_site_ids
        current_mapped_sites = env._data.site_xpos[site_mapping]
        if env._goal._site_mapper.requires_mapping:
            traj_indices = env._goal._site_mapper.model_ids_to_traj_indices(site_mapping)
            ref_mapped_sites = traj_data_ref.site_xpos[traj_indices]
        else:
            ref_mapped_sites = traj_data_ref.site_xpos

        # Apply same offset to reference sites as the handler does
        root_xy = init_traj_data.qpos[:2]
        offset = np.concatenate([root_xy, np.zeros(1)])
        ref_mapped_sites = ref_mapped_sites - offset

        site_deviations = np.linalg.norm(current_mapped_sites - ref_mapped_sites, axis=-1)
        # Add small epsilon to avoid floating-point precision issues
        carry = set_threshold(carry, float(np.mean(site_deviations)) + 1e-6)

        is_terminal, _ = handler.is_absorbing(env, obs, {}, env._data, carry)
        assert not is_terminal, "Deviation at threshold should not terminate"


# =============================================================================
# MeanRelativeSiteDeviationTerminalStateHandler Integration Tests
# =============================================================================


class TestMeanRelativeHandlerIntegration:
    """Integration tests for MeanRelativeSiteDeviationTerminalStateHandler."""

    def test_perfect_tracking_no_termination(self, env_with_trajectory):
        """Perfect tracking should not trigger termination."""
        env = env_with_trajectory

        handler = MeanRelativeSiteDeviationTerminalStateHandler(
            env, mean_site_deviation_threshold=0.3, enable_site_check=True
        )

        obs = env.reset()
        carry = env._additional_carry
        carry = set_threshold(carry, 0.3)
        actual_step = carry.traj_state.subtraj_step_no

        traj_data_ref = env.th.traj.data.get(0, actual_step, np)
        env._data.qpos[:] = traj_data_ref.qpos.copy()
        env._data.qvel[:] = traj_data_ref.qvel.copy()
        mujoco.mj_forward(env._model, env._data)

        is_terminal, _ = handler.is_absorbing(env, obs, {}, env._data, carry)
        assert not is_terminal, "Perfect tracking should not trigger termination"

    def test_configuration_change_terminates(self, env_with_trajectory):
        """Body configuration change should trigger termination."""
        env = env_with_trajectory

        handler = MeanRelativeSiteDeviationTerminalStateHandler(
            env, mean_site_deviation_threshold=0.2, enable_site_check=True
        )

        obs = env.reset()
        carry = env._additional_carry
        carry = set_threshold(carry, 0.2)
        actual_step = carry.traj_state.subtraj_step_no

        traj_data_ref = env.th.traj.data.get(0, actual_step, np)
        env._data.qpos[:] = traj_data_ref.qpos.copy()
        # Large configuration changes
        env._data.qpos[7] += 1.0
        env._data.qpos[8] += 1.0
        env._data.qpos[13] += 1.0
        mujoco.mj_forward(env._model, env._data)

        is_terminal, _ = handler.is_absorbing(env, obs, {}, env._data, carry)
        assert is_terminal, "Body configuration change should trigger termination"


# =============================================================================
# Comparison Tests
# =============================================================================


class TestHandlerComparisonIntegration:
    """Integration tests comparing handler behaviors."""

    def test_relative_vs_absolute_on_translation(self, env_with_trajectory):
        """Relative handler ignores translation, absolute does not."""
        env = env_with_trajectory

        handler_abs = MeanSiteDeviationTerminalStateHandler(
            env, mean_site_deviation_threshold=0.2, enable_site_check=True
        )
        handler_rel = MeanRelativeSiteDeviationTerminalStateHandler(
            env, mean_site_deviation_threshold=0.2, enable_site_check=True
        )

        obs = env.reset()
        carry = env._additional_carry
        carry = set_threshold(carry, 0.2)
        actual_step = carry.traj_state.subtraj_step_no

        traj_data_ref = env.th.traj.data.get(0, actual_step, np)
        env._data.qpos[:] = traj_data_ref.qpos.copy()
        env._data.qpos[0] += 0.5  # Translate X
        env._data.qpos[1] += 0.3  # Translate Y
        mujoco.mj_forward(env._model, env._data)

        is_terminal_abs, _ = handler_abs.is_absorbing(env, obs, {}, env._data, carry)
        is_terminal_rel, _ = handler_rel.is_absorbing(env, obs, {}, env._data, carry)

        assert is_terminal_abs, "Absolute handler should terminate on translation"
        assert not is_terminal_rel, "Relative handler should NOT terminate on pure translation"

    def test_site_check_disabled_bypasses_checks(self, env_with_trajectory):
        """Both handlers should not terminate when site checks are disabled."""
        env = env_with_trajectory

        handler_abs = MeanSiteDeviationTerminalStateHandler(
            env, mean_site_deviation_threshold=0.2, enable_site_check=False
        )
        handler_rel = MeanRelativeSiteDeviationTerminalStateHandler(
            env, mean_site_deviation_threshold=0.2, enable_site_check=False
        )

        obs = env.reset()
        carry = env._additional_carry
        carry = set_threshold(carry, 0.2)
        actual_step = carry.traj_state.subtraj_step_no

        traj_data_ref = env.th.traj.data.get(0, actual_step, np)
        env._data.qpos[:] = traj_data_ref.qpos.copy()
        env._data.qvel[:] = traj_data_ref.qvel.copy()
        env._data.qpos[2] = 0.05  # Large height change
        mujoco.mj_forward(env._model, env._data)

        is_terminal_abs, _ = handler_abs.is_absorbing(env, obs, {}, env._data, carry)
        is_terminal_rel, _ = handler_rel.is_absorbing(env, obs, {}, env._data, carry)

        assert not is_terminal_abs, "Absolute handler should not terminate with site checks disabled"
        assert not is_terminal_rel, "Relative handler should not terminate with site checks disabled"


# =============================================================================
# Data Consistency Tests
# =============================================================================


class TestTrajectoryDataConsistency:
    """Tests for trajectory data accessor consistency."""

    def test_trajectory_accessors_return_consistent_data(self, env_with_trajectory):
        """Trajectory accessors should return consistent data for the same step."""
        env = env_with_trajectory

        env.reset()
        carry = env._additional_carry
        actual_step = carry.traj_state.subtraj_step_no

        traj_data_direct = env.th.traj.data.get(0, actual_step, np)
        traj_data_via_carry = env.th.get_current_traj_data(carry, np)

        np.testing.assert_allclose(traj_data_direct.qpos, traj_data_via_carry.qpos)
        np.testing.assert_allclose(traj_data_direct.qvel, traj_data_via_carry.qvel)
        np.testing.assert_allclose(traj_data_direct.site_xpos, traj_data_via_carry.site_xpos)
