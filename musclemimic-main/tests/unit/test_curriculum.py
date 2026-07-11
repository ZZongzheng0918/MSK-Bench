"""Unit tests for adaptive termination curriculum learning."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from musclemimic.algorithms.common.curriculum import (
    CurriculumParams,
    compute_early_termination_stats,
    create_curriculum_params,
    create_curriculum_state,
    update_curriculum_state,
    validate_curriculum_config,
)


def _to_float_tree(tree):
    return jax.tree_util.tree_map(lambda x: float(x), tree)


# =============================================================================
# Config Validation Tests
# =============================================================================


class TestCurriculumConfigValidation:
    """Tests for curriculum configuration validation."""

    def test_validate_valid_config(self):
        """Valid config should pass validation."""
        cfg = {
            "enabled": True,
            "init_threshold": 0.5,
            "init_ema_val": 0.15,
            "low_band": 0.10,
            "high_band": 0.20,
            "adjust_factor": 0.95,
            "consecutive_k": 5,
            "min_threshold": 0.1,
            "ema_alpha": 0.1,
        }
        # Should not raise
        validate_curriculum_config(cfg)

    def test_validate_adjust_factor_out_of_range(self):
        """adjust_factor must be in (0, 1)."""
        cfg = {
            "init_threshold": 0.5,
            "adjust_factor": 1.5,  # Invalid
            "min_threshold": 0.1,
            "low_band": 0.10,
            "high_band": 0.20,
            "ema_alpha": 0.1,
            "consecutive_k": 5,
        }
        with pytest.raises(AssertionError, match="adjust_factor"):
            validate_curriculum_config(cfg)

    def test_validate_adjust_factor_one(self):
        """adjust_factor cannot be 1 (no-op update)."""
        cfg = {
            "init_threshold": 0.5,
            "adjust_factor": 1.0,  # Invalid (must be < 1)
            "min_threshold": 0.1,
            "low_band": 0.10,
            "high_band": 0.20,
            "ema_alpha": 0.1,
            "consecutive_k": 5,
        }
        with pytest.raises(AssertionError, match="adjust_factor"):
            validate_curriculum_config(cfg)

    def test_validate_init_threshold_missing(self):
        """init_threshold is required."""
        cfg = {
            "adjust_factor": 0.95,
            "min_threshold": 0.1,
            "low_band": 0.10,
            "high_band": 0.20,
            "ema_alpha": 0.1,
            "consecutive_k": 5,
        }
        with pytest.raises(AssertionError, match="init_threshold"):
            validate_curriculum_config(cfg)

    def test_validate_init_threshold_below_min(self):
        """init_threshold must be >= min_threshold."""
        cfg = {
            "init_threshold": 0.05,  # < min_threshold
            "adjust_factor": 0.95,
            "min_threshold": 0.1,
            "low_band": 0.10,
            "high_band": 0.20,
            "ema_alpha": 0.1,
            "consecutive_k": 5,
        }
        with pytest.raises(AssertionError, match="init_threshold must be >= min_threshold"):
            validate_curriculum_config(cfg)

    def test_validate_low_band_greater_than_high_band(self):
        """low_band must be less than high_band."""
        cfg = {
            "init_threshold": 0.5,
            "adjust_factor": 0.95,
            "min_threshold": 0.1,
            "low_band": 0.25,  # > high_band
            "high_band": 0.20,
            "ema_alpha": 0.1,
            "consecutive_k": 5,
        }
        with pytest.raises(AssertionError, match="low_band must be less than high_band"):
            validate_curriculum_config(cfg)

    def test_validate_low_band_equal_high_band(self):
        """low_band cannot equal high_band."""
        cfg = {
            "init_threshold": 0.5,
            "adjust_factor": 0.95,
            "min_threshold": 0.1,
            "low_band": 0.20,
            "high_band": 0.20,  # Equal
            "ema_alpha": 0.1,
            "consecutive_k": 5,
        }
        with pytest.raises(AssertionError, match="low_band must be less than high_band"):
            validate_curriculum_config(cfg)

    def test_validate_ema_alpha_zero(self):
        """ema_alpha must be in (0, 1]."""
        cfg = {
            "init_threshold": 0.5,
            "adjust_factor": 0.95,
            "min_threshold": 0.1,
            "low_band": 0.10,
            "high_band": 0.20,
            "ema_alpha": 0.0,  # Invalid (must be > 0)
            "consecutive_k": 5,
        }
        with pytest.raises(AssertionError, match="ema_alpha"):
            validate_curriculum_config(cfg)

    def test_validate_ema_alpha_above_one(self):
        """ema_alpha cannot exceed 1."""
        cfg = {
            "init_threshold": 0.5,
            "adjust_factor": 0.95,
            "min_threshold": 0.1,
            "low_band": 0.10,
            "high_band": 0.20,
            "ema_alpha": 1.1,
            "consecutive_k": 5,
        }
        with pytest.raises(AssertionError, match="ema_alpha"):
            validate_curriculum_config(cfg)

    def test_validate_consecutive_k_zero(self):
        """consecutive_k must be >= 1."""
        cfg = {
            "init_threshold": 0.5,
            "adjust_factor": 0.95,
            "min_threshold": 0.1,
            "low_band": 0.10,
            "high_band": 0.20,
            "ema_alpha": 0.1,
            "consecutive_k": 0,  # Invalid
        }
        with pytest.raises(AssertionError, match="consecutive_k"):
            validate_curriculum_config(cfg)


# =============================================================================
# Early Termination Stats Tests
# =============================================================================


class TestEarlyTerminationStats:
    """Tests for early termination rate computation."""

    def test_compute_stats_float32_dtype(self):
        """Verify output dtypes are float32."""
        done = jnp.array([True, False, True, False])
        absorbing = jnp.array([True, False, False, False])

        early_count, rate = compute_early_termination_stats(done, absorbing)

        assert early_count.dtype == jnp.float32
        assert rate.dtype == jnp.float32
        assert float(early_count) == pytest.approx(1.0)
        assert float(rate) == pytest.approx(0.5)

    def test_compute_stats_no_done(self):
        """Test early termination stats when no episodes are done."""
        done = jnp.array([False, False, False])
        absorbing = jnp.array([True, True, True])

        early_count, rate = compute_early_termination_stats(done, absorbing)

        assert float(early_count) == pytest.approx(0.0)
        assert float(rate) == pytest.approx(0.0)  # No division by zero

    def test_compute_stats_all_early(self):
        """Test when all completed episodes are early terminated."""
        done = jnp.array([True, True, True])
        absorbing = jnp.array([True, True, True])

        early_count, rate = compute_early_termination_stats(done, absorbing)

        assert float(early_count) == pytest.approx(3.0)
        assert float(rate) == pytest.approx(1.0)

    def test_compute_stats_none_early(self):
        """Test when no completed episodes are early terminated."""
        done = jnp.array([True, True, True])
        absorbing = jnp.array([False, False, False])

        early_count, rate = compute_early_termination_stats(done, absorbing)

        assert float(early_count) == pytest.approx(0.0)
        assert float(rate) == pytest.approx(0.0)


# =============================================================================
# Curriculum Creation Tests
# =============================================================================


class TestCurriculumCreation:
    """Tests for curriculum params and state creation."""

    def test_create_curriculum_params(self):
        """Test CurriculumParams creation from config dict."""
        cfg = {
            "init_ema_val": 0.15,
            "low_band": 0.10,
            "high_band": 0.20,
            "adjust_factor": 0.95,
            "consecutive_k": 5,
            "min_threshold": 0.1,
            "ema_alpha": 0.1,
        }
        params = create_curriculum_params(cfg)

        assert float(params.init_ema_val) == pytest.approx(0.15)
        assert float(params.low_band) == pytest.approx(0.10)
        assert float(params.high_band) == pytest.approx(0.20)
        assert float(params.adjust_factor) == pytest.approx(0.95)
        assert int(params.consecutive_k) == 5
        assert float(params.min_threshold) == pytest.approx(0.1)
        assert float(params.ema_alpha) == pytest.approx(0.1)

        # Verify dtypes
        assert params.init_ema_val.dtype == jnp.float32
        assert params.consecutive_k.dtype == jnp.int32

    def test_create_curriculum_state(self):
        """Test CurriculumState creation."""
        state = create_curriculum_state(init_threshold=0.5, init_ema_val=0.15)

        assert float(state.current_threshold) == pytest.approx(0.5)
        assert float(state.ema_rate) == pytest.approx(0.15)  # Initialized to target
        assert int(state.consecutive_count) == 0
        assert int(state.last_direction) == 0
        assert float(state.initial_threshold) == pytest.approx(0.5)  # Ratchet

        # Verify dtypes
        assert state.current_threshold.dtype == jnp.float32
        assert state.ema_rate.dtype == jnp.float32
        assert state.consecutive_count.dtype == jnp.int32
        assert state.last_direction.dtype == jnp.int32
        assert state.initial_threshold.dtype == jnp.float32


# =============================================================================
# Curriculum Update Tests
# =============================================================================


class TestCurriculumUpdate:
    """Tests for curriculum state update logic."""

    @pytest.fixture
    def default_params(self):
        """Default curriculum params for testing."""
        return CurriculumParams(
            init_ema_val=jnp.asarray(0.15, dtype=jnp.float32),
            low_band=jnp.asarray(0.10, dtype=jnp.float32),
            high_band=jnp.asarray(0.20, dtype=jnp.float32),
            adjust_factor=jnp.asarray(0.9, dtype=jnp.float32),  # 10% change
            consecutive_k=jnp.asarray(3, dtype=jnp.int32),
            min_threshold=jnp.asarray(0.1, dtype=jnp.float32),
            ema_alpha=jnp.asarray(1.0, dtype=jnp.float32),  # No smoothing for test
        )

    def test_low_rate_decreases_threshold(self, default_params):
        """Rate below low_band should decrease threshold after consecutive_k updates."""
        state = create_curriculum_state(init_threshold=0.5, init_ema_val=0.15)
        low_rate = jnp.asarray(0.05, dtype=jnp.float32)  # < 0.10

        # First update: direction=-1, consecutive=1
        state, direction, adjusted = update_curriculum_state(state, low_rate, default_params)
        assert int(direction) == -1
        assert float(adjusted) == pytest.approx(0.0)  # Not yet adjusted
        assert float(state.current_threshold) == pytest.approx(0.5)  # Unchanged

        # Second update: direction=-1, consecutive=2
        state, direction, adjusted = update_curriculum_state(state, low_rate, default_params)
        assert int(state.consecutive_count) == 2
        assert float(adjusted) == pytest.approx(0.0)

        # Third update: direction=-1, consecutive=3 -> adjust
        state, direction, adjusted = update_curriculum_state(state, low_rate, default_params)
        assert float(adjusted) == pytest.approx(1.0)  # Adjusted
        assert float(state.current_threshold) == pytest.approx(0.5 * 0.9)  # Decreased

    def test_high_rate_increases_threshold_capped_by_ratchet(self, default_params):
        """Rate above high_band triggers increase but ratchet caps at initial_threshold."""
        state = create_curriculum_state(init_threshold=0.5, init_ema_val=0.15)
        high_rate = jnp.asarray(0.30, dtype=jnp.float32)  # > 0.20

        # Three consecutive high rates
        for _ in range(2):
            state, direction, adjusted = update_curriculum_state(state, high_rate, default_params)
            assert int(direction) == 1
            assert float(adjusted) == pytest.approx(0.0)

        # Third update triggers adjustment but ratchet prevents increase above initial
        state, direction, adjusted = update_curriculum_state(state, high_rate, default_params)
        assert float(adjusted) == pytest.approx(1.0)
        # Ratchet: threshold cannot exceed initial_threshold (0.5)
        assert float(state.current_threshold) == pytest.approx(0.5)

    def test_neutral_rate_no_change(self, default_params):
        """Rate within [low_band, high_band] should not change threshold."""
        state = create_curriculum_state(init_threshold=0.5, init_ema_val=0.15)
        neutral_rate = jnp.asarray(0.15, dtype=jnp.float32)  # Within bands

        # Multiple neutral updates
        for _ in range(10):
            old_threshold = float(state.current_threshold)
            state, direction, adjusted = update_curriculum_state(state, neutral_rate, default_params)
            assert int(direction) == 0
            assert float(adjusted) == pytest.approx(0.0)
            assert float(state.current_threshold) == pytest.approx(old_threshold)

    def test_boundary_rates_are_neutral(self, default_params):
        """Rates equal to low/high band should be treated as neutral."""
        state = create_curriculum_state(init_threshold=0.5, init_ema_val=0.15)
        low_rate = jnp.asarray(float(default_params.low_band), dtype=jnp.float32)
        high_rate = jnp.asarray(float(default_params.high_band), dtype=jnp.float32)

        state, direction, adjusted = update_curriculum_state(state, low_rate, default_params)
        assert int(direction) == 0
        assert float(adjusted) == pytest.approx(0.0)
        assert int(state.consecutive_count) == 0

        state, direction, adjusted = update_curriculum_state(state, high_rate, default_params)
        assert int(direction) == 0
        assert float(adjusted) == pytest.approx(0.0)
        assert int(state.consecutive_count) == 0

    def test_direction_change_resets_consecutive(self, default_params):
        """Changing direction should reset consecutive counter."""
        state = create_curriculum_state(init_threshold=0.5, init_ema_val=0.15)

        # Two low rate updates
        low_rate = jnp.asarray(0.05, dtype=jnp.float32)
        state, _, _ = update_curriculum_state(state, low_rate, default_params)
        state, _, _ = update_curriculum_state(state, low_rate, default_params)
        assert int(state.consecutive_count) == 2
        assert int(state.last_direction) == -1

        # Now high rate - should reset
        high_rate = jnp.asarray(0.30, dtype=jnp.float32)
        state, direction, _ = update_curriculum_state(state, high_rate, default_params)
        assert int(direction) == 1
        assert int(state.consecutive_count) == 1  # Reset to 1
        assert int(state.last_direction) == 1

    def test_neutral_resets_consecutive(self, default_params):
        """Entering neutral zone should reset consecutive counter."""
        state = create_curriculum_state(init_threshold=0.5, init_ema_val=0.15)

        # Two low rate updates
        low_rate = jnp.asarray(0.05, dtype=jnp.float32)
        state, _, _ = update_curriculum_state(state, low_rate, default_params)
        state, _, _ = update_curriculum_state(state, low_rate, default_params)
        assert int(state.consecutive_count) == 2

        # Now neutral rate - should reset
        neutral_rate = jnp.asarray(0.15, dtype=jnp.float32)
        state, direction, _ = update_curriculum_state(state, neutral_rate, default_params)
        assert int(direction) == 0
        assert int(state.consecutive_count) == 0
        assert int(state.last_direction) == 0

    def test_threshold_clipped_to_min(self, default_params):
        """Threshold should not go below min_threshold."""
        state = create_curriculum_state(init_threshold=0.15, init_ema_val=0.15)
        low_rate = jnp.asarray(0.05, dtype=jnp.float32)

        # Keep decreasing
        for _ in range(10):
            for _ in range(3):  # consecutive_k = 3
                state, _, _ = update_curriculum_state(state, low_rate, default_params)

        # Should be clipped at min_threshold=0.1
        assert float(state.current_threshold) == pytest.approx(0.1)

    def test_threshold_clipped_to_initial_by_ratchet(self, default_params):
        """Threshold should not go above initial_threshold (ratchet)."""
        state = create_curriculum_state(init_threshold=0.9, init_ema_val=0.15)
        high_rate = jnp.asarray(0.30, dtype=jnp.float32)

        # Keep trying to increase
        for _ in range(10):
            for _ in range(3):  # consecutive_k = 3
                state, _, _ = update_curriculum_state(state, high_rate, default_params)

        # Ratchet: should stay at initial_threshold=0.9
        assert float(state.current_threshold) == pytest.approx(0.9)

    def test_ratchet_allows_recovery_up_to_initial(self, default_params):
        """After decrease, threshold can recover up to but not beyond initial."""
        state = create_curriculum_state(init_threshold=0.5, init_ema_val=0.15)
        low_rate = jnp.asarray(0.05, dtype=jnp.float32)
        high_rate = jnp.asarray(0.30, dtype=jnp.float32)

        # First decrease the threshold
        for _ in range(3):  # consecutive_k = 3
            state, _, _ = update_curriculum_state(state, low_rate, default_params)
        decreased_threshold = float(state.current_threshold)
        assert decreased_threshold == pytest.approx(0.5 * 0.9)  # 0.45

        # Now try to increase - should recover toward initial
        for _ in range(3):
            state, _, _ = update_curriculum_state(state, high_rate, default_params)
        assert float(state.current_threshold) == pytest.approx(0.45 / 0.9)  # 0.5

        # Further increases should be capped at initial_threshold
        for _ in range(3):
            state, _, _ = update_curriculum_state(state, high_rate, default_params)
        assert float(state.current_threshold) == pytest.approx(0.5)  # Capped

    def test_consecutive_k_one_adjusts_immediately(self):
        """consecutive_k=1 should adjust on first out-of-band update."""
        params = CurriculumParams(
            init_ema_val=jnp.asarray(0.15, dtype=jnp.float32),
            low_band=jnp.asarray(0.10, dtype=jnp.float32),
            high_band=jnp.asarray(0.20, dtype=jnp.float32),
            adjust_factor=jnp.asarray(0.9, dtype=jnp.float32),
            consecutive_k=jnp.asarray(1, dtype=jnp.int32),
            min_threshold=jnp.asarray(0.1, dtype=jnp.float32),
            ema_alpha=jnp.asarray(1.0, dtype=jnp.float32),
        )
        state = create_curriculum_state(init_threshold=0.5, init_ema_val=0.15)
        low_rate = jnp.asarray(0.05, dtype=jnp.float32)

        state, direction, adjusted = update_curriculum_state(state, low_rate, params)
        assert int(direction) == -1
        assert float(adjusted) == pytest.approx(1.0)
        assert float(state.current_threshold) == pytest.approx(0.5 * 0.9)
        assert int(state.consecutive_count) == 0

    def test_ema_smoothing(self):
        """Test EMA smoothing of termination rate."""
        params = CurriculumParams(
            init_ema_val=jnp.asarray(0.15, dtype=jnp.float32),
            low_band=jnp.asarray(0.10, dtype=jnp.float32),
            high_band=jnp.asarray(0.20, dtype=jnp.float32),
            adjust_factor=jnp.asarray(0.9, dtype=jnp.float32),
            consecutive_k=jnp.asarray(3, dtype=jnp.int32),
            min_threshold=jnp.asarray(0.1, dtype=jnp.float32),
            ema_alpha=jnp.asarray(0.1, dtype=jnp.float32),  # 10% weight to new
        )
        state = create_curriculum_state(init_threshold=0.5, init_ema_val=0.15)

        # Initial ema_rate = 0.15, new observation = 0.25
        new_rate = jnp.asarray(0.25, dtype=jnp.float32)
        state, _, _ = update_curriculum_state(state, new_rate, params)

        # EMA: 0.9 * 0.15 + 0.1 * 0.25 = 0.135 + 0.025 = 0.16
        expected_ema = 0.9 * 0.15 + 0.1 * 0.25
        assert float(state.ema_rate) == pytest.approx(expected_ema)

    def test_direction_uses_ema(self):
        """Direction should be computed from EMA, not raw rate."""
        params = CurriculumParams(
            init_ema_val=jnp.asarray(0.15, dtype=jnp.float32),
            low_band=jnp.asarray(0.10, dtype=jnp.float32),
            high_band=jnp.asarray(0.20, dtype=jnp.float32),
            adjust_factor=jnp.asarray(0.9, dtype=jnp.float32),
            consecutive_k=jnp.asarray(3, dtype=jnp.int32),
            min_threshold=jnp.asarray(0.1, dtype=jnp.float32),
            ema_alpha=jnp.asarray(0.01, dtype=jnp.float32),
        )
        state = create_curriculum_state(init_threshold=0.5, init_ema_val=0.15)
        high_rate = jnp.asarray(1.0, dtype=jnp.float32)

        state, direction, adjusted = update_curriculum_state(state, high_rate, params)
        assert int(direction) == 0
        assert float(adjusted) == pytest.approx(0.0)
        assert int(state.consecutive_count) == 0

    def test_consecutive_resets_after_adjustment(self, default_params):
        """Consecutive counter should reset after threshold adjustment."""
        state = create_curriculum_state(init_threshold=0.5, init_ema_val=0.15)
        low_rate = jnp.asarray(0.05, dtype=jnp.float32)

        # Three consecutive to trigger adjustment
        for _ in range(3):
            state, _, _ = update_curriculum_state(state, low_rate, default_params)

        # After adjustment, consecutive should reset
        assert int(state.consecutive_count) == 0

        # Next update starts fresh
        state, _, _ = update_curriculum_state(state, low_rate, default_params)
        assert int(state.consecutive_count) == 1


# =============================================================================
# JIT Compatibility Tests
# =============================================================================


class TestCurriculumJITCompatibility:
    """Tests to verify JIT compatibility."""

    def test_update_is_jittable(self):
        """update_curriculum_state should be JIT-compilable."""
        params = CurriculumParams(
            init_ema_val=jnp.asarray(0.15, dtype=jnp.float32),
            low_band=jnp.asarray(0.10, dtype=jnp.float32),
            high_band=jnp.asarray(0.20, dtype=jnp.float32),
            adjust_factor=jnp.asarray(0.9, dtype=jnp.float32),
            consecutive_k=jnp.asarray(3, dtype=jnp.int32),
            min_threshold=jnp.asarray(0.1, dtype=jnp.float32),
            ema_alpha=jnp.asarray(0.1, dtype=jnp.float32),
        )

        @jax.jit
        def jitted_update(state, rate):
            return update_curriculum_state(state, rate, params)

        state = create_curriculum_state(init_threshold=0.5, init_ema_val=0.15)
        rate = jnp.asarray(0.05, dtype=jnp.float32)

        # Should not raise
        new_state, direction, adjusted = jitted_update(state, rate)

        assert new_state.current_threshold.dtype == jnp.float32
        assert direction.dtype == jnp.int32

    def test_compute_stats_is_jittable(self):
        """compute_early_termination_stats should be JIT-compilable."""

        @jax.jit
        def jitted_stats(done, absorbing):
            return compute_early_termination_stats(done, absorbing)

        done = jnp.array([True, True, False])
        absorbing = jnp.array([True, False, False])

        # Should not raise
        early_count, rate = jitted_stats(done, absorbing)
        assert early_count.dtype == jnp.float32
        assert rate.dtype == jnp.float32


# =============================================================================
# Legacy Tests (kept for backwards compatibility)
# =============================================================================


def test_consecutive_k_triggers_decrease():
    """Legacy test: consecutive_k triggers threshold decrease."""
    cfg = {
        "init_ema_val": 0.15,
        "low_band": 0.10,
        "high_band": 0.20,
        "adjust_factor": 0.95,
        "consecutive_k": 2,
        "min_threshold": 0.1,
        "ema_alpha": 1.0,
    }
    params = create_curriculum_params(cfg)
    state = create_curriculum_state(0.3, params.init_ema_val)

    low_rate = jnp.asarray(0.05, dtype=jnp.float32)
    state, _, _ = update_curriculum_state(state, low_rate, params)
    state, _, adjusted = update_curriculum_state(state, low_rate, params)

    state_f = _to_float_tree(state)
    assert float(adjusted) == pytest.approx(1.0)
    assert state_f.current_threshold == pytest.approx(0.3 * 0.95, rel=1e-5)
    assert state_f.consecutive_count == pytest.approx(0.0)


def test_consecutive_k_triggers_increase_capped_by_ratchet():
    """Legacy test: consecutive_k triggers threshold increase but capped by ratchet."""
    cfg = {
        "init_ema_val": 0.15,
        "low_band": 0.10,
        "high_band": 0.20,
        "adjust_factor": 0.95,
        "consecutive_k": 2,
        "min_threshold": 0.1,
        "ema_alpha": 1.0,
    }
    params = create_curriculum_params(cfg)
    state = create_curriculum_state(0.3, params.init_ema_val)

    high_rate = jnp.asarray(0.4, dtype=jnp.float32)
    state, _, _ = update_curriculum_state(state, high_rate, params)
    state, _, adjusted = update_curriculum_state(state, high_rate, params)

    state_f = _to_float_tree(state)
    assert float(adjusted) == pytest.approx(1.0)
    # Ratchet: threshold capped at initial_threshold (0.3)
    assert state_f.current_threshold == pytest.approx(0.3, rel=1e-5)
