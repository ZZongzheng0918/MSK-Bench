"""Unit tests for adaptive motion sampling functionality (count-EMA based)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# Import the functions from adaptive_sampling module
from musclemimic.algorithms.common.adaptive_sampling import (
    compute_adaptive_weights,
    compute_per_traj_termination_stats,
    compute_topk_weights,
)


class TestComputeAdaptiveWeights:
    """Test the compute_adaptive_weights function (count-EMA based)."""

    def test_basic_weight_computation(self):
        """Test that weights are computed correctly from early termination counts."""
        n_traj = 5

        # Trajectory 0 has high early termination rate, trajectory 4 has low rate
        done_counts = jnp.array([10.0, 10.0, 10.0, 10.0, 10.0])
        early_counts = jnp.array([8.0, 6.0, 4.0, 2.0, 1.0])

        # Initialize EMA states
        ema_done = jnp.zeros(n_traj)
        ema_early = jnp.zeros(n_traj)

        weights, new_ema_done, new_ema_early, rate_hat = compute_adaptive_weights(
            done_counts, early_counts, ema_done, ema_early,
            beta=1.0,  # No smoothing for this test
            floor_mix=0.1,
            alpha=1.0,
            eps_div=1e-6,
            eps_pow=1e-6,
        )

        # Check shape
        assert weights.shape == (n_traj,), f"Expected shape {(n_traj,)}, got {weights.shape}"

        # Weights should sum to 1
        np.testing.assert_almost_equal(float(jnp.sum(weights)), 1.0, decimal=5)

        # Higher early termination rate should have higher weight
        # Trajectory 0 has rate 0.8, trajectory 4 has rate 0.1
        assert weights[0] > weights[4], "Higher early term rate should have higher weight"

    def test_ema_update(self):
        """Test that EMA state is updated correctly."""
        n_traj = 3
        beta = 0.5

        done_counts = jnp.array([10.0, 20.0, 30.0])
        early_counts = jnp.array([5.0, 10.0, 15.0])

        ema_done = jnp.array([20.0, 20.0, 20.0])
        ema_early = jnp.array([10.0, 10.0, 10.0])

        _, new_ema_done, new_ema_early, _ = compute_adaptive_weights(
            done_counts, early_counts, ema_done, ema_early,
            beta=beta,
        )

        # EMA update: new = (1-beta) * old + beta * new
        expected_ema_done = (1 - beta) * ema_done + beta * done_counts
        expected_ema_early = (1 - beta) * ema_early + beta * early_counts

        np.testing.assert_array_almost_equal(new_ema_done, expected_ema_done, decimal=5)
        np.testing.assert_array_almost_equal(new_ema_early, expected_ema_early, decimal=5)

    def test_floor_mix_ensures_minimum_probability(self):
        """Test that floor_mix ensures all trajectories have some probability."""
        n_traj = 10

        # Only trajectory 0 has terminations
        done_counts = jnp.array([100.0] + [0.0] * 9)
        early_counts = jnp.array([50.0] + [0.0] * 9)

        ema_done = jnp.zeros(n_traj)
        ema_early = jnp.zeros(n_traj)

        weights, _, _, _ = compute_adaptive_weights(
            done_counts, early_counts, ema_done, ema_early,
            beta=1.0,
            floor_mix=0.1,
            alpha=1.0,
        )

        # With floor_mix=0.1, all trajectories should have at least floor_mix * uniform
        uniform_weight = 1.0 / n_traj
        min_expected = 0.1 * uniform_weight

        # All weights should be positive due to floor_mix
        assert jnp.all(weights >= min_expected - 1e-6), "All weights should be at least floor_mix * uniform"

    def test_alpha_controls_prioritization(self):
        """Test that higher alpha leads to more aggressive prioritization."""
        n_traj = 3

        done_counts = jnp.array([100.0, 100.0, 100.0])
        early_counts = jnp.array([90.0, 50.0, 10.0])

        ema_done = jnp.zeros(n_traj)
        ema_early = jnp.zeros(n_traj)

        weights_alpha1, _, _, _ = compute_adaptive_weights(
            done_counts, early_counts, ema_done, ema_early,
            beta=1.0,
            floor_mix=0.0,
            alpha=1.0,
        )
        weights_alpha2, _, _, _ = compute_adaptive_weights(
            done_counts, early_counts, ema_done, ema_early,
            beta=1.0,
            floor_mix=0.0,
            alpha=2.0,
        )

        # Higher alpha should make the ratio between high and low weights larger
        ratio_alpha1 = weights_alpha1[0] / weights_alpha1[2]
        ratio_alpha2 = weights_alpha2[0] / weights_alpha2[2]

        assert ratio_alpha2 > ratio_alpha1, "Higher alpha should increase weight disparity"

    def test_weights_are_valid_probability_distribution(self):
        """Test that weights form a valid probability distribution."""
        n_traj = 20

        # Random counts
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)
        done_counts = jax.random.uniform(key1, (n_traj,), minval=1.0, maxval=100.0)
        early_counts = jax.random.uniform(key2, (n_traj,), minval=0.0, maxval=50.0)

        ema_done = jnp.zeros(n_traj)
        ema_early = jnp.zeros(n_traj)

        weights, _, _, _ = compute_adaptive_weights(
            done_counts, early_counts, ema_done, ema_early,
            beta=0.5,
            floor_mix=0.1,
            alpha=1.5,
        )

        # All weights should be non-negative
        assert jnp.all(weights >= 0), "All weights should be non-negative"

        # Weights should sum to 1
        np.testing.assert_almost_equal(float(jnp.sum(weights)), 1.0, decimal=5)

    def test_rate_hat_computation(self):
        """Test that rate_hat is computed correctly."""
        n_traj = 3
        done_counts = jnp.array([100.0, 50.0, 200.0])
        early_counts = jnp.array([50.0, 25.0, 100.0])

        ema_done = jnp.zeros(n_traj)
        ema_early = jnp.zeros(n_traj)

        _, new_ema_done, new_ema_early, rate_hat = compute_adaptive_weights(
            done_counts, early_counts, ema_done, ema_early,
            beta=1.0,  # No smoothing
            eps_div=1e-6,
        )

        # Rate should be early / done
        expected_rate = early_counts / (done_counts + 1e-6)
        np.testing.assert_array_almost_equal(rate_hat, expected_rate, decimal=5)


class TestComputePerTrajTerminationStats:
    """Test the compute_per_traj_termination_stats function."""

    def test_basic_stats_computation(self):
        """Test basic per-trajectory statistics computation."""
        n_traj = 3
        num_steps = 5
        num_envs = 2

        # Create mock data
        # Trajectory 0: 2 done, 1 early
        # Trajectory 1: 3 done, 2 early
        # Trajectory 2: 1 done, 0 early
        final_traj_no = jnp.array([
            [0, 1],
            [1, 2],
            [0, 1],
            [1, 0],
            [2, 1],
        ], dtype=jnp.int32)

        done = jnp.array([
            [True, True],
            [False, True],
            [True, True],
            [True, False],
            [False, True],
        ])

        absorbing = jnp.array([
            [True, False],
            [False, True],
            [False, True],
            [True, False],
            [False, True],
        ])

        traj_batch_info = {"final_traj_no": final_traj_no}

        term_rate, done_counts, early_counts = compute_per_traj_termination_stats(
            traj_batch_info, done, absorbing, n_traj
        )

        # Check shapes
        assert term_rate.shape == (n_traj,)
        assert done_counts.shape == (n_traj,)
        assert early_counts.shape == (n_traj,)

        # Verify counts are non-negative
        assert jnp.all(done_counts >= 0)
        assert jnp.all(early_counts >= 0)
        assert jnp.all(early_counts <= done_counts)

    def test_empty_done_gives_zero_rate(self):
        """Test that trajectories with no done episodes have zero rate."""
        n_traj = 3
        num_steps = 5
        num_envs = 2

        final_traj_no = jnp.zeros((num_steps, num_envs), dtype=jnp.int32)
        done = jnp.zeros((num_steps, num_envs), dtype=bool)
        absorbing = jnp.zeros((num_steps, num_envs), dtype=bool)

        traj_batch_info = {"final_traj_no": final_traj_no}

        term_rate, done_counts, early_counts = compute_per_traj_termination_stats(
            traj_batch_info, done, absorbing, n_traj
        )

        # With no done episodes, rate should be 0
        np.testing.assert_array_almost_equal(term_rate, jnp.zeros(n_traj), decimal=5)


class TestComputeTopkWeights:
    """Test the compute_topk_weights function."""

    def test_basic_topk(self):
        """Test basic top-k extraction."""
        weights = jnp.array([0.1, 0.5, 0.2, 0.15, 0.05])
        k = 3

        vals, ids = compute_topk_weights(weights, k=k)

        assert vals.shape == (k,), f"Expected shape ({k},), got {vals.shape}"
        assert ids.shape == (k,), f"Expected shape ({k},), got {ids.shape}"

        # Top value should be 0.5 (index 1)
        np.testing.assert_almost_equal(float(vals[0]), 0.5, decimal=5)
        assert int(ids[0]) == 1

    def test_topk_sorted_descending(self):
        """Test that top-k values are sorted in descending order."""
        weights = jnp.array([0.3, 0.1, 0.4, 0.2])
        k = 4

        vals, ids = compute_topk_weights(weights, k=k)

        # Check values are descending
        for i in range(k - 1):
            assert vals[i] >= vals[i + 1], f"Values should be descending: {vals}"

    def test_topk_with_k_equal_to_array_size(self):
        """Test top-k when k equals array size."""
        weights = jnp.array([0.3, 0.7])
        k = 2

        vals, ids = compute_topk_weights(weights, k=k)

        assert vals.shape == (2,)
        np.testing.assert_almost_equal(float(vals[0]), 0.7, decimal=5)


class TestEdgeCases:
    """Test edge cases for adaptive sampling."""

    def test_single_trajectory(self):
        """Test with a single trajectory."""
        n_traj = 1

        done_counts = jnp.array([10.0])
        early_counts = jnp.array([5.0])
        ema_done = jnp.zeros(n_traj)
        ema_early = jnp.zeros(n_traj)

        weights, _, _, _ = compute_adaptive_weights(
            done_counts, early_counts, ema_done, ema_early,
        )

        # Single trajectory should have weight 1.0
        np.testing.assert_almost_equal(float(weights[0]), 1.0, decimal=5)

    def test_all_zero_counts(self):
        """Test with all zero counts."""
        n_traj = 5

        done_counts = jnp.zeros(n_traj)
        early_counts = jnp.zeros(n_traj)
        ema_done = jnp.zeros(n_traj)
        ema_early = jnp.zeros(n_traj)

        weights, _, _, _ = compute_adaptive_weights(
            done_counts, early_counts, ema_done, ema_early,
        )

        # With all zeros, should fall back to uniform (via NaN fallback)
        expected = 1.0 / n_traj
        np.testing.assert_array_almost_equal(weights, jnp.full(n_traj, expected), decimal=5)

    def test_equal_rates_give_uniform_weights(self):
        """Test that equal early termination rates give approximately uniform weights."""
        n_traj = 5

        # All trajectories have the same rate (50%)
        done_counts = jnp.ones(n_traj) * 100.0
        early_counts = jnp.ones(n_traj) * 50.0
        ema_done = jnp.zeros(n_traj)
        ema_early = jnp.zeros(n_traj)

        weights, _, _, _ = compute_adaptive_weights(
            done_counts, early_counts, ema_done, ema_early,
            beta=1.0,
            floor_mix=0.0,
            alpha=1.0,
        )

        # All weights should be approximately equal
        expected = 1.0 / n_traj
        for i in range(n_traj):
            np.testing.assert_almost_equal(float(weights[i]), expected, decimal=4)

    def test_nan_fallback(self):
        """Test that NaN/zero fallback works correctly."""
        n_traj = 3

        # Zero done counts should trigger fallback
        done_counts = jnp.zeros(n_traj)
        early_counts = jnp.zeros(n_traj)
        ema_done = jnp.zeros(n_traj)
        ema_early = jnp.zeros(n_traj)

        weights, _, _, _ = compute_adaptive_weights(
            done_counts, early_counts, ema_done, ema_early,
        )

        # Should fall back to uniform
        assert jnp.all(jnp.isfinite(weights)), "Weights should be finite (no NaN)"
        np.testing.assert_almost_equal(float(jnp.sum(weights)), 1.0, decimal=5)

    def test_nan_in_ema_input_fallback(self):
        """Test that NaN in EMA inputs produces valid uniform fallback."""
        n_traj = 5

        # Inject NaN into ema_done (simulates corrupted carry state)
        done_counts = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        early_counts = jnp.array([5.0, 10.0, 15.0, 20.0, 25.0])
        ema_done = jnp.array([10.0, jnp.nan, 30.0, 40.0, 50.0])  # NaN at index 1
        ema_early = jnp.array([5.0, 10.0, 15.0, 20.0, 25.0])

        weights, _, _, _ = compute_adaptive_weights(
            done_counts, early_counts, ema_done, ema_early,
            beta=0.5,
            floor_mix=0.0,  # No floor mix to test pure fallback
        )

        # All weights should be finite
        assert jnp.all(jnp.isfinite(weights)), f"Weights contain NaN/Inf: {weights}"

        # Weights should sum to 1
        np.testing.assert_almost_equal(float(jnp.sum(weights)), 1.0, decimal=5)

        # All weights should be non-negative
        assert jnp.all(weights >= 0), f"Negative weights found: {weights}"

    def test_inf_in_ema_input_fallback(self):
        """Test that Inf in EMA inputs produces valid uniform fallback."""
        n_traj = 4

        # Inject Inf into ema_early (simulates overflow)
        done_counts = jnp.array([10.0, 20.0, 30.0, 40.0])
        early_counts = jnp.array([5.0, 10.0, 15.0, 20.0])
        ema_done = jnp.array([10.0, 20.0, 30.0, 40.0])
        ema_early = jnp.array([5.0, jnp.inf, 15.0, 20.0])  # Inf at index 1

        weights, _, _, _ = compute_adaptive_weights(
            done_counts, early_counts, ema_done, ema_early,
            beta=0.5,
            floor_mix=0.0,
        )

        # All weights should be finite
        assert jnp.all(jnp.isfinite(weights)), f"Weights contain NaN/Inf: {weights}"

        # Weights should sum to 1
        np.testing.assert_almost_equal(float(jnp.sum(weights)), 1.0, decimal=5)


class TestJITCompatibility:
    """Test that functions work correctly under JIT compilation."""

    def test_compute_weights_jit(self):
        """Test that compute_adaptive_weights works under JIT."""
        n_traj = 10

        @jax.jit
        def jitted_compute(done_counts, early_counts, ema_done, ema_early):
            return compute_adaptive_weights(
                done_counts, early_counts, ema_done, ema_early,
                beta=0.2, floor_mix=0.1, alpha=1.0,
            )

        key = jax.random.PRNGKey(0)
        done_counts = jax.random.uniform(key, (n_traj,), minval=1.0, maxval=100.0)
        early_counts = jax.random.uniform(key, (n_traj,), minval=0.0, maxval=50.0)
        ema_done = jnp.zeros(n_traj)
        ema_early = jnp.zeros(n_traj)

        # Should not raise
        weights, new_ema_done, new_ema_early, rate_hat = jitted_compute(
            done_counts, early_counts, ema_done, ema_early
        )
        assert weights.shape == (n_traj,)
        assert new_ema_done.shape == (n_traj,)
        assert new_ema_early.shape == (n_traj,)
        assert rate_hat.shape == (n_traj,)

    def test_compute_topk_jit(self):
        """Test that compute_topk_weights works under JIT."""
        @jax.jit
        def jitted_topk(weights):
            return compute_topk_weights(weights, k=5)

        weights = jnp.array([0.1, 0.3, 0.2, 0.25, 0.15])
        vals, ids = jitted_topk(weights)

        assert vals.shape == (5,)
        assert ids.shape == (5,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
