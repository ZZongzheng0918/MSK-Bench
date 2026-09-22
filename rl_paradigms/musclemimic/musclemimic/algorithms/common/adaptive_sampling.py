"""Adaptive trajectory sampling utilities for PPO training.

Provides utilities for computing trajectory sampling weights based on
early termination rates, enabling more training focus on harder trajectories.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_adaptive_weights(
    done_counts: jnp.ndarray,
    early_counts: jnp.ndarray,
    ema_done: jnp.ndarray,
    ema_early: jnp.ndarray,
    beta: float = 0.2,
    alpha: float = 1.0,
    floor_mix: float = 0.1,
    eps_div: float = 1e-6,
    eps_pow: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute trajectory sampling weights from early termination rates using count-EMA.

    Higher early termination rate -> higher sampling probability (prioritize harder trajectories).

    Pure 1D interface - runner is responsible for broadcasting to (num_envs, n_traj).

    Args:
        done_counts: (n_traj,) done counts from current rollout
        early_counts: (n_traj,) early termination counts from current rollout
        ema_done: (n_traj,) EMA state for done counts
        ema_early: (n_traj,) EMA state for early termination counts
        beta: EMA decay rate (higher = faster adaptation)
        alpha: Prioritization exponent (higher = more aggressive)
        floor_mix: Uniform mixture floor, i.e., 1 - lambda (ensures all trajectories get some samples)
        eps_div: Small constant for division stability
        eps_pow: Small constant for power stability

    Returns:
        weights_1d: (n_traj,) sampling weights
        new_ema_done: (n_traj,) updated EMA done counts
        new_ema_early: (n_traj,) updated EMA early counts
        rate_hat: (n_traj,) estimated early termination rates (for logging)
    """
    # EMA update
    new_ema_done = (1 - beta) * ema_done + beta * done_counts
    new_ema_early = (1 - beta) * ema_early + beta * early_counts

    # Rate estimation (eps_div prevents division by zero)
    rate_hat = new_ema_early / (new_ema_done + eps_div)

    # Priority weights (eps_pow prevents zero priorities, higher rate = higher weight)
    priorities = jnp.power(rate_hat + eps_pow, alpha)

    # NaN/Inf safeguard - replace non-finite values with 0
    priorities = jnp.where(jnp.isfinite(priorities), priorities, 0.0)

    # NaN/zero fallback - if sum is 0, NaN, or Inf, use uniform
    n_traj = done_counts.shape[0]
    uniform = jnp.ones(n_traj) / n_traj
    sum_p = jnp.sum(priorities)
    use_priorities = jnp.isfinite(sum_p) & (sum_p > 0)
    normalized = jnp.where(use_priorities, priorities / sum_p, uniform)

    # Uniform floor mix (ensures all trajectories get at least floor_mix probability)
    weights_1d = (1 - floor_mix) * normalized + floor_mix * uniform

    return weights_1d, new_ema_done, new_ema_early, rate_hat


def compute_topk_weights(weights_1d: jnp.ndarray, k: int = 10) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get top-k highest weighted trajectories.

    Args:
        weights_1d: 1D array of sampling weights
        k: Number of top trajectories to return

    Returns:
        topk_vals: Top-k weight values
        topk_ids: Trajectory indices of top-k weights
    """
    topk_vals, topk_ids = jax.lax.top_k(weights_1d, k)
    return topk_vals, topk_ids


def compute_per_traj_termination_stats(
    traj_batch_info: dict,
    done: jnp.ndarray,
    absorbing: jnp.ndarray,
    n_trajectories: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute per-trajectory termination stats from a rollout batch.

    Args:
        traj_batch_info: Info dict from traj_batch with 'final_traj_no'
        done: Done flags from rollout, shape (num_steps, num_envs)
        absorbing: Absorbing state flags (early termination), shape (num_steps, num_envs)
        n_trajectories: Total number of trajectories in dataset

    Returns:
        term_rate: per-trajectory early-termination rate (early / done)
        done_counts: per-trajectory done counts
        early_counts: per-trajectory early-termination counts
    """
    final_traj_no = traj_batch_info["final_traj_no"]
    traj_ids = final_traj_no.flatten().astype(jnp.int32)
    done_flat = done.flatten().astype(jnp.float32)
    early_flat = jnp.logical_and(done, absorbing).flatten().astype(jnp.float32)

    done_counts = jax.ops.segment_sum(done_flat, traj_ids, num_segments=n_trajectories)
    early_counts = jax.ops.segment_sum(early_flat, traj_ids, num_segments=n_trajectories)
    term_rate = early_counts / jnp.maximum(done_counts, 1.0)
    return term_rate, done_counts, early_counts
