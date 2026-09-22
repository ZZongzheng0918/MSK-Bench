"""
PPO Loss Functions.

Pure functions for computing PPO losses.
"""

from typing import NamedTuple

import jax.numpy as jnp


class PPOLossOutput(NamedTuple):
    """
    Output from PPO loss computation.

    All fields are scalars (mean over batch).
    This is a NamedTuple for pytree compatibility with jax.tree_util.
    """

    total_loss: jnp.ndarray
    value_loss: jnp.ndarray
    actor_loss: jnp.ndarray
    entropy: jnp.ndarray
    kl_mean: jnp.ndarray


def ppo_value_loss(
    value: jnp.ndarray,
    value_old: jnp.ndarray,
    targets: jnp.ndarray,
    clip_eps: float,
) -> jnp.ndarray:
    """
    Clipped value function loss.

    Args:
        value: Current value predictions, shape (batch,)
        value_old: Value predictions from rollout, shape (batch,)
        targets: Value targets (returns), shape (batch,)
        clip_eps: Clipping epsilon for value updates

    Returns:
        Scalar mean value loss
    """
    value_pred_clipped = value_old + (value - value_old).clip(-clip_eps, clip_eps)
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    return value_loss


def ppo_actor_loss(
    log_prob: jnp.ndarray,
    log_prob_old: jnp.ndarray,
    advantages: jnp.ndarray,
    clip_eps: float,
    return_ratio_stats: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, dict]:
    """
    Clipped surrogate policy loss.

    Note: advantages should be normalized BEFORE calling this function.

    Args:
        log_prob: Current policy log probabilities, shape (batch,)
        log_prob_old: Log probabilities from rollout, shape (batch,)
        advantages: Normalized advantages, shape (batch,)
        clip_eps: Clipping epsilon for policy ratio
        return_ratio_stats: If True, also return ratio statistics dict

    Returns:
        Scalar mean actor loss, or tuple of (loss, ratio_stats) if return_ratio_stats=True
    """
    ratio = jnp.exp(log_prob - log_prob_old)
    loss_actor1 = ratio * advantages
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss = loss_actor.mean()

    if return_ratio_stats:
        ratio_stats = {
            "ratio_mean": jnp.mean(ratio),
            "ratio_std": jnp.std(ratio),
            "ratio_min": jnp.min(ratio),
            "ratio_max": jnp.max(ratio),
            "clipped_ratio_frac": jnp.mean(jnp.abs(ratio - 1.0) > clip_eps),
        }
        return loss, ratio_stats
    return loss


def approx_kl(
    log_prob_old: jnp.ndarray,
    log_prob: jnp.ndarray,
) -> jnp.ndarray:
    """
    Sample-based approximate KL divergence.

    KL ≈ E[log π_old(a|s) - log π_new(a|s)]

    This is the most common KL approximation in PPO implementations.
    Works for any distribution (Gaussian, categorical, squashed, etc.)
    and is simple and cheap to compute.

    Args:
        log_prob_old: Log probabilities from rollout policy, shape (batch,)
        log_prob: Log probabilities from current policy, shape (batch,)

    Returns:
        Scalar approximate KL divergence
    """
    return (log_prob_old - log_prob).mean()


def normalize_advantages(advantages: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize advantages to zero mean and unit variance.

    Args:
        advantages: Raw advantages, shape (batch,)

    Returns:
        Normalized advantages, shape (batch,)
    """
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def ppo_loss(
    log_prob: jnp.ndarray,
    log_prob_old: jnp.ndarray,
    value: jnp.ndarray,
    value_old: jnp.ndarray,
    targets: jnp.ndarray,
    advantages: jnp.ndarray,
    entropy: jnp.ndarray,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    clip_eps_vf: float | None = None,
) -> PPOLossOutput:
    """
    Complete PPO loss computation.

    Note: This function normalizes advantages internally.

    Args:
        log_prob: Current policy log probs, shape (batch,)
        log_prob_old: Old policy log probs, shape (batch,)
        value: Current value predictions, shape (batch,)
        value_old: Old value predictions, shape (batch,)
        targets: Value targets, shape (batch,)
        advantages: Raw advantages (will be normalized), shape (batch,)
        entropy: Policy entropy (scalar or per-sample mean)
        clip_eps: PPO clipping epsilon for actor
        vf_coef: Value function loss coefficient
        ent_coef: Entropy bonus coefficient
        clip_eps_vf: PPO clipping epsilon for value function (defaults to clip_eps)

    Returns:
        PPOLossOutput with total_loss and component losses
    """
    vf_clip = clip_eps_vf if clip_eps_vf is not None else clip_eps

    # Value loss
    v_loss = ppo_value_loss(value, value_old, targets, vf_clip)

    # Actor loss (advantages normalized here, matching original)
    normalized_adv = normalize_advantages(advantages)
    a_loss = ppo_actor_loss(log_prob, log_prob_old, normalized_adv, clip_eps)

    # KL divergence (sample-based approximation)
    kl = approx_kl(log_prob_old, log_prob)

    # Combined loss
    total = a_loss + vf_coef * v_loss - ent_coef * entropy

    return PPOLossOutput(
        total_loss=total,
        value_loss=v_loss,
        actor_loss=a_loss,
        entropy=entropy,
        kl_mean=kl,
    )
