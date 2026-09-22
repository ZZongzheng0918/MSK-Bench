from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import jax


def count_params(params: Any) -> int:
    """Count the total number of trainable parameters in a pytree.

    This is the standard approach for counting parameters in JAX/Flax models,
    using `jax.tree_util.tree_leaves` to traverse all parameter arrays.

    Args:
        params: A pytree of parameters (e.g., from a Flax model).
            Can be nested dicts, NamedTuples, or any JAX-compatible pytree.

    Returns:
        Total number of scalar parameters across all leaves.

    Example:
        >>> import jax.numpy as jnp
        >>> params = {'dense': {'kernel': jnp.zeros((10, 5)), 'bias': jnp.zeros(5)}}
        >>> count_params(params)
        55
    """
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def count_params_by_path(params: dict, prefix: str) -> int:
    """Count parameters for keys matching a prefix.

    This is useful for counting parameters in specific submodules when using
    Flax's explicit `name` argument for submodules.

    Args:
        params: Parameter dict from a Flax model.
        prefix: Key prefix to match (e.g., 'actor', 'critic').

    Returns:
        Total number of parameters for matching keys.

    Example:
        >>> params = {'actor': {...}, 'critic': {...}, 'log_std': ...}
        >>> count_params_by_path(params, 'actor')  # counts params['actor']
    """
    total = 0
    for key, value in params.items():
        if key.startswith(prefix):
            total += count_params(value)
    return total


def count_actor_critic_params(params: dict) -> dict[str, int]:
    """Count parameters separately for actor, critic, and shared components.

    This function supports two common naming styles:
    - Explicit submodules named "actor" / "critic"
    - Prefix-based names like "actor_dense_0" / "critic_moe_layer_1"
    - "log_std" is treated as part of the actor parameters

    Args:
        params: Parameter dict from an ActorCritic-style network.

    Returns:
        Dict with 'actor', 'critic', 'shared', and 'total' param counts.

    Example:
        >>> counts = count_actor_critic_params(train_state.params)
        >>> print(f"Actor: {counts['actor']:,}, Critic: {counts['critic']:,}")
    """
    total_count = count_params(params)
    actor_count = count_params_by_path(params, "actor")
    critic_count = count_params_by_path(params, "critic")

    # Treat log_std as actor parameters when present.
    if "log_std" in params:
        actor_count += count_params(params["log_std"])

    shared_count = total_count - actor_count - critic_count

    return {
        "actor": actor_count,
        "critic": critic_count,
        "shared": shared_count,
        "total": total_count,
    }


# Backward compatibility alias
count_trainable_params = count_params


def log_network_architecture(
    params: Any,
    *,
    title: str = "Network Architecture",
    print_fn: Callable[[str], None] | None = None,
) -> None:
    """Pretty-print a flat view of the parameter pytree for quick inspection."""

    printer = print_fn or print

    def _emit(msg: str = "") -> None:
        printer(msg)

    def _print_params(node: Any, prefix: str = "") -> None:
        if isinstance(node, Mapping):
            for key in sorted(node.keys()):
                _print_params(node[key], prefix + key + "/")
        else:
            shape = getattr(node, "shape", ())
            size = int(getattr(node, "size", 0))
            name = prefix[:-1] if prefix else "<leaf>"
            _emit(f"{name:<55s} {str(shape):<20s} {size:>12,}")

    try:
        _emit("\n" + "=" * 80)
        _emit(title + ":")
        _emit("=" * 80)
        _emit("")
        _emit(f"{'Layer':<55s} {'Shape':<20s} {'Params':>12s}")
        _emit("-" * 89)
        _print_params(params)
        _emit("-" * 89)
        _emit(f"{'Total':<55s} {'':<20s} {count_params(params):>12,}")
        _emit("=" * 80 + "\n")
    except Exception as exc:  # pragma: no cover - debug utility
        _emit(f"warning: failed to print network architecture: {exc}")
