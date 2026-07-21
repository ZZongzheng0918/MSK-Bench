"""
MoE (Mixture of Experts) loss and metrics helpers for PPO.

All functions here must be JIT-compatible since they run inside traced code.
"""

from __future__ import annotations

from typing import Any


def aggregate_moe_metrics(moe_metrics: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    """
    Aggregate MoE metrics across actor/critic towers and layers.

    This function is JIT-safe (no Python control flow that depends on traced values).

    Args:
        moe_metrics: dict from network.apply with return_metrics=True

    Returns:
        (moe_loss, gate_entropy, expert_usage_var, top2_usage, gate_w_mean, gate_w_std)
    """
    moe_loss = moe_metrics.get("total_moe_loss", 0.0)
    gate_entropy = 0.0
    expert_usage_var = 0.0
    top2_usage = 0.0
    gate_w_mean = 0.0
    gate_w_std = 0.0
    num_layers = 0

    for tower in ["actor", "critic"]:
        if tower in moe_metrics:
            tower_metrics = moe_metrics[tower]
            for layer_key, layer_data in tower_metrics.items():
                if layer_key.endswith("_metrics") and isinstance(layer_data, dict):
                    gate_entropy += layer_data.get("gate_entropy", 0.0)
                    expert_usage_var += layer_data.get("expert_utilization_var", 0.0)
                    top2_usage += layer_data.get("top2_expert_usage", 0.0)
                    gate_w_mean += layer_data.get("gate_weights_mean", 0.0)
                    gate_w_std += layer_data.get("gate_weights_std", 0.0)
                    num_layers += 1

    if num_layers > 0:
        gate_entropy /= num_layers
        expert_usage_var /= num_layers
        top2_usage /= num_layers
        gate_w_mean /= num_layers
        gate_w_std /= num_layers

    return moe_loss, gate_entropy, expert_usage_var, top2_usage, gate_w_mean, gate_w_std


def zero_moe_metrics() -> tuple[float, float, float, float, float, float]:
    """Return zeroed MoE metrics tuple for non-MoE networks."""
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
