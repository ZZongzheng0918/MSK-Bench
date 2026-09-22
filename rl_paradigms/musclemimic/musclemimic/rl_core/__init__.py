"""
Core RL.

This module provides reusable components for on-policy RL algorithms:
- RolloutBuffer: Storage and processing for collected trajectories
- GAE computation
- Minibatch generation
"""

from musclemimic.rl_core.rollout_buffer import (
    RolloutBuffer,
    compute_gae,
    create_minibatches,
)

__all__ = [
    "RolloutBuffer",
    "compute_gae",
    "create_minibatches",
]
