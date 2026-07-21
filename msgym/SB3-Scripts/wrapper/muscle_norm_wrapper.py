from typing import Any, Union

import gymnasium as gym
import numpy as np

from msk_bench.action_transform import canonical_policy_action_to_excitation


class MuscleNormWrapper(gym.ActionWrapper):
    """Maps canonical policy actions from [-1, 1] linearly to muscle excitation [0, 1]."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_space.shape[0],),
            dtype=np.float32,
        )

    def action(self, action: Union[np.ndarray, Any]) -> np.ndarray:
        return canonical_policy_action_to_excitation(action)
