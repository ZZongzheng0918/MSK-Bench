from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class PowerliftRewardConfig:
    hold_bonus: float = 100.0
    overhead_height: float = 50.0
    torso_upright: float = 100.0
    spine_straight: float = 80.0
    lumbar_effort: float = 30.0
    feet_anchor: float = 40.0
    bar_level: float = 30.0
    balance: float = 20.0
    drop_penalty: float = -100.0
    bend_over_penalty: float = -200.0
    act_reg: float = 0.001
    alive: float = 5.0
    done: float = -100.0

    def weights(self) -> Mapping[str, float]:
        return {
            "hold_bonus": self.hold_bonus,
            "overhead_height": self.overhead_height,
            "torso_upright": self.torso_upright,
            "spine_straight": self.spine_straight,
            "lumbar_effort": self.lumbar_effort,
            "feet_anchor": self.feet_anchor,
            "bar_level": self.bar_level,
            "balance": self.balance,
            "drop_penalty": self.drop_penalty,
            "bend_over_penalty": self.bend_over_penalty,
            "act_reg": self.act_reg,
            "alive": self.alive,
            "done": self.done,
        }


@dataclass(frozen=True)
class RewardResult:
    terms: dict[str, float]
    dense: float


def compute_powerlift_reward(
    *,
    hold_bonus: float,
    overhead_height: float,
    torso_upright: float,
    spine_straight: float,
    lumbar_effort: float,
    bend_over_penalty: float,
    activation,
    done: bool,
    config: PowerliftRewardConfig | None = None,
) -> RewardResult:
    config = PowerliftRewardConfig() if config is None else config
    activation_array = np.asarray(activation, dtype=np.float32)
    activation_cost = float(np.mean(np.square(activation_array))) if activation_array.size else 0.0
    terms = {
        "hold_bonus": float(hold_bonus),
        "overhead_height": float(overhead_height),
        "torso_upright": float(torso_upright),
        "spine_straight": float(spine_straight),
        "lumbar_effort": float(lumbar_effort),
        "feet_anchor": 0.0,
        "bar_level": 0.0,
        "balance": 0.0,
        "drop_penalty": 0.0,
        "bend_over_penalty": float(bend_over_penalty),
        "act_reg": -activation_cost,
        "alive": 1.0,
        "done": 1.0 if done else 0.0,
        "sparse": 0.0,
        "solved": float(hold_bonus > 0.5),
    }
    dense = 0.0
    for key, weight in config.weights().items():
        dense += weight * terms.get(key, 0.0)
    terms["dense"] = float(dense)
    return RewardResult(terms=terms, dense=float(dense))
