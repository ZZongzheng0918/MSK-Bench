"""Lossless structuring of the controller-visible 2,418-D policy observation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ObservationLayout:
    """Layout derived from the environment, not guessed constants."""

    joint_position_dim: int
    joint_velocity_dim: int
    muscle_count: int
    relative_site_count: int
    touch_names: tuple[str, ...] = ("r_foot", "r_toes", "l_foot", "l_toes")
    root_position_dim: int = 5
    root_velocity_dim: int = 6
    lookahead_count: int = 4

    @classmethod
    def from_env(cls, env: Any) -> "ObservationLayout":
        base = _unwrap(env)
        wrapper = base._goal_wrapper
        wrapper._lazy_init()
        lookahead_count = int(wrapper.N_STEP_LOOKAHEAD) - 1
        actual_goal_dim = int(np.asarray(wrapper.get_goal_obs()).size)
        # Goal formula for s relative sites and k lookaheads:
        # current=(3s+3s+6s), reference=3s+k*(3+6+3s), phase=1.
        constant = lookahead_count * 9 + 1
        coefficient = 15 + lookahead_count * 3
        numerator = actual_goal_dim - constant
        if numerator <= 0 or numerator % coefficient:
            raise ValueError(
                f"Cannot derive relative-site count from goal dim {actual_goal_dim} "
                f"and {lookahead_count} lookaheads."
            )
        layout = cls(
            joint_position_dim=int(base._model.nq) - 7,
            joint_velocity_dim=int(base._model.nv) - 6,
            muscle_count=int(base._model.nu),
            relative_site_count=numerator // coefficient,
            lookahead_count=lookahead_count,
        )
        if layout.goal_dim != actual_goal_dim:
            raise AssertionError(f"Derived goal dim {layout.goal_dim}, actual {actual_goal_dim}.")
        return layout

    @property
    def current_site_position_dim(self) -> int:
        return 3 * self.relative_site_count

    @property
    def current_site_angle_dim(self) -> int:
        return 3 * self.relative_site_count

    @property
    def current_site_velocity_dim(self) -> int:
        return 6 * self.relative_site_count

    @property
    def goal_dim(self) -> int:
        site = self.current_site_position_dim
        current = site + self.current_site_angle_dim + self.current_site_velocity_dim
        reference = site + self.lookahead_count * (3 + 6 + site)
        return current + reference + 1

    @property
    def base_dim(self) -> int:
        return (
            self.root_position_dim
            + self.joint_position_dim
            + self.root_velocity_dim
            + self.joint_velocity_dim
            + 5 * self.muscle_count
            + len(self.touch_names)
        )

    @property
    def total_dim(self) -> int:
        return self.base_dim + self.goal_dim

    def as_dict(self) -> dict[str, Any]:
        return {
            "root_position_no_xy": self.root_position_dim,
            "joint_position": self.joint_position_dim,
            "root_velocity": self.root_velocity_dim,
            "joint_velocity": self.joint_velocity_dim,
            "muscle_count": self.muscle_count,
            "muscle_fields": ["length", "velocity", "force", "excitation", "activation"],
            "touch_names": list(self.touch_names),
            "relative_site_count": self.relative_site_count,
            "lookahead_count": self.lookahead_count,
            "base_dim": self.base_dim,
            "goal_dim": self.goal_dim,
            "total_dim": self.total_dim,
        }


def _unwrap(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    return getattr(current, "unwrapped", current)


class _Cursor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.index = 0

    def take(self, size: int) -> list[float]:
        result = self.values[self.index : self.index + size]
        if result.size != size:
            raise ValueError(f"Observation ended at {self.index}; needed {size} more values.")
        self.index += size
        return [float(value) for value in result]


def serialize_policy_observation(
    observation: Any,
    layout: ObservationLayout,
    *,
    task: str,
    decision_index: int,
    simulation_time: float,
    exported_at_utc: str | None = None,
) -> dict[str, Any]:
    """Return a lossless controller-visible payload with no simulator-only diagnostics."""

    values = np.asarray(observation, dtype=np.float32).reshape(-1)
    if values.size != layout.total_dim:
        raise ValueError(f"Expected observation dim {layout.total_dim}, got {values.size}.")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("Controller observation contains NaN or infinity.")
    cursor = _Cursor(values)
    payload: dict[str, Any] = {
        "visibility": "controller-visible",
        "contract": "Only this file and env_info.json may be used to choose the next GPT action.",
        "task": str(task),
        "decision_index": int(decision_index),
        "simulation_time": float(simulation_time),
        "observation_dim": int(layout.total_dim),
    }
    if exported_at_utc is not None:
        payload["exported_at_utc"] = str(exported_at_utc)
    payload["kinematics"] = {
        "root_position_no_xy": cursor.take(layout.root_position_dim),
        "joint_position": cursor.take(layout.joint_position_dim),
        "root_velocity": cursor.take(layout.root_velocity_dim),
        "joint_velocity": cursor.take(layout.joint_velocity_dim),
    }
    muscles = np.asarray(cursor.take(5 * layout.muscle_count), dtype=np.float32)
    payload["muscles"] = {
        "fields": ["length", "velocity", "force", "excitation", "activation"],
        "values": muscles.reshape(layout.muscle_count, 5).astype(float).tolist(),
    }
    payload["touch"] = dict(zip(layout.touch_names, cursor.take(len(layout.touch_names)), strict=True))
    site = layout.current_site_position_dim
    goal: dict[str, Any] = {
        "current_relative_site_position": cursor.take(site),
        "current_relative_site_angle": cursor.take(layout.current_site_angle_dim),
        "current_relative_site_velocity": cursor.take(layout.current_site_velocity_dim),
        "reference_current_site_position": cursor.take(site),
        "lookahead": [],
    }
    for _ in range(layout.lookahead_count):
        goal["lookahead"].append(
            {
                "root_position_delta": cursor.take(3),
                "root_velocity_delta": cursor.take(6),
                "site_position": cursor.take(site),
            }
        )
    goal["motion_phase"] = cursor.take(1)[0]
    payload["goal"] = goal
    if cursor.index != values.size:
        raise AssertionError(f"Serializer consumed {cursor.index}/{values.size} values.")
    return payload


def flatten_serialized_observation(payload: dict[str, Any], layout: ObservationLayout) -> np.ndarray:
    """Reconstruct the original vector; proves serialization is lossless."""

    kin = payload["kinematics"]
    sections: list[Any] = [
        kin["root_position_no_xy"],
        kin["joint_position"],
        kin["root_velocity"],
        kin["joint_velocity"],
        np.asarray(payload["muscles"]["values"], dtype=np.float32).reshape(-1),
        [payload["touch"][name] for name in layout.touch_names],
    ]
    goal = payload["goal"]
    sections.extend(
        [
            goal["current_relative_site_position"],
            goal["current_relative_site_angle"],
            goal["current_relative_site_velocity"],
            goal["reference_current_site_position"],
        ]
    )
    for lookahead in goal["lookahead"]:
        sections.extend(
            [lookahead["root_position_delta"], lookahead["root_velocity_delta"], lookahead["site_position"]]
        )
    sections.append([goal["motion_phase"]])
    result = np.concatenate([np.asarray(section, dtype=np.float32).reshape(-1) for section in sections])
    if result.size != layout.total_dim:
        raise ValueError(f"Serialized payload reconstructs {result.size}, expected {layout.total_dim}.")
    return result
