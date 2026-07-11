from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from .dataclasses import (
    Trajectory,
    TrajectoryInfo,
    TrajectoryModel,
    TrajectoryData,
    TrajectoryTransitions,
    interpolate_trajectories,
)

__all__ = [
    "Trajectory",
    "TrajectoryInfo",
    "TrajectoryModel",
    "TrajectoryData",
    "TrajectoryTransitions",
    "interpolate_trajectories",
    "TrajectoryHandler",
    "TrajState",
]

_LAZY_ATTRS: Dict[str, str] = {
    "TrajectoryHandler": "TrajectoryHandler",
    "TrajState": "TrajState",
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        from . import handler as _handler

        value = getattr(_handler, _LAZY_ATTRS[name])
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


if TYPE_CHECKING:
    from .handler import TrajectoryHandler, TrajState
