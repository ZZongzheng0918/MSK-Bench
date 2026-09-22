from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def unwrap_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    return current.unwrapped if hasattr(current, "unwrapped") else current


def sim_from_env(env: Any) -> Any:
    base = unwrap_env(env)
    sim = getattr(base, "sim", None)
    if sim is None:
        raise AttributeError("The environment does not expose a MuJoCo sim object.")
    return sim


def actuator_priors(sim: Any) -> np.ndarray:
    return np.stack(
        [
            np.asarray(sim.model.actuator_gainprm[:, 0], dtype=np.float32),
            np.asarray(sim.model.actuator_biasprm[:, 0], dtype=np.float32),
        ],
        axis=1,
    )


def muscle_states(sim: Any) -> np.ndarray:
    num_muscles = int(sim.model.nu)
    if getattr(sim.model, "na", 0) > 0 and getattr(sim.data, "act", None) is not None:
        activation = np.asarray(sim.data.act, dtype=np.float32)
    else:
        activation = np.zeros(num_muscles, dtype=np.float32)
    return np.stack(
        [
            np.asarray(sim.data.actuator_length, dtype=np.float32),
            np.asarray(sim.data.actuator_velocity, dtype=np.float32),
            np.asarray(sim.data.actuator_force, dtype=np.float32) / 1000.0,
            activation,
        ],
        axis=1,
    )


def actuator_moments(sim: Any) -> np.ndarray:
    num_muscles = int(sim.model.nu)
    num_joints = int(sim.model.nv)
    if hasattr(sim.data, "actuator_moment"):
        moments = np.asarray(sim.data.actuator_moment, dtype=np.float32)
    else:
        moments = np.asarray(
            getattr(sim.data, "ten_moment", np.zeros(num_muscles * num_joints, dtype=np.float32)),
            dtype=np.float32,
        )
    if moments.size == num_muscles * num_joints:
        moments = moments.reshape(num_muscles, num_joints)
    if moments.ndim == 1:
        padded = np.zeros((num_muscles, num_joints), dtype=np.float32)
        padded.reshape(-1)[: min(padded.size, moments.size)] = moments.reshape(-1)[: min(padded.size, moments.size)]
        moments = padded
    return moments


def extract_env_data(env: Any, action: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sim = sim_from_env(env)
    return (
        actuator_priors(sim),
        muscle_states(sim),
        actuator_moments(sim),
        np.asarray(action, dtype=np.float32).copy(),
    )


def save_expert_dataset(
    output_path: str | Path,
    priors: list[np.ndarray],
    states: list[np.ndarray],
    moments: list[np.ndarray],
    actions: list[np.ndarray],
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "priors": torch.tensor(np.asarray(priors), dtype=torch.float32),
            "states": torch.tensor(np.asarray(states), dtype=torch.float32),
            "moments": torch.tensor(np.asarray(moments), dtype=torch.float32),
            "actions": torch.tensor(np.asarray(actions), dtype=torch.float32),
        },
        path,
    )
    return path
