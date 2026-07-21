from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / "MSK-Bench", _REPO_ROOT / "depRL"):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    import gymnasium as gym
    from gymnasium.envs.registration import register
except ModuleNotFoundError:
    gym = None
    register = None

try:
    import msk_bench  # noqa: F401
    from msk_bench.registry import CANONICAL_TASKS
except ModuleNotFoundError:
    msk_bench = None
    CANONICAL_TASKS = ()


@dataclass(frozen=True)
class TaskSpec:
    env_id: str
    slug: str
    mode: str
    steps: int

    @property
    def middleware_env_id(self) -> str:
        return self.env_id.replace("-v0", "-Middleware-v0")


MODE_BY_ENV_ID = {
    "MSKBenchBalance-v0": "residual",
    "MSKBenchDoorOpen-v0": "residual",
    "MSKBenchChinUp-v0": "residual",
    "MSKBenchReach-v0": "primate_bimanual",
    "MSKBenchCatch-v0": "primate_bimanual",
}

MSK_BENCH_TASKS: tuple[TaskSpec, ...] = tuple(
    TaskSpec(task.env_id, task.slug, MODE_BY_ENV_ID.get(task.env_id, "hard"), task.horizon)
    for task in CANONICAL_TASKS
)


def env_slug(env_id: str) -> str:
    name = env_id.removeprefix("MSKBench").removesuffix("-v0")
    return re.sub(r"(?<!^)([A-Z])", r"_\1", name).lower()


def spec_for_env(env_id: str) -> TaskSpec:
    for spec in MSK_BENCH_TASKS:
        if env_id in {spec.env_id, spec.middleware_env_id, spec.slug}:
            return spec
    raise KeyError(f"Unknown MSK-Bench task: {env_id}")


def make_middleware_env(**kwargs: Any):
    if gym is None:
        raise ModuleNotFoundError("gymnasium is required to build middleware environments.")
    from .wrapper import BioMiddlewareWrapper

    base_env_id = kwargs.pop("base_env_id")
    latent_dim = kwargs.pop("latent_dim", 64)
    mode = kwargs.pop("mode", spec_for_env(base_env_id).mode)
    encoder_path = kwargs.pop("encoder_path", None)
    decoder_path = kwargs.pop("decoder_path", None)
    strict_weights = kwargs.pop("strict_weights", False)
    residual_scale = kwargs.pop("residual_scale", 1.0)
    tube_radius = kwargs.pop("tube_radius", 0.25)
    lower_body_tube_radius = kwargs.pop("lower_body_tube_radius", 0.10)
    lower_body_cutoff = kwargs.pop("lower_body_cutoff", 290)
    penalty_scale = kwargs.pop("penalty_scale", 5.0)
    decay_steps = kwargs.pop("decay_steps", 2_000_000)
    device = kwargs.pop("device", None)
    env = gym.make(base_env_id, **kwargs)
    return BioMiddlewareWrapper(
        env,
        latent_dim=latent_dim,
        mode=mode,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        strict_weights=strict_weights,
        residual_scale=residual_scale,
        tube_radius=tube_radius,
        lower_body_tube_radius=lower_body_tube_radius,
        lower_body_cutoff=lower_body_cutoff,
        penalty_scale=penalty_scale,
        decay_steps=decay_steps,
        device=device,
    )


def register_all() -> None:
    if gym is None or register is None:
        return
    for spec in MSK_BENCH_TASKS:
        if spec.middleware_env_id in gym.envs.registry:
            continue
        register(
            id=spec.middleware_env_id,
            entry_point="deprl_middleware_22tasks.registry:make_middleware_env",
            max_episode_steps=spec.steps,
            kwargs={
                "base_env_id": spec.env_id,
                "latent_dim": 64,
                "mode": spec.mode,
            },
        )


register_all()
