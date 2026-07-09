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
except ModuleNotFoundError:
    msk_bench = None



@dataclass(frozen=True)
class TaskSpec:
    env_id: str
    slug: str
    mode: str
    steps: int

    @property
    def middleware_env_id(self) -> str:
        return self.env_id.replace("-v0", "-Middleware-v0")


MSK_BENCH_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("MSKBenchStand-v0", "stand", "hard", 1000),
    TaskSpec("MSKBenchPowerlift-v0", "powerlift", "hard", 900),
    TaskSpec("MSKBenchSingleLegStand-v0", "single_leg_stand", "hard", 1000),
    TaskSpec("MSKBenchSit-v0", "sit", "hard", 1000),
    TaskSpec("MSKBenchBalance-v0", "balance", "residual", 1000),
    TaskSpec("MSKBenchSquat-v0", "squat", "hard", 1000),
    TaskSpec("MSKBenchWalk-v0", "walk", "hard", 1000),
    TaskSpec("MSKBenchCrawl-v0", "crawl", "hard", 1000),
    TaskSpec("MSKBenchRun-v0", "run", "hard", 1000),
    TaskSpec("MSKBenchJump-v0", "jump", "hard", 1000),
    TaskSpec("MSKBenchWalkTurn-v0", "walk_turn", "hard", 2000),
    TaskSpec("MSKBenchSidestep-v0", "sidestep", "hard", 1000),
    TaskSpec("MSKBenchStairs-v0", "stairs", "hard", 2000),
    TaskSpec("MSKBenchHurdle-v0", "hurdle", "hard", 1000),
    TaskSpec("MSKBenchStepStones-v0", "step_stones", "hard", 1000),
    TaskSpec("MSKBenchSlide-v0", "slide", "hard", 1500),
    TaskSpec("MSKBenchDoorOpen-v0", "door_open", "residual", 1000),
    TaskSpec("MSKBenchReach-v0", "reach", "primate_bimanual", 500),
    TaskSpec("MSKBenchWalkAndSit-v0", "walk_and_sit", "hard", 500),
    TaskSpec("MSKBenchChinUp-v0", "chin_up", "residual", 500),
    TaskSpec("MSKBenchCatch-v0", "catch", "primate_bimanual", 200),
    TaskSpec("MSKBenchPoleWalk-v0", "pole_walk", "hard", 1000),
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
