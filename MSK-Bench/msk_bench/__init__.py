"""MSK-Bench package entry point."""

from typing import List

from .utils import gym
from .utils.implement_for import implement_for
from .version import __version_tuple__


@implement_for("gym", None, "0.24")
def gym_registry_specs():
    return gym.envs.registry.env_specs


@implement_for("gym", "0.24", None)
def gym_registry_specs():
    return gym.envs.registry


@implement_for("gymnasium")
def gym_registry_specs():
    return gym.envs.registry


_existing_envs = set(gym_registry_specs().keys())
from .envs.msk import benchmark  # noqa: E402,F401

msk_bench_env_suite = sorted(set(gym_registry_specs().keys()) - _existing_envs)
msk_bench_tasks = msk_bench_env_suite

__version__ = ".".join(str(x) for x in __version_tuple__)
__all__: List[str] = ["gym", "gym_registry_specs", "msk_bench_env_suite", "msk_bench_tasks"]
