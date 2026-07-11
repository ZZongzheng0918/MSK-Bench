"""Public MuscleMimic integration surface for MSK-Bench."""

from __future__ import annotations

import os
from functools import lru_cache
from types import SimpleNamespace

import numpy as np


CHECKPOINT_ENV_VAR = "MSK_BENCH_MUSCLEMIMIC_CHECKPOINT"
DEFAULT_CHECKPOINT_SOURCE = "hf://amathislab/mm-fullbody-base"


def resolve_checkpoint_source(value: str | os.PathLike | None = None) -> str:
    """Resolve an explicit, environment-configured, or official checkpoint source."""
    explicit = "" if value is None else str(value).strip()
    if explicit:
        return explicit
    configured = os.environ.get(CHECKPOINT_ENV_VAR, "").strip()
    return configured or DEFAULT_CHECKPOINT_SOURCE


def validate_policy_observation(observation, expected_dim: int) -> np.ndarray:
    """Return a flat policy observation after checking its expected dimension."""
    value = np.asarray(observation, dtype=np.float32).reshape(-1)
    if value.size != expected_dim:
        raise ValueError(
            f"MuscleMimic policy expected observation dimension {expected_dim}, got {value.size}"
        )
    return value


def validate_policy_action(action, expected_dim: int) -> np.ndarray:
    """Return a flat policy action after checking its expected dimension."""
    value = np.asarray(action, dtype=np.float32).reshape(-1)
    if value.size != expected_dim:
        raise ValueError(f"MuscleMimic policy expected action dimension {expected_dim}, got {value.size}")
    return value


@lru_cache(maxsize=None)
def get_jax_policy(checkpoint_source: str | None, expected_obs_dim: int, act_dim: int):
    """Load and cache a validated JAX PPO inference function and train state."""
    import jax
    import jax.numpy as jnp
    from gymnasium import spaces
    from omegaconf import OmegaConf

    from musclemimic.algorithms.ppo import PPOJax
    from musclemimic.runner.eval_utils import align_agent_state, load_checkpoint

    source = resolve_checkpoint_source(checkpoint_source)
    config, agent_state, _ = load_checkpoint(source)
    OmegaConf.set_struct(config, False)

    class DummyEnv:
        def __init__(self):
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(expected_obs_dim,),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(act_dim,),
                dtype=np.float32,
            )
            self.mdp_info = SimpleNamespace(
                observation_space=self.observation_space,
                action_space=self.action_space,
            )
            self.info = self.mdp_info

    agent_conf = PPOJax.init_agent_conf(DummyEnv(), config)
    train_state = align_agent_state(agent_state, agent_conf).train_state

    @jax.jit
    def apply_policy(state, observation):
        variables = {"params": state.params, "run_stats": state.run_stats}
        output, _ = agent_conf.network.apply(
            variables,
            jnp.atleast_2d(observation),
            mutable=["run_stats"],
        )
        return jnp.squeeze(output[0].mean())

    def policy(state, observation):
        checked_observation = validate_policy_observation(observation, expected_obs_dim)
        action = apply_policy(state, checked_observation)
        return validate_policy_action(action, act_dim)

    return policy, train_state


def __getattr__(name: str):
    if name == "MyoFullBody":
        from musclemimic.environments.humanoids import MyoFullBody

        globals()[name] = MyoFullBody
        return MyoFullBody
    raise AttributeError(name)


__all__ = [
    "CHECKPOINT_ENV_VAR",
    "DEFAULT_CHECKPOINT_SOURCE",
    "MyoFullBody",
    "get_jax_policy",
    "resolve_checkpoint_source",
    "validate_policy_action",
    "validate_policy_observation",
]
