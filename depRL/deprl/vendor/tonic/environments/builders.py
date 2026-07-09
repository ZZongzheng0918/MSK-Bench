"""Environment builders for popular domains."""

import importlib.util
import os
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from gymnasium import wrappers

try:
    import msk_bench  # noqa: F401 - registers MSK-Bench Gymnasium environments.
except ModuleNotFoundError:
    msk_bench = None

if importlib.util.find_spec("myosuite") is not None:
    import myosuite  # noqa: F401
    import myosuite.envs  # noqa: F401

from deprl.vendor.tonic import environments
from deprl.vendor.tonic.utils import logger


def gym_environment(*args, **kwargs):
    """Returns a wrapped Gymnasium environment."""
    if "header" in kwargs:
        kwargs.pop("header")

    def _builder(*args, **kwargs):
        env = gym.make(*args, **kwargs)
        if len(args) > 0 and isinstance(args[0], str):
            env.name = args[0]
        elif "id" in kwargs:
            env.name = kwargs["id"]
        else:
            env.name = "gym_env_unknown"
        return env

    return build_environment(_builder, *args, **kwargs, header=None)


def bullet_environment(*args, **kwargs):
    """Returns a wrapped PyBullet environment."""
    if "header" in kwargs:
        kwargs.pop("header")

    def _builder(*args, **kwargs):
        import pybullet_envs  # noqa: F401

        env = gym.make(*args, **kwargs)
        if len(args) > 0:
            env.name = args[0]
        return env

    return build_environment(_builder, *args, **kwargs, header=None)


def control_suite_environment(*args, **kwargs):
    """Returns a wrapped Control Suite environment."""
    if "header" in kwargs:
        kwargs.pop("header")

    def _builder(name, *args, **kwargs):
        domain, task = name.split("-")
        environment = ControlSuiteEnvironment(
            domain_name=domain, task_name=task, *args, **kwargs
        )
        time_limit = int(environment.environment._step_limit)
        environment.spec = SimpleNamespace(
            max_episode_steps=time_limit, id="ostrichrl-dmcontrol"
        )
        return wrappers.TimeLimit(environment, time_limit)

    return build_environment(_builder, *args, **kwargs, header=None)


def build_environment(
    builder,
    name,
    terminal_timeouts=False,
    time_feature=False,
    max_episode_steps="default",
    scaled_actions=True,
    header=None,
    *args,
    **kwargs,
):
    """Builds and wraps an environment."""
    if header is not None:
        exec(header)

    environment = builder(name, *args, **kwargs)

    if max_episode_steps == "default":
        if hasattr(environment, "spec") and environment.spec is not None:
            max_episode_steps = environment.spec.max_episode_steps
        elif hasattr(environment, "_max_episode_steps"):
            max_episode_steps = environment._max_episode_steps
        elif hasattr(environment, "horizon"):
            max_episode_steps = environment.horizon
        elif hasattr(environment, "max_episode_steps"):
            max_episode_steps = environment.max_episode_steps
        else:
            logger.log("No max episode steps found, setting them to 1000")
            max_episode_steps = 1000

    if not terminal_timeouts and isinstance(environment, wrappers.TimeLimit):
        environment = environment.env

    if time_feature:
        environment = environments.wrappers.TimeFeature(
            environment, max_episode_steps
        )

    if scaled_actions:
        environment = environments.wrappers.ActionRescaler(environment)

    environment.name = name
    environment.max_episode_steps = max_episode_steps
    return environment


def _flatten_observation(observation):
    """Turns OrderedDict observations into vectors."""
    observation = [
        np.array([o]) if np.isscalar(o) else o.ravel()
        for o in observation.values()
    ]
    return np.concatenate(observation, axis=0)


class ControlSuiteEnvironment(gym.core.Env):
    """Turns a Control Suite environment into a Gym environment."""

    def __init__(
        self,
        domain_name,
        task_name,
        task_kwargs=None,
        visualize_reward=True,
        environment_kwargs=None,
    ):
        from dm_control import suite
        from dm_control.rl.control import PhysicsError

        self.environment = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs,
        )
        observation_spec = self.environment.observation_spec()
        dim = sum(int(np.prod(spec.shape)) for spec in observation_spec.values())
        high = np.full(dim, np.inf, np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        action_spec = self.environment.action_spec()
        self.action_space = gym.spaces.Box(
            action_spec.minimum, action_spec.maximum, dtype=np.float32
        )
        self.error = PhysicsError

    def seed(self, seed):
        self.environment.task._random = np.random.RandomState(seed)

    def step(self, action):
        try:
            time_step = self.environment.step(action)
            observation = _flatten_observation(time_step.observation)
            reward = time_step.reward
            done = time_step.last()
            if done:
                done = self.environment.task.get_termination(
                    self.environment.physics
                )
                done = done is not None
            self.last_time_step = time_step
        except self.error as e:
            path = logger.get_path()
            os.makedirs(path, exist_ok=True)
            save_path = os.path.join(path, "crashes.txt")
            error = str(e)
            with open(save_path, "a", encoding="utf-8") as file:
                file.write(error + "\n")
            logger.error(error)
            observation = _flatten_observation(self.last_time_step.observation)
            observation = np.zeros_like(observation)
            reward = 0.0
            done = True
        return observation, reward, done, {}

    def reset(self):
        time_step = self.environment.reset()
        self.last_time_step = time_step
        return _flatten_observation(time_step.observation)

    def render(self, mode="rgb_array", height=None, width=None, camera_id=0):
        assert mode == "rgb_array"
        return self.environment.physics.render(
            height=height, width=width, camera_id=camera_id
        )


Gym = gym_environment
Bullet = bullet_environment
ControlSuite = control_suite_environment