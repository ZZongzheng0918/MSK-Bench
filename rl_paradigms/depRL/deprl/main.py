"""Train depRL agents from a repository YAML/JSON config."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import deprl  # noqa: F401 - namespace for trusted experiment expressions.
import numpy as np
import torch
import yaml

from deprl import custom_distributed
from deprl.utils import load, load_checkpoint
from deprl.vendor.tonic import logger
from rl_paradigms.training_contract import TrainingArtifact


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    for option in ("steps", "epoch-steps", "save-steps", "parallel", "sequential",
                   "seed", "test-episodes", "hidden-size", "batch-size", "buffer-size",
                   "steps-before-batches", "steps-between-batches", "batch-iterations"):
        parser.add_argument("--" + option, type=int)
    parser.add_argument("--output-dir", "--working-dir", type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, int) and not isinstance(value, bool):
            minimum = 0 if key in {"seed", "test_episodes", "steps_before_batches"} else 1
            if value < minimum:
                parser.error(f"--{key.replace('_', '-')} must be >= {minimum}")
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    tonic = config["tonic"]
    for key in ("parallel", "sequential", "seed"):
        if getattr(args, key) is not None:
            tonic[key] = getattr(args, key)
    if args.cpu:
        tonic["cpu_override"] = True
    if args.output_dir is not None:
        config["working_dir"] = str(args.output_dir.resolve())
    trainer_args = config.setdefault("trainer_args", {})
    for key in ("steps", "epoch_steps", "save_steps", "test_episodes"):
        if getattr(args, key) is not None:
            trainer_args[key] = getattr(args, key)
    if args.hidden_size is not None:
        config.setdefault("mpo_args", {})["hidden_size"] = args.hidden_size
    replay_args = config.setdefault("replay_args", {})
    for key in ("batch_size", "steps_before_batches", "steps_between_batches", "batch_iterations"):
        if getattr(args, key) is not None:
            replay_args[key] = getattr(args, key)
    if args.buffer_size is not None:
        replay_args["full_max_size"] = args.buffer_size
    return config


def train(config):
    """Train, verify the saved policy, and always close owned environments."""
    tonic = config["tonic"]
    exec(tonic.get("header", ""))
    env_args = config.get("env_args") or {}
    config["env_args"] = env_args
    with ExitStack() as resources:
        environment = custom_distributed.distribute(tonic["environment"], tonic, env_args)
        resources.callback(environment.close)
        environment.initialize(seed=tonic["seed"])
        test_environment = custom_distributed.distribute(
            tonic.get("test_environment") or tonic["environment"], tonic,
            config.get("test_env_args", env_args), parallel=1, sequential=1,
        )
        resources.callback(test_environment.close)
        test_environment.initialize(seed=tonic["seed"] + 1000000)
        agent = eval(tonic["agent"])
        if "mpo_args" in config:
            agent.set_params(**config["mpo_args"])
        for key, value in config.get("replay_args", {}).items():
            if not hasattr(agent.replay, key):
                raise ValueError(f"Unknown replay setting: {key}")
            setattr(agent.replay, key, value)
        agent.initialize(environment.observation_space, environment.action_space, seed=tonic["seed"])
        if hasattr(agent, "expl") and "DEP" in config:
            agent.expl.set_params(config["DEP"])
        logger.initialize(script_path=__file__, config=config, test_env=test_environment,
                          resume=tonic.get("resume", False))
        run_dir = Path(logger.get_path())
        time_dict = {"steps": 0, "epochs": 0, "episodes": 0}
        if tonic.get("resume"):
            _, checkpoint, loaded_time = load_checkpoint(str(run_dir), checkpoint=tonic.get("checkpoint", "last"))
            if checkpoint:
                time_dict = loaded_time or time_dict
                logger.load(checkpoint, time_dict)
                agent.load(checkpoint)
        trainer = eval(tonic.get("trainer") or "deprl.custom_trainer.Trainer()")
        mapping = {"steps": "max_steps"}
        for key, value in config.get("trainer_args", {}).items():
            setattr(trainer, mapping.get(key, key), value)
        if trainer.max_steps <= 0 or trainer.epoch_steps <= 0 or trainer.save_steps <= 0:
            raise ValueError("Trainer step counts must be positive")
        trainer.initialize(agent, environment, test_environment, full_save=tonic.get("full_save", False))
        if tonic.get("before_training"):
            exec(tonic["before_training"])
        trainer.run(config, **time_dict)
        if tonic.get("after_training"):
            exec(tonic["after_training"])

        # A manifest certifies a real, reloadable checkpoint, never just an exit code.
        restored = load(str(run_dir), test_environment.environments[0])
        for key, value in agent.model.state_dict().items():
            torch.testing.assert_close(restored.model.state_dict()[key], value, rtol=0, atol=0)
        observation, _ = test_environment.start()
        action = restored.test_step(observation, trainer.steps)
        if action.shape != (1, *test_environment.action_space.shape) or not np.isfinite(action).all():
            raise ValueError("Reloaded policy returned invalid actions")
        env_id = test_environment.name
        return TrainingArtifact(
            algorithm="middleware" if "-Middleware-" in env_id else "deprl",
            env_id=env_id, seed=tonic["seed"], timesteps=trainer.steps,
            model_path=run_dir / "checkpoints" / f"step_{trainer.steps}.pt",
        ).write(run_dir)


def set_tensor_device():
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")
    else:
        torch.set_default_device("cpu")


def main(argv=None):
    config = parse_config(argv)
    if config["tonic"].get("cpu_override"):
        torch.set_default_device("cpu")
    else:
        set_tensor_device()
    manifest = train(config)
    print(f"Training manifest: {manifest}")


if __name__ == "__main__":
    main()
