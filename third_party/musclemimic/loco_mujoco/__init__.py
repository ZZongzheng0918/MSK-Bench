"""
loco_mujoco - MuscleMimic Fork

This package is a heavily modified fork of the original loco-mujoco library.
Many core components have been rewritten or extended.
Using original loco-mujoco package will lead to incompatibilities
with this codebase.

Original: https://github.com/robfiras/loco-mujoco
"""

import os
from pathlib import Path

import yaml

__version__ = '1.0.1'


_PACKAGE_ROOT = Path(__file__).resolve().parent


def get_musclemimic_home() -> Path:
    """Return the base directory for user-scoped MuscleMimic state."""
    home_override = os.environ.get("MUSCLEMIMIC_HOME")
    if home_override:
        return Path(home_override).expanduser()
    return Path.home() / ".musclemimic"


def get_variables_path() -> Path:
    """Return the active path config file.

    The packaged YAML is treated as a read-only template. User overrides live in a
    writable location by default and can be redirected via `MUSCLEMIMIC_CONFIG_PATH`.
    """
    config_override = os.environ.get("MUSCLEMIMIC_CONFIG_PATH")
    if config_override:
        return Path(config_override).expanduser()
    return get_musclemimic_home() / "MUSCLEMIMIC_VARIABLES.yaml"


def load_path_config(path: str | Path | None = None) -> dict:
    """Load the active user path config.

    Returns an empty dict when the config file does not exist or is empty.
    """
    config_path = Path(path).expanduser() if path is not None else get_variables_path()

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except FileNotFoundError:
        return {}

    if not isinstance(data, dict):
        return {}
    return data


# Paths - defaults work without any setup
PATH_TO_VARIABLES_TEMPLATE = _PACKAGE_ROOT / "MUSCLEMIMIC_VARIABLES.yaml"
PATH_TO_VARIABLES = get_variables_path()
PATH_TO_SMPL_ROBOT_CONF = _PACKAGE_ROOT / "smpl" / "robot_confs"

def get_registered_envs():
    from musclemimic.environments import LocoEnv
    return LocoEnv.registered_envs

def __getattr__(name):
    if name == 'Mujoco':
        from .core import Mujoco as _Mujoco
        globals()['Mujoco'] = _Mujoco
        return _Mujoco
    if name in ('TaskFactory', 'RLFactory', 'ImitationFactory'):
        from .task_factories import (
            TaskFactory as _TaskFactory,
            RLFactory as _RLFactory,
            ImitationFactory as _ImitationFactory,
        )
        globals()['TaskFactory'] = _TaskFactory
        globals()['RLFactory'] = _RLFactory
        globals()['ImitationFactory'] = _ImitationFactory
        return {
            'TaskFactory': _TaskFactory,
            'RLFactory': _RLFactory,
            'ImitationFactory': _ImitationFactory,
        }[name]
    raise AttributeError(f"module 'loco_mujoco' has no attribute {name!r}")
