"""MuscleMimic utilities - custom implementations"""

from .utd import compute_utd
from .metrics import MetricsHandler, ValidationSummary, QuantityContainer
from .logging import setup_logger, TimestepTracker
from .display import (
    detect_headless_environment,
    setup_headless_rendering,
    setup_headless_rendering_if_needed,
)
from .model import count_actor_critic_params, count_params, count_params_by_path, count_trainable_params

# Re-export loco_mujoco utilities for convenience
from loco_mujoco.utils import (
    set_all_caches,
    set_amass_path,
    set_converted_amass_path,
    set_converted_lafan1_path,
    set_lafan1_path,
    set_smpl_model_path,
)

__all__ = [
    "MetricsHandler",
    "QuantityContainer",
    "TimestepTracker",
    "ValidationSummary",
    # MuscleMimic custom utilities
    "compute_utd",
    "count_actor_critic_params",
    "count_params",
    "count_params_by_path",
    "count_trainable_params",
    "detect_headless_environment",
    "download_gmr_dataset_group",
    # LocoMuJoCo utilities
    "set_all_caches",
    "set_amass_path",
    "set_converted_amass_path",
    "set_converted_lafan1_path",
    "set_lafan1_path",
    "set_smpl_model_path",
    "setup_headless_rendering",
    "setup_headless_rendering_if_needed",
    "setup_logger",
]


def download_gmr_dataset_group(*args, **kwargs):
    from .gmr_cache import download_gmr_dataset_group as _download_gmr_dataset_group

    return _download_gmr_dataset_group(*args, **kwargs)
