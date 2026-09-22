"""MuscleMimic: Accelerated Motion Imitation for Biomechanical Models"""

__version__ = "0.1.0"

# Lazy imports to avoid dependency issues during package discovery
__all__ = [
    "set_amass_path",
    "set_smpl_model_path",
    "set_converted_amass_path",
    "set_lafan1_path",
    "set_converted_lafan1_path",
    "set_all_caches",
]

def __getattr__(name):
    """Lazy import for utils functions"""
    if name in __all__:
        from .utils import (
            set_amass_path,
            set_smpl_model_path,
            set_converted_amass_path,
            set_lafan1_path,
            set_converted_lafan1_path,
            set_all_caches,
        )
        globals()[name] = locals()[name]
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
