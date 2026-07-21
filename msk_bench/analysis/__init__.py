"""Paper-facing analysis helpers for MSK-Bench."""

from .efficiency import peak_efficiency_steps
from .emg import (
    NORMALIZED_GAIT_POINTS,
    TARGET_TASK,
    best_shifted_pearson,
    mean_emg_similarity,
    minmax_normalize,
    pearson_similarity,
    process_and_align_simulated_emg,
    resample_to_gait_cycle,
)
from .muscle_energy import (
    activation_abs_sum,
    activation_energy,
    mean_squared_activation,
    timestep_activation_energy,
)
from .recruitment import (
    classify_actuator,
    family_activation_mass,
    family_activation_mean,
)
from .smoothness import finite_difference_jerk, log_mean_squared_jerk

__all__ = [
    "NORMALIZED_GAIT_POINTS",
    "TARGET_TASK",
    "activation_abs_sum",
    "activation_energy",
    "best_shifted_pearson",
    "classify_actuator",
    "family_activation_mass",
    "family_activation_mean",
    "finite_difference_jerk",
    "log_mean_squared_jerk",
    "mean_emg_similarity",
    "mean_squared_activation",
    "minmax_normalize",
    "peak_efficiency_steps",
    "pearson_similarity",
    "process_and_align_simulated_emg",
    "resample_to_gait_cycle",
    "timestep_activation_energy",
]
