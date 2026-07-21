"""Robustness perturbation families used by MSK-Bench."""

PERTURBATION_FAMILIES = ("action_noise", "observation_noise", "dynamics_randomization")
ACTION_NOISE_SIGMAS = (0.0, 0.05, 0.1, 0.15, 0.2)
OBSERVATION_NOISE_SIGMAS = (0.0, 0.02, 0.05, 0.08, 0.10)
DYNAMICS_RANDOMIZATION_SIGMAS = (0.0, 0.05, 0.1, 0.15, 0.2)