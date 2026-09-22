"""Robustness perturbation families used by MSK-Bench."""

PERTURBATION_FAMILIES = ("action_noise", "observation_noise", "dynamics_randomization")
ACTION_NOISE_SIGMAS = (0.0, 0.05, 0.1, 0.15, 0.2)
OBSERVATION_NOISE_SIGMAS = (0.0, 0.02, 0.05, 0.08, 0.10)
DYNAMICS_RANDOMIZATION_SIGMAS = (0.0, 0.05, 0.1, 0.15, 0.2)


def randomize_muscle_strength(model, scale, *, rng=None):
    """Multiplicative uniform strength perturbation of MuJoCo muscle actuators.

    gainprm/biasprm[2] is force; [3] is scale for automatically inferred force.
    [0:2] is length range and must not be used as a strength parameter.
    Apply once to a fresh model for each sweep point.
    """
    import numpy as np

    scale = float(scale)
    if not np.isfinite(scale) or not 0 <= scale < 1:
        raise ValueError("Dynamics perturbation scale must be in [0, 1)")
    muscles = np.flatnonzero(np.asarray(model.actuator_gaintype) == 2)  # mjGAIN_MUSCLE
    if not len(muscles):
        raise ValueError("Dynamics perturbation requires MuJoCo muscle actuators")
    generator = rng if rng is not None else np.random
    factors = generator.uniform(1 - scale, 1 + scale, size=len(muscles))
    for index, factor in zip(muscles, factors):
        for parameters in (model.actuator_gainprm, model.actuator_biasprm):
            column = 2 if parameters[index, 2] >= 0 else 3
            parameters[index, column] *= factor
    return factors
