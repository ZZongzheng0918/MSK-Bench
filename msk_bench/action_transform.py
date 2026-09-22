from __future__ import annotations

import numpy as np


def canonical_policy_action_to_excitation(action, *, clip: bool = True):
    """Map canonical policy actions in [-1, 1] to muscle excitations in [0, 1]."""
    values = np.asarray(action, dtype=np.float32)
    if clip:
        values = np.clip(values, -1.0, 1.0)
    return ((values + 1.0) * 0.5).astype(np.float32, copy=False)


def excitation_to_canonical_policy_action(excitation, *, clip: bool = True):
    """Map muscle excitations in [0, 1] back to canonical policy actions in [-1, 1]."""
    values = np.asarray(excitation, dtype=np.float32)
    if clip:
        values = np.clip(values, 0.0, 1.0)
    return (values * 2.0 - 1.0).astype(np.float32, copy=False)
