"""Deterministic constraints for reward-tuning proposals; no API calls."""

import math
import numpy as np


def normalize_weights(weights, l1_norm):
    values = {key: float(value) for key, value in weights.items()}
    if not values or not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Weights must be nonempty and finite")
    if l1_norm is None:
        return values
    if not math.isfinite(l1_norm) or l1_norm <= 0:
        raise ValueError("l1_norm must be finite and positive")
    mass = sum(abs(v) for v in values.values())
    if mass == 0:
        raise ValueError("Cannot normalize all-zero weights")
    return {key: value * l1_norm / mass for key, value in values.items()}


def constrain_weights(current, proposed, *, l1_norm=None, relative_limit=.2):
    """Clip a proposal; optionally project onto fixed-L1 and relative-change bounds.

    In fixed-norm mode all keys (including penalties) participate; zero weights
    stay zero. Projection on signed magnitudes preserves signs. The current
    vector must already have the requested norm, making the box feasible.
    """
    keys = list(current)
    old = np.array([float(current[k]) for k in keys])
    candidate = np.array([float(proposed.get(k, current[k])) for k in keys])
    if not keys or not np.isfinite(old).all() or not np.isfinite(candidate).all():
        raise ValueError("Reward weights must be nonempty and finite")
    if not 0 <= relative_limit < 1:
        raise ValueError("relative_limit must be in [0, 1)")
    delta = np.abs(old) * relative_limit
    if l1_norm is None:
        bounded = np.clip(candidate, old - delta, old + delta)
        # Legacy mode allows activating a previously zero-weight term.
        bounded[old == 0] = candidate[old == 0]
    else:
        if not math.isfinite(l1_norm) or l1_norm <= 0 or not np.isclose(np.abs(old).sum(), l1_norm):
            raise ValueError("Initialize current weights to the requested L1 norm first")
        lower, upper = np.abs(old) - delta, np.abs(old) + delta
        magnitude = candidate * np.sign(old)
        lo, hi = float(np.min(magnitude - upper)), float(np.max(magnitude - lower))
        for _ in range(100):
            midpoint = (lo + hi) / 2
            if np.clip(magnitude - midpoint, lower, upper).sum() > l1_norm:
                lo = midpoint
            else:
                hi = midpoint
        bounded = np.sign(old) * np.clip(magnitude - (lo + hi) / 2, lower, upper)
    return dict(zip(keys, bounded.tolist()))
