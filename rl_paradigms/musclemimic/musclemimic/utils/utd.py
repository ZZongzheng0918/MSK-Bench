from __future__ import annotations


def compute_utd(
    num_envs: int, num_steps: int, num_minibatches: int, update_epochs: int, minibatch_size: int | None = None
) -> float:
    """Compute Update-To-Data (UTD) ratio for on-policy PPO-like updates.

    For the common setting where minibatch_size = (num_envs * num_steps) / num_minibatches,
    UTD simplifies to update_epochs. This function computes the general form:

        UTD = (update_epochs * num_minibatches * minibatch_size) / (num_envs * num_steps)

    providing a future-proof calculation if batch partitioning changes.
    """
    assert num_envs > 0 and num_steps > 0 and num_minibatches > 0 and update_epochs > 0
    if minibatch_size is None:
        # Default to the common setting used in this codebase
        assert (num_envs * num_steps) % num_minibatches == 0
        minibatch_size = (num_envs * num_steps) // num_minibatches
    return (update_epochs * num_minibatches * float(minibatch_size)) / float(num_envs * num_steps)
