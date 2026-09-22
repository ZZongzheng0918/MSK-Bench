"""Debug/profiling tools for training."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import jax

from musclemimic.core.wrappers import LogEnvState


@dataclass
class DebugFlags:
    enabled: bool = False
    profile_traj_batch: bool = False
    profile_val_batch: bool = False
    track_nacon: bool = False
    _max_nacon_seen: int = field(default=0, repr=False)
    _profile_done: bool = field(default=False, repr=False)

    @staticmethod
    def from_config(cfg_debug: bool | dict | Any) -> DebugFlags:
        if isinstance(cfg_debug, bool):
            flags = DebugFlags(enabled=cfg_debug)
        elif hasattr(cfg_debug, "get"):
            flags = DebugFlags(
                enabled=cfg_debug.get("enabled", False),
                profile_traj_batch=cfg_debug.get("profile_traj_batch", False),
                profile_val_batch=cfg_debug.get("profile_val_batch", False),
                track_nacon=cfg_debug.get("track_nacon", False),
            )
        else:
            flags = DebugFlags(enabled=bool(cfg_debug))

        # Env var overrides (only affect their own flag, not enabled)
        if os.environ.get("PROFILE_TRAJ_BATCH", "0") == "1":
            flags.profile_traj_batch = True
        if os.environ.get("PROFILE_VAL_BATCH", "0") == "1":
            flags.profile_val_batch = True
        if os.environ.get("TRACK_NCON", "0") == "1":
            flags.track_nacon = True
        return flags


def maybe_debug_callback(env_state: Any, config: Any, flags: DebugFlags) -> None:
    if not flags.enabled:
        return

    def callback(metrics):
        returns = metrics.returned_episode_returns[metrics.done]
        steps = metrics.timestep[metrics.done] * config.num_envs
        for t in range(len(steps)):
            print(f"global step={steps[t]}, episodic return={returns[t]}")

    jax.debug.callback(callback, env_state.find(LogEnvState).metrics)


def _key_str(k):
    """Extract string from JAX key path element."""
    if hasattr(k, "key"):
        return str(k.key)
    if hasattr(k, "idx"):
        return str(k.idx)
    return str(k)


def maybe_profile_traj_batch(traj_batch: Any, flags: DebugFlags) -> None:
    if not flags.profile_traj_batch or flags._profile_done:
        return

    def _profile(tb):
        total = [0]
        print("\n" + "=" * 50 + "\n[PROFILE] traj_batch:")
        def visit(p, x):
            if hasattr(x, "nbytes"):
                print(f"  {'.'.join(_key_str(k) for k in p)}: {x.shape} = {x.nbytes/1e6:.1f}MB")
                total[0] += x.nbytes
        jax.tree_util.tree_map_with_path(visit, tb)
        print(f"  TOTAL: {total[0]/1e9:.2f}GB\n" + "=" * 50)

    jax.debug.callback(_profile, traj_batch)
    flags._profile_done = True


def maybe_profile_val_batch(val_batch: Any, K: int, flags: DebugFlags) -> None:
    if not flags.profile_val_batch or flags._profile_done:
        return

    def _profile(tb, k):
        total = [0]
        print(f"\n" + "=" * 50 + f"\n[PROFILE] val_batch (K={k}):")
        def visit(p, x):
            if hasattr(x, "nbytes"):
                print(f"  {'.'.join(_key_str(kk) for kk in p)}: {x.shape} = {x.nbytes/1e6:.1f}MB")
                total[0] += x.nbytes
        jax.tree_util.tree_map_with_path(visit, tb)
        print(f"  TOTAL: {total[0]/1e9:.2f}GB\n" + "=" * 50)

    jax.debug.callback(_profile, val_batch, K)
    flags._profile_done = True


def maybe_track_nacon(data_impl: Any, flags: DebugFlags) -> None:
    if not flags.track_nacon or not hasattr(data_impl, "nacon"):
        return

    def _track(nacon):
        mx = int(nacon.max()) if hasattr(nacon, "max") else int(nacon)
        if mx > flags._max_nacon_seen:
            flags._max_nacon_seen = mx
            print(f"[NACON] new max: {mx}")

    jax.debug.callback(_track, data_impl.nacon)
