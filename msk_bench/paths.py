"""Portable repository-layout helpers for MSK-Bench integrations."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RL_PARADIGMS_DIR = REPO_ROOT / "rl_paradigms"


def paradigm_path(name: str, *parts: str) -> Path:
    """Return a path within one named RL paradigm directory."""
    if not name or Path(name).name != name:
        raise ValueError(f"Invalid paradigm name: {name!r}")
    return RL_PARADIGMS_DIR.joinpath(name, *parts)
