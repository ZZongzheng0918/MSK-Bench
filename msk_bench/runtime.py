"""Small platform compatibility helpers used by environments and evaluators."""

from __future__ import annotations

import os
import sys


def configure_mujoco_gl(platform_name: str | None = None) -> str:
    """Select a portable MuJoCo GL backend without overriding user configuration."""
    platform_name = sys.platform if platform_name is None else platform_name
    default = "glfw" if platform_name.startswith("win") or platform_name == "darwin" else "egl"
    return os.environ.setdefault("MUJOCO_GL", default)


def normalize_camera_id(camera_id: int | None) -> int:
    """Return a concrete camera id accepted by every repository renderer."""
    return 0 if camera_id is None else int(camera_id)
