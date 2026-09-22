from .utils import MDPInfo, Box, assert_backend_is_supported
from .stateful_object import StatefulObject, EmptyState
from .observations import ObservationContainer, Observation, ObservationType


def __getattr__(name: str):
    if name == "Mujoco":
        # Lazy import to avoid importing visualization dependencies when not needed.
        from .mujoco_base import Mujoco as _Mujoco

        globals()["Mujoco"] = _Mujoco
        return _Mujoco
    if name in {"MujocoViewer", "VideoRecorder"}:
        from .visuals import MujocoViewer as _MujocoViewer
        from .visuals import VideoRecorder as _VideoRecorder

        globals()["MujocoViewer"] = _MujocoViewer
        globals()["VideoRecorder"] = _VideoRecorder
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
