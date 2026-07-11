import sys
import types

import numpy as np

from loco_mujoco.core.visuals.video_recorder import VideoRecorder


def test_stop_is_idempotent(tmp_path, monkeypatch):
    release_calls = {"n": 0}

    class DummyWriter:
        def write(self, _frame):
            return None

        def release(self):
            release_calls["n"] += 1
            return None

    cv2 = types.ModuleType("cv2")
    cv2.COLOR_RGB2BGR = 0
    cv2.cvtColor = lambda frame, _code: frame
    cv2.VideoWriter_fourcc = lambda *_args: 0
    cv2.VideoWriter = lambda *_args, **_kwargs: DummyWriter()
    cv2.destroyAllWindows = lambda: None
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    recorder = VideoRecorder(
        path=str(tmp_path),
        tag="test",
        video_name="recording",
        fps=30,
        compress=False,
    )
    recorder(np.zeros((8, 8, 3), dtype=np.uint8))

    path1 = recorder.stop()
    path2 = recorder.stop()

    assert path1 is not None
    assert path2 == path1
    assert release_calls["n"] == 1


def test_stop_without_frames_returns_none(tmp_path):
    recorder = VideoRecorder(
        path=str(tmp_path),
        tag="test",
        video_name="recording",
        fps=30,
        compress=False,
    )
    assert recorder.stop() is None

