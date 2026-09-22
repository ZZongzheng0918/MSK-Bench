"""Acceptance checks for evaluator tables and actual encoded videos."""

import json
import sys

import numpy as np
import pytest


@pytest.mark.parametrize("suffix,content", [
    ("json", "[]"), ("json", "{}"), ("csv", "reward,steps\n"),
    ("json", '[{"reward": NaN}]'), ("csv", "reward,steps\ninf,2\n"),
])
def test_reject_empty_or_nonfinite_tables(tmp_path, suffix, content):
    from msk_bench.validation import validate_table

    path = tmp_path / f"invalid.{suffix}"
    path.write_text(content)
    with pytest.raises(ValueError):
        validate_table(path)


def test_written_tables_are_parseable(tmp_path):
    from benchmark_eval.common import write_rows
    from msk_bench.validation import validate_table

    rows = [{"env_id": "MSKBenchSquat-v0", "steps": 2, "reward": 1.5}]
    paths = [tmp_path / "result.json", tmp_path / "result.csv"]
    write_rows(rows, *paths)
    for path in paths:
        assert validate_table(path) == 1
    assert json.loads(paths[0].read_text()) == rows
    with pytest.raises(ValueError):
        write_rows([], tmp_path / "empty.json", None)


def test_video_is_encoded_and_decodable(tmp_path):
    from benchmark_eval.common import save_video
    from msk_bench.validation import validate_video

    frames = [np.full((32, 48, 3), value, np.uint8) for value in (20, 100)]
    path = tmp_path / "real.mp4"
    assert save_video(frames, path, 10) == path
    assert validate_video(path) == (32, 48, 3)


def test_reject_undecodable_video(tmp_path):
    from msk_bench.validation import validate_video

    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="decode"):
        validate_video(path)


def test_unified_evaluator_uses_current_interpreter():
    from benchmark_eval.evaluate import EvaluationRequest, build_command

    assert build_command(EvaluationRequest(algorithms=("ppo",)), "ppo")[0] == sys.executable
