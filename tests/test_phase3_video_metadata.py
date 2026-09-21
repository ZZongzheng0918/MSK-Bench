from experiments.gpt6_direct_control import analyze_failure


class Reader:
    def count_frames(self):
        return 17


def test_video_frame_count_api_exists():
    assert hasattr(analyze_failure, "video_frame_count")


def test_video_frame_count_falls_back_when_metadata_is_infinite():
    assert analyze_failure.video_frame_count(Reader(), {"nframes": float("inf")}) == 17


def test_video_frame_count_accepts_finite_metadata():
    assert analyze_failure.video_frame_count(Reader(), {"nframes": 12.0}) == 12
