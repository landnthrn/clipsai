from pathlib import Path

import pytest

from clipsai.reframe import PLAN_FILE_SUFFIX
from clipsai.reframe import build_plan
from clipsai.reframe import build_render_media_file
from clipsai.reframe import default_output_path
from clipsai.reframe import default_plan_path
from clipsai.reframe import discover_plan_files
from clipsai.reframe import discover_video_files
from clipsai.resize.crops import Crops
from clipsai.resize.segment import Segment


def test_discover_video_files_from_directory(tmp_path: Path):
    (tmp_path / "b.mp4").write_text("x", encoding="utf-8")
    (tmp_path / "a.mov").write_text("x", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")

    results = discover_video_files(tmp_path)

    assert [file_path.name for file_path in results] == ["a.mov", "b.mp4"]


def test_discover_plan_files_from_directory(tmp_path: Path):
    (tmp_path / f"one{PLAN_FILE_SUFFIX}").write_text("{}", encoding="utf-8")
    (tmp_path / f"two{PLAN_FILE_SUFFIX}").write_text("{}", encoding="utf-8")
    (tmp_path / "ignore.json").write_text("{}", encoding="utf-8")

    results = discover_plan_files(tmp_path)

    assert [file_path.name for file_path in results] == [
        f"one{PLAN_FILE_SUFFIX}",
        f"two{PLAN_FILE_SUFFIX}",
    ]


def test_build_plan_contains_expected_shape(tmp_path: Path):
    video_path = tmp_path / "podcast.mp4"
    crops = Crops(
        original_width=1920,
        original_height=1080,
        crop_width=608,
        crop_height=1080,
        segments=[
            Segment(speakers=[0], start_time=0.0, end_time=2.5, x=100, y=0),
            Segment(speakers=[1], start_time=2.5, end_time=5.0, x=900, y=0),
        ],
    )

    plan = build_plan(
        video_path=video_path,
        crops=crops,
        aspect_ratio=(9, 16),
        output_width=1080,
        output_height=1920,
        render_preset="high",
        analysis_settings={"min_segment_duration": 0.75},
    )

    assert plan["source_filename"] == "podcast.mp4"
    assert plan["analysis"]["crop_width"] == 608
    assert plan["render"]["preset_name"] == "high"
    assert plan["segments"][0]["speakers"] == [0]
    assert plan["segments"][1]["x"] == 900


def test_default_paths(tmp_path: Path):
    source_path = tmp_path / "episode01.mp4"
    plan_path = default_plan_path(source_path, tmp_path / "plans")
    output_path = default_output_path(source_path, tmp_path / "output")

    assert plan_path.name == f"episode01{PLAN_FILE_SUFFIX}"
    assert output_path.name == "episode01_vertical.mp4"


def test_build_render_media_file_uses_audiovideo_for_sources_with_audio(tmp_path: Path):
    source_path = tmp_path / "episode01.mp4"
    source_path.write_text("x", encoding="utf-8")

    class FakeTemporalMediaFile:
        def __init__(self, path: str):
            self.path = path

        def assert_exists(self) -> None:
            return None

        def has_video_stream(self) -> bool:
            return True

        def has_audio_stream(self) -> bool:
            return True

    class FakeVideoFile:
        def __init__(self, path: str):
            self.path = path

    class FakeAudioVideoFile:
        def __init__(self, path: str):
            self.path = path

    media_file = build_render_media_file(
        source_path,
        temporal_media_cls=FakeTemporalMediaFile,
        video_file_cls=FakeVideoFile,
        audiovideo_file_cls=FakeAudioVideoFile,
    )

    assert isinstance(media_file, FakeAudioVideoFile)
    assert media_file.path == str(source_path)


def test_build_render_media_file_uses_video_file_for_video_only_sources(tmp_path: Path):
    source_path = tmp_path / "episode01.mp4"
    source_path.write_text("x", encoding="utf-8")

    class FakeTemporalMediaFile:
        def __init__(self, path: str):
            self.path = path

        def assert_exists(self) -> None:
            return None

        def has_video_stream(self) -> bool:
            return True

        def has_audio_stream(self) -> bool:
            return False

    class FakeVideoFile:
        def __init__(self, path: str):
            self.path = path

    class FakeAudioVideoFile:
        def __init__(self, path: str):
            self.path = path

    media_file = build_render_media_file(
        source_path,
        temporal_media_cls=FakeTemporalMediaFile,
        video_file_cls=FakeVideoFile,
        audiovideo_file_cls=FakeAudioVideoFile,
    )

    assert isinstance(media_file, FakeVideoFile)
    assert media_file.path == str(source_path)


def test_build_render_media_file_rejects_sources_without_video(tmp_path: Path):
    source_path = tmp_path / "episode01.mp4"
    source_path.write_text("x", encoding="utf-8")

    class FakeTemporalMediaFile:
        def __init__(self, path: str):
            self.path = path

        def assert_exists(self) -> None:
            return None

        def has_video_stream(self) -> bool:
            return False

        def has_audio_stream(self) -> bool:
            return True

    class FakeVideoFile:
        def __init__(self, path: str):
            self.path = path

    class FakeAudioVideoFile:
        def __init__(self, path: str):
            self.path = path

    with pytest.raises(ValueError, match="does not contain a video stream"):
        build_render_media_file(
            source_path,
            temporal_media_cls=FakeTemporalMediaFile,
            video_file_cls=FakeVideoFile,
            audiovideo_file_cls=FakeAudioVideoFile,
        )
