from pathlib import Path

import pytest

from clipsai.reframe import PLAN_FILE_SUFFIX
from clipsai.reframe import PLAN_VERSION
from clipsai.reframe import build_plan
from clipsai.reframe import build_render_media_file
from clipsai.reframe import create_crops_from_plan
from clipsai.reframe import default_output_path
from clipsai.reframe import default_plan_path
from clipsai.reframe import discover_plan_files
from clipsai.reframe import discover_video_files
from clipsai.reframe import get_enabled_segments
from clipsai.reframe import normalize_plan_data
from clipsai.reframe import original_output_filename
from clipsai.reframe import resolve_output_filename
from clipsai.reframe import resolve_render_settings
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

    assert plan["plan_version"] == PLAN_VERSION
    assert plan["source_filename"] == "podcast.mp4"
    assert plan["analysis"]["crop_width"] == 608
    assert plan["render"]["mode"] == "preset"
    assert plan["render"]["preset_name"] == "high"
    assert plan["render"]["output_name_mode"] == "suffix"
    assert plan["render"]["output_suffix"] == "_vertical"
    assert plan["render"]["output_name"] == "podcast_vertical.mp4"
    assert plan["segments"][0]["segment_id"] == "segment_0001"
    assert plan["segments"][0]["enabled"] is True
    assert plan["segments"][0]["speakers"] == [0]
    assert plan["segments"][0]["notes"] == ""
    assert plan["segments"][1]["x"] == 900


def test_default_paths(tmp_path: Path):
    source_path = tmp_path / "episode01.mp4"
    plan_path = default_plan_path(source_path, tmp_path / "plans")
    output_path = default_output_path(source_path, tmp_path / "output")

    assert plan_path.name == f"episode01{PLAN_FILE_SUFFIX}"
    assert output_path.name == "episode01_vertical.mp4"


def test_normalize_plan_data_upgrades_old_plan_shape(tmp_path: Path):
    source_path = tmp_path / "episode01.mp4"
    legacy_plan = {
        "plan_version": 1,
        "source_path": str(source_path),
        "analysis": {
            "original_width": 1920,
            "original_height": 1080,
            "crop_width": 608,
            "crop_height": 1080,
        },
        "render": {
            "preset_name": "high",
            "output_width": 1080,
            "output_height": 1920,
            "video_codec": "libx264",
            "audio_codec": "aac",
            "audio_bitrate": "320k",
            "preset": "slow",
            "crf": "14",
            "scale_flags": "lanczos",
        },
        "segments": [
            {
                "speakers": [0],
                "start_time": 0.0,
                "end_time": 2.5,
                "x": 100,
                "y": 0,
            }
        ],
    }

    normalized = normalize_plan_data(legacy_plan)

    assert normalized["plan_version"] == PLAN_VERSION
    assert normalized["source_filename"] == "episode01.mp4"
    assert normalized["render"]["mode"] == "preset"
    assert normalized["render"]["output_name_mode"] == "suffix"
    assert normalized["render"]["output_suffix"] == "_vertical"
    assert normalized["render"]["output_name"] == "episode01_vertical.mp4"
    assert normalized["render"]["overwrite"] is True
    assert normalized["segments"][0]["segment_id"] == "segment_0001"
    assert normalized["segments"][0]["enabled"] is True
    assert normalized["segments"][0]["notes"] == ""


def test_normalize_plan_data_infers_keep_original_output_mode(tmp_path: Path):
    source_path = tmp_path / "episode01.mov"
    legacy_plan = {
        "plan_version": 1,
        "source_path": str(source_path),
        "analysis": {
            "original_width": 1920,
            "original_height": 1080,
            "crop_width": 608,
            "crop_height": 1080,
        },
        "render": {
            "preset_name": "high",
            "output_name": "episode01.mp4",
        },
        "segments": [
            {
                "speakers": [0],
                "start_time": 0.0,
                "end_time": 2.5,
                "x": 100,
                "y": 0,
            }
        ],
    }

    normalized = normalize_plan_data(legacy_plan)

    assert normalized["render"]["output_name_mode"] == "keep_original"
    assert normalized["render"]["output_suffix"] == ""
    assert normalized["render"]["output_name"] == "episode01.mp4"


def test_resolve_output_filename_supports_suffix_and_keep_original(tmp_path: Path):
    source_path = tmp_path / "episode01.mov"

    suffixed_name = resolve_output_filename(
        source_path,
        {"output_name_mode": "suffix", "output_suffix": "_preview"},
    )
    original_name = resolve_output_filename(
        source_path,
        {"output_name_mode": "keep_original"},
    )

    assert suffixed_name == "episode01_preview.mp4"
    assert original_name == original_output_filename(source_path)


def test_get_enabled_segments_filters_disabled_entries():
    plan_data = {
        "segments": [
            {"segment_id": "segment_0001", "enabled": True},
            {"segment_id": "segment_0002", "enabled": False},
            {"segment_id": "segment_0003", "enabled": True},
        ]
    }

    enabled_segments = get_enabled_segments(plan_data)

    assert [segment["segment_id"] for segment in enabled_segments] == [
        "segment_0001",
        "segment_0003",
    ]


def test_create_crops_from_plan_uses_only_enabled_segments(tmp_path: Path):
    source_path = tmp_path / "episode01.mp4"
    plan_data = {
        "plan_version": PLAN_VERSION,
        "source_path": str(source_path),
        "analysis": {
            "original_width": 1920,
            "original_height": 1080,
            "crop_width": 608,
            "crop_height": 1080,
        },
        "render": {"preset_name": "high"},
        "segments": [
            {
                "segment_id": "segment_0001",
                "enabled": True,
                "speakers": [0],
                "start_time": 0.0,
                "end_time": 2.5,
                "x": 100,
                "y": 0,
            },
            {
                "segment_id": "segment_0002",
                "enabled": False,
                "speakers": [1],
                "start_time": 2.5,
                "end_time": 5.0,
                "x": 900,
                "y": 0,
            },
        ],
    }

    crops = create_crops_from_plan(plan_data)

    assert len(crops.segments) == 1
    assert crops.segments[0].x == 100


def test_resolve_render_settings_uses_cli_preset_override(tmp_path: Path):
    source_path = tmp_path / "episode01.mp4"
    plan_data = {
        "source_path": str(source_path),
        "render": {
            "mode": "preset",
            "preset_name": "high",
            "output_name_mode": "suffix",
            "output_suffix": "_vertical",
            "output_name": "episode01_vertical.mp4",
            "output_width": 1080,
            "output_height": 1920,
            "overwrite": True,
        },
    }

    render_settings = resolve_render_settings(
        plan_data=plan_data,
        render_preset_override="preview",
        output_width_override=720,
        output_height_override=1280,
        overwrite_override=False,
    )

    assert render_settings["mode"] == "preset"
    assert render_settings["preset_name"] == "preview"
    assert render_settings["preset"] == "veryfast"
    assert render_settings["crf"] == "22"
    assert render_settings["output_width"] == 720
    assert render_settings["output_height"] == 1280
    assert render_settings["overwrite"] is False
    assert render_settings["output_name_mode"] == "suffix"
    assert render_settings["output_suffix"] == "_vertical"
    assert render_settings["output_name"] == "episode01_vertical.mp4"


def test_resolve_render_settings_supports_custom_mode(tmp_path: Path):
    source_path = tmp_path / "episode01.mp4"
    plan_data = {
        "source_path": str(source_path),
        "render": {
            "mode": "custom",
            "preset_name": "manual",
            "output_name_mode": "suffix",
            "output_suffix": "_manual-test",
            "output_name": "manual.mp4",
            "output_width": 1080,
            "output_height": 1920,
            "overwrite": True,
            "video_codec": "libx264",
            "audio_codec": "aac",
            "audio_bitrate": "256k",
            "preset": "slow",
            "crf": "16",
            "scale_flags": "lanczos",
        },
    }

    render_settings = resolve_render_settings(plan_data=plan_data)

    assert render_settings["mode"] == "custom"
    assert render_settings["video_codec"] == "libx264"
    assert render_settings["audio_bitrate"] == "256k"
    assert render_settings["crf"] == "16"
    assert render_settings["output_name"] == "episode01_manual-test.mp4"


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
