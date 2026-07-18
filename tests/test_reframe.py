from datetime import datetime
from pathlib import Path

import pytest

from clipsai.reframe import PLAN_FILE_SUFFIX, PLAN_VERSION, build_plan
from clipsai.reframe import build_render_media_file, build_render_summary_markdown
from clipsai.reframe import build_summary_and_logs_batch_root
from clipsai.reframe import build_summary_and_logs_payload, build_timeline_csv_rows
from clipsai.reframe import create_crops_from_plan, default_output_path, default_plan_path
from clipsai.reframe import default_raw_diarization_path
from clipsai.reframe import discover_plan_files, discover_video_files
from clipsai.reframe import format_summary_and_logs_timestamp, get_enabled_segments
from clipsai.reframe import load_plan
from clipsai.reframe import normalize_plan_data, original_output_filename
from clipsai.reframe import resolve_output_filename, resolve_render_settings
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
        analysis_settings={
            "diarization_model": "legacy-3.1",
            "num_speakers": None,
            "min_speakers": None,
            "max_speakers": None,
            "min_segment_duration": 0.75,
            "face_detect_backend": "mtcnn",
            "mediapipe_face_detect_model_selection": 0,
            "mediapipe_face_detect_min_detection_confidence": 0.5,
            "raw_diarization_path": None,
        },
    )

    assert plan["plan_version"] == PLAN_VERSION
    assert "_editing_help" in plan
    assert plan["source_filename"] == "podcast.mp4"
    assert plan["analysis"]["crop_width"] == 608
    assert plan["analysis"]["diarization_model"] == "legacy-3.1"
    assert plan["analysis"]["face_detect_backend"] == "mtcnn"
    assert plan["analysis"]["raw_diarization_path"] is None
    assert plan["render"]["mode"] == "preset"
    assert plan["render"]["preset_name"] == "high"
    assert plan["render"]["output_name_mode"] == "suffix"
    assert plan["render"]["output_suffix"] == "_vertical"
    assert plan["render"]["output_summary_and_logs"] is True
    assert "output_name" not in plan["render"]
    assert plan["segments"][0]["segment_id"] == "segment_0001"
    assert plan["segments"][0]["enabled"] is True
    assert plan["segments"][0]["speakers"] == [0]
    assert plan["segments"][0]["notes"] == ""
    assert plan["segments"][1]["x"] == 900


def test_default_paths(tmp_path: Path):
    source_path = tmp_path / "episode01.mp4"
    plan_path = default_plan_path(source_path, tmp_path / "plans")
    output_path = default_output_path(source_path, tmp_path / "output")
    raw_diarization_path = default_raw_diarization_path(source_path, tmp_path / "plans")

    assert plan_path.name == f"episode01{PLAN_FILE_SUFFIX}"
    assert output_path.name == "episode01_vertical.mp4"
    assert raw_diarization_path.name == "episode01.raw-diarization.json"


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
            "export_summary_markdown": True,
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
    assert "_editing_help" in normalized
    assert normalized["source_filename"] == "episode01.mp4"
    assert normalized["analysis"]["diarization_model"] == "legacy-3.1"
    assert normalized["analysis"]["num_speakers"] is None
    assert normalized["analysis"]["min_speakers"] is None
    assert normalized["analysis"]["max_speakers"] is None
    assert normalized["analysis"]["face_detect_backend"] == "mtcnn"
    assert normalized["analysis"]["mediapipe_face_detect_model_selection"] == 0
    assert normalized["analysis"]["mediapipe_face_detect_min_detection_confidence"] == 0.5
    assert normalized["analysis"]["raw_diarization_path"] is None
    assert normalized["render"]["mode"] == "preset"
    assert normalized["render"]["output_name_mode"] == "suffix"
    assert normalized["render"]["output_suffix"] == "_vertical"
    assert normalized["render"]["overwrite"] is True
    assert normalized["render"]["output_summary_and_logs"] is True
    assert "export_summary_markdown" not in normalized["render"]
    assert "export_result_debug" not in normalized["render"]
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


def test_normalize_plan_data_defaults_community_runs_to_mediapipe(tmp_path: Path):
    source_path = tmp_path / "episode01.mov"
    plan_data = {
        "plan_version": 1,
        "source_path": str(source_path),
        "analysis": {
            "diarization_model": "community-1",
            "original_width": 1920,
            "original_height": 1080,
            "crop_width": 608,
            "crop_height": 1080,
        },
        "render": {
            "preset_name": "high",
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

    normalized = normalize_plan_data(plan_data)

    assert normalized["analysis"]["face_detect_backend"] == "mediapipe"


def test_load_plan_rewrites_old_toggle_fields_to_new_shape(tmp_path: Path):
    plan_path = tmp_path / f"episode01{PLAN_FILE_SUFFIX}"
    legacy_plan = {
        "plan_version": 1,
        "source_path": str(tmp_path / "episode01.mp4"),
        "analysis": {
            "original_width": 1920,
            "original_height": 1080,
            "crop_width": 608,
            "crop_height": 1080,
        },
        "render": {
            "preset_name": "high",
            "export_result_debug": True,
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
    plan_path.write_text(__import__("json").dumps(legacy_plan), encoding="utf-8")

    loaded_plan = load_plan(plan_path)
    rewritten_plan = __import__("json").loads(plan_path.read_text(encoding="utf-8"))

    assert loaded_plan["render"]["output_summary_and_logs"] is True
    assert "export_result_debug" not in rewritten_plan["render"]
    assert "export_summary_markdown" not in rewritten_plan["render"]
    assert rewritten_plan["render"]["output_summary_and_logs"] is True


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
            "output_width": 1080,
            "output_height": 1920,
            "overwrite": True,
            "output_summary_and_logs": True,
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
    assert render_settings["output_summary_and_logs"] is True
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
            "output_width": 1080,
            "output_height": 1920,
            "overwrite": True,
            "output_summary_and_logs": True,
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
    assert render_settings["output_summary_and_logs"] is True
    assert render_settings["output_name"] == "episode01_manual-test.mp4"


def test_build_render_summary_markdown_contains_core_details(tmp_path: Path):
    plan_path = tmp_path / f"episode01{PLAN_FILE_SUFFIX}"
    source_path = tmp_path / "episode01.mp4"
    output_path = tmp_path / "episode01_vertical.mp4"
    summary_path = tmp_path / "summary-and-logs" / "episode01" / "summary.md"
    render_settings = {
        "mode": "preset",
        "preset_name": "preview",
        "output_name_mode": "suffix",
        "output_suffix": "_vertical",
        "output_width": 1080,
        "output_height": 1920,
        "overwrite": True,
        "video_codec": "libx264",
        "audio_codec": "aac",
        "audio_bitrate": "160k",
        "scale_flags": "lanczos",
    }
    enabled_segments = [
        {"segment_id": "segment_0001", "start_time": 0.0, "end_time": 5.5},
        {"segment_id": "segment_0002", "start_time": 5.5, "end_time": 12.0},
    ]
    disabled_segments = [
        {"segment_id": "segment_0003", "start_time": 12.0, "end_time": 14.0}
    ]

    summary_markdown = build_render_summary_markdown(
        generated_at="07/16/2026 02:30 AM",
        plan_path=plan_path,
        source_path=source_path,
        output_path=output_path,
        summary_path=summary_path,
        analysis_data={
            "diarization_model": "community-1",
            "num_speakers": None,
            "min_speakers": 2,
            "max_speakers": 4,
            "face_detect_backend": "mediapipe",
            "mediapipe_face_detect_model_selection": 1,
            "mediapipe_face_detect_min_detection_confidence": 0.65,
            "raw_diarization_path": "plans/raw-diarization/episode01.raw-diarization.json",
        },
        render_settings=render_settings,
        enabled_segments=enabled_segments,
        disabled_segments=disabled_segments,
    )

    assert "# Render Summary" in summary_markdown
    assert "Created: 07/16/2026 02:30 AM" in summary_markdown
    assert "Rendered output" in summary_markdown
    assert "Diarization model: `community-1`" in summary_markdown
    assert "Min speakers: `2`" in summary_markdown
    assert "Face-detection backend: `mediapipe`" in summary_markdown
    assert "MediaPipe face-detect model selection: `1`" in summary_markdown
    assert "Output size: `1080x1920`" in summary_markdown
    assert "Enabled segments rendered: `2`" in summary_markdown
    assert "Detected speaker count: `0`" in summary_markdown
    assert "| segment_0001 | enabled | 0.000000 | 5.500000 | 5.500000 | - |  |  | - |" in summary_markdown


def test_format_summary_and_logs_timestamp_uses_windows_safe_layout():
    formatted = format_summary_and_logs_timestamp(datetime(2026, 7, 16, 18, 32))

    assert formatted == "6-32PM_07-16"


def test_build_summary_and_logs_batch_root_uses_batch_label_and_timestamp(tmp_path: Path):
    batch_root = build_summary_and_logs_batch_root(
        output_dir=tmp_path / "output",
        batch_label="Podcast EP2",
        now=datetime(2026, 7, 16, 18, 32),
    )

    assert (
        batch_root
        == tmp_path / "output" / "summary-and-logs" / "Podcast EP2_6-32PM_07-16"
    )


def test_build_timeline_csv_rows_marks_enabled_and_disabled_segments():
    rows = build_timeline_csv_rows(
        enabled_segments=[
            {
                "segment_id": "segment_0001",
                "start_time": 1.0,
                "end_time": 4.5,
                "speakers": [0, 1],
                "x": 120,
                "y": 10,
                "notes": "keep",
            }
        ],
        disabled_segments=[
            {
                "segment_id": "segment_0002",
                "start_time": 4.5,
                "end_time": 6.0,
                "speakers": [1],
                "x": 400,
                "y": 0,
                "notes": "skip",
            }
        ],
    )

    assert rows[0]["status"] == "enabled"
    assert rows[0]["duration_seconds"] == 3.5
    assert rows[0]["speakers"] == "0,1"
    assert rows[1]["status"] == "disabled"
    assert rows[1]["notes"] == "skip"


def test_build_summary_and_logs_payload_contains_file_render_and_segment_details(tmp_path: Path):
    export_paths = {
        "video_root": tmp_path / "summary-and-logs" / "episode01",
        "summary_path": tmp_path / "summary-and-logs" / "episode01" / "summary.md",
        "full_record_path": (
            tmp_path / "summary-and-logs" / "episode01" / "full-record.json"
        ),
        "timeline_path": tmp_path / "summary-and-logs" / "episode01" / "timeline.csv",
    }
    render_settings = {
        "mode": "preset",
        "preset_name": "high",
        "output_name_mode": "suffix",
        "output_suffix": "_vertical",
        "output_name": "episode01_vertical.mp4",
        "output_width": 1080,
        "output_height": 1920,
        "overwrite": True,
        "video_codec": "libx264",
        "audio_codec": "aac",
        "audio_bitrate": "320k",
        "scale_flags": "lanczos",
    }
    enabled_segments = [
        {
            "segment_id": "segment_0001",
            "start_time": 0.0,
            "end_time": 5.0,
            "speakers": [0],
            "x": 100,
            "y": 0,
            "notes": "",
        }
    ]
    disabled_segments = [
        {
            "segment_id": "segment_0002",
            "start_time": 5.0,
            "end_time": 8.0,
            "speakers": [1],
            "x": 800,
            "y": 0,
            "notes": "skip",
        }
    ]

    payload = build_summary_and_logs_payload(
        generated_at="07/16/2026 06:32 PM",
        plan_path=tmp_path / f"episode01{PLAN_FILE_SUFFIX}",
        source_path=tmp_path / "episode01.mp4",
        output_path=tmp_path / "output" / "episode01_vertical.mp4",
        export_paths=export_paths,
        analysis_data={
            "diarization_model": "legacy-3.1",
            "num_speakers": 2,
            "min_speakers": None,
            "max_speakers": None,
            "face_detect_backend": "mtcnn",
            "mediapipe_face_detect_model_selection": 0,
            "mediapipe_face_detect_min_detection_confidence": 0.5,
            "raw_diarization_path": "plans/raw-diarization/episode01.raw-diarization.json",
        },
        render_settings=render_settings,
        enabled_segments=enabled_segments,
        disabled_segments=disabled_segments,
    )

    assert payload["created_at"] == "07/16/2026 06:32 PM"
    assert payload["files"]["timeline_path"].endswith("timeline.csv")
    assert payload["files"]["full_record_path"].endswith("full-record.json")
    assert payload["analysis"]["num_speakers"] == 2
    assert payload["analysis"]["face_detect_backend"] == "mtcnn"
    assert payload["render"]["output_name"] == "episode01_vertical.mp4"
    assert payload["segments"]["enabled_count"] == 1
    assert payload["segments"]["disabled_count"] == 1
    assert payload["segments"]["rendered_start_time"] == 0.0
    assert payload["segments"]["rendered_end_time"] == 5.0
    assert payload["segments"]["timeline_rows"][1]["status"] == "disabled"


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
