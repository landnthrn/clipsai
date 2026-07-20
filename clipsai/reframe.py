"""
Simple analyze/render workflow for speaker-focused vertical reframing.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
import shutil
import subprocess
from pathlib import Path

from clipsai.diarize.config import DEFAULT_DIARIZATION_MODEL
from clipsai.diarize.config import get_supported_diarization_models
from clipsai.resize.config import DEFAULT_FACE_DETECT_BACKEND
from clipsai.resize.config import DEFAULT_MEDIAPIPE_FACE_DETECT_MIN_DETECTION_CONFIDENCE
from clipsai.resize.config import get_default_face_detect_backend
from clipsai.resize.config import get_default_mediapipe_face_detect_model_selection
from clipsai.resize.config import get_supported_face_detect_backends

PLAN_VERSION = 4
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
    ".webm",
}
PLAN_FILE_SUFFIX = ".reframe-plan.json"
DEFAULT_OUTPUT_WIDTH = 1080
DEFAULT_OUTPUT_HEIGHT = 1920
DEFAULT_ASPECT_RATIO = (9, 16)
DEFAULT_MIN_SEGMENT_DURATION = 1.5
DEFAULT_SAMPLES_PER_SEGMENT = 13
DEFAULT_FACE_DETECT_WIDTH = 960
DEFAULT_FACE_DETECT_BACKEND_NAME = DEFAULT_FACE_DETECT_BACKEND
DEFAULT_MEDIAPIPE_FACE_DETECT_CONFIDENCE = (
    DEFAULT_MEDIAPIPE_FACE_DETECT_MIN_DETECTION_CONFIDENCE
)
DEFAULT_SCENE_MERGE_THRESHOLD = 0.25
DEFAULT_RENDER_PRESET = "high"
DEFAULT_RENDER_MODE = "preset"
DEFAULT_OUTPUT_EXTENSION = ".mp4"
DEFAULT_OUTPUT_NAME_MODE = "suffix"
DEFAULT_OUTPUT_SUFFIX = "_vertical"
DEFAULT_SUMMARY_AND_LOGS_DIRNAME = "summary-and-logs"
RAW_DIARIZATION_DIRNAME = "raw-diarization"
RAW_DIARIZATION_FILE_SUFFIX = ".raw-diarization.json"

RENDER_PRESETS = {
    "preview": {
        "video_codec": "libx264",
        "audio_codec": "aac",
        "audio_bitrate": "160k",
        "preset": "veryfast",
        "crf": "22",
        "scale_flags": "lanczos",
    },
    "high": {
        "video_codec": "libx264",
        "audio_codec": "aac",
        "audio_bitrate": "320k",
        "preset": "slow",
        "crf": "14",
        "scale_flags": "lanczos",
    },
    "fast": {
        "video_codec": "h264_nvenc",
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "preset": "p7",
        "cq": "16",
        "bitrate": "0",
        "scale_flags": "lanczos",
    },
    "master": {
        "video_codec": "libx264",
        "audio_codec": "aac",
        "audio_bitrate": "320k",
        "preset": "slow",
        "crf": "0",
        "scale_flags": "lanczos",
    },
}


def discover_video_files(input_path: str | Path) -> list[Path]:
    """
    Return supported video files from a single file or a folder.
    """
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video file type: {path}")
        return [path]

    video_files = sorted(
        file_path.resolve()
        for file_path in path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )
    if not video_files:
        raise FileNotFoundError(f"No supported video files found in: {path}")
    return video_files


def discover_plan_files(input_path: str | Path) -> list[Path]:
    """
    Return plan files from a single file or a folder.
    """
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Plan path does not exist: {path}")

    if path.is_file():
        if not path.name.endswith(PLAN_FILE_SUFFIX):
            raise ValueError(f"Unsupported plan file: {path}")
        return [path]

    plan_files = sorted(file_path.resolve() for file_path in path.glob(f"*{PLAN_FILE_SUFFIX}"))
    if not plan_files:
        raise FileNotFoundError(f"No plan files found in: {path}")
    return plan_files


def resolve_hf_token(explicit_token: str | None) -> str:
    """
    Resolve the Hugging Face token from argument or environment.
    """
    token = explicit_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Set it in the environment or pass --hf-token.")
    return token


def original_output_filename(source_path: str | Path) -> str:
    """
    Return the rendered filename that keeps the source base name.
    """
    return f"{Path(source_path).stem}{DEFAULT_OUTPUT_EXTENSION}"


def default_output_filename(
    source_path: str | Path,
    suffix: str = DEFAULT_OUTPUT_SUFFIX,
) -> str:
    """
    Return the default rendered output filename for a source video.
    """
    return f"{Path(source_path).stem}{suffix}{DEFAULT_OUTPUT_EXTENSION}"


def infer_output_name_settings(source_path: str | Path, render_data: dict) -> tuple[str, str]:
    """
    Infer output naming mode and suffix from older plan files when possible.
    """
    output_name = render_data.get("output_name")
    if not output_name:
        return DEFAULT_OUTPUT_NAME_MODE, DEFAULT_OUTPUT_SUFFIX

    if output_name == default_output_filename(source_path):
        return "suffix", DEFAULT_OUTPUT_SUFFIX
    if output_name == original_output_filename(source_path):
        return "keep_original", ""

    output_name_path = Path(output_name)
    if output_name_path.suffix.lower() == DEFAULT_OUTPUT_EXTENSION:
        source_stem = Path(source_path).stem
        output_stem = output_name_path.stem
        if output_stem.startswith(source_stem):
            inferred_suffix = output_stem[len(source_stem) :]
            if inferred_suffix:
                return "suffix", inferred_suffix

    return "explicit", ""


def resolve_output_filename(source_path: str | Path, render_data: dict) -> str:
    """
    Resolve the final output filename from editable render naming fields.
    """
    output_name_mode = render_data.get("output_name_mode", DEFAULT_OUTPUT_NAME_MODE)
    if output_name_mode == "keep_original":
        return original_output_filename(source_path)
    if output_name_mode == "suffix":
        return default_output_filename(
            source_path,
            render_data.get("output_suffix", DEFAULT_OUTPUT_SUFFIX),
        )
    if output_name_mode == "explicit":
        output_name = render_data.get("output_name")
        if not output_name:
            raise ValueError("Render naming mode 'explicit' requires an output_name value.")
        return output_name
    raise ValueError(f"Unsupported output naming mode in plan: {output_name_mode}")


def build_segment_plan_entry(segment, index: int) -> dict:
    """
    Convert a resize segment into a hand-editable plan entry.
    """
    segment_data = segment.to_dict()
    return {
        "segment_id": f"segment_{index + 1:04d}",
        "enabled": True,
        "speakers": segment_data["speakers"],
        "start_time": segment_data["start_time"],
        "end_time": segment_data["end_time"],
        "x": segment_data["x"],
        "y": segment_data["y"],
        "notes": "",
    }


def build_render_plan_entry(
    video_path: str | Path,
    render_preset: str,
    output_width: int,
    output_height: int,
    output_name_mode: str = DEFAULT_OUTPUT_NAME_MODE,
    output_suffix: str = DEFAULT_OUTPUT_SUFFIX,
) -> dict:
    """
    Create the editable render section stored in each plan.
    """
    render_data = {
        "mode": DEFAULT_RENDER_MODE,
        "preset_name": render_preset,
        "output_name_mode": output_name_mode,
        "output_suffix": output_suffix,
        "output_width": output_width,
        "output_height": output_height,
        "overwrite": True,
        "output_summary_and_logs": True,
        **RENDER_PRESETS[render_preset],
    }
    return {
        **render_data,
    }


def build_plan_editing_help() -> dict:
    """
    Return ignored guidance fields that make plan editing easier by hand.
    """
    return {
        "summary": (
            "Fields that start with '_' are ignored by the renderer and are only "
            "here as editing help."
        ),
        "plan_version": "Internal plan format version. Leave this value alone.",
        "source_path": "Original source video path. Usually not meant to be edited.",
        "source_filename": "Original source video filename. Usually not meant to be edited.",
        "analysis": {
            "original_width": "Detected source width. Usually not meant to be edited.",
            "original_height": "Detected source height. Usually not meant to be edited.",
            "crop_width": "Detected crop width from analysis. Usually not meant to be edited.",
            "crop_height": "Detected crop height from analysis. Usually not meant to be edited.",
            "aspect_ratio": "Target crop shape from analysis. Usually not meant to be edited.",
            "diarization_model": "Analyze-time diarization pipeline choice. Usually not meant to be edited after analysis.",
            "num_speakers": "Analyze-time exact speaker count. Usually not meant to be edited after analysis.",
            "min_speakers": "Analyze-time lower speaker-count bound. Usually not meant to be edited after analysis.",
            "max_speakers": "Analyze-time upper speaker-count bound. Usually not meant to be edited after analysis.",
            "min_segment_duration": "Analyze-time setting used to build this plan. Usually not meant to be edited after analysis.",
            "samples_per_segment": "Analyze-time setting used to build this plan. Usually not meant to be edited after analysis.",
            "face_detect_width": "Analyze-time face-detection size. Usually not meant to be edited after analysis.",
            "face_detect_backend": "Analyze-time face-detection backend choice. Usually not meant to be edited after analysis.",
            "mediapipe_face_detect_model_selection": "Analyze-time MediaPipe face-detection model selection. Usually not meant to be edited after analysis.",
            "mediapipe_face_detect_min_detection_confidence": "Analyze-time MediaPipe face-detection minimum confidence. Usually not meant to be edited after analysis.",
            "scene_merge_threshold": "Analyze-time merge setting. Usually not meant to be edited after analysis.",
            "raw_diarization_path": "Optional saved raw diarization JSON path from analysis. Usually not meant to be edited.",
        },
        "render": {
            "mode": (
                "preset = use preset_name. custom = use the advanced codec fields "
                "below."
            ),
            "preset_name": "Named quality preset such as preview, high, fast, or master.",
            "output_name_mode": (
                "suffix = source base name plus output_suffix. "
                "keep_original = keep the source base name."
            ),
            "output_suffix": (
                "Used only when output_name_mode is suffix. "
                "Examples: _vertical or _social-cut."
            ),
            "overwrite": "true = replace same-name output. false = fail instead.",
            "output_summary_and_logs": (
                "true = export a per-video folder with summary.md, "
                "full-record.json, and timeline.csv. false = skip it."
            ),
            "output_width": "Rendered width in pixels. Usually not meant to be edited unless you want a different export size.",
            "output_height": "Rendered height in pixels. Usually not meant to be edited unless you want a different export size.",
            "video_codec": "Advanced render field. Usually not meant to be edited unless mode is custom.",
            "audio_codec": "Advanced render field. Usually not meant to be edited unless mode is custom.",
            "audio_bitrate": "Advanced render field. Usually not meant to be edited unless mode is custom.",
            "preset": "Advanced encoder preset. Usually not meant to be edited unless mode is custom.",
            "crf": "Advanced quality value for libx264. Usually not meant to be edited unless mode is custom.",
            "cq": "Advanced quality value for NVENC-style codecs. Usually not meant to be edited unless mode is custom.",
            "bitrate": "Advanced bitrate field for NVENC-style codecs. Usually not meant to be edited unless mode is custom.",
            "scale_flags": "Advanced scaling field. Usually not meant to be edited unless mode is custom.",
        },
        "segments": {
            "segment_id": "Stable label for this segment. Usually not meant to be edited.",
            "enabled": "true = render this segment. false = skip it.",
            "speakers": "Detected speaker IDs for this segment. Usually not meant to be edited.",
            "start_time": "Segment start time in seconds.",
            "end_time": "Segment end time in seconds.",
            "x": "Horizontal crop position in source pixels.",
            "y": "Vertical crop position in source pixels.",
            "notes": "Optional personal note. Ignored by the renderer.",
        },
    }


def build_plan(
    video_path: str | Path,
    crops,
    aspect_ratio: tuple[int, int],
    output_width: int,
    output_height: int,
    render_preset: str,
    analysis_settings: dict,
    output_name_mode: str = DEFAULT_OUTPUT_NAME_MODE,
    output_suffix: str = DEFAULT_OUTPUT_SUFFIX,
) -> dict:
    """
    Build a serializable plan from a Crops result.
    """
    video_path = Path(video_path).resolve()
    return {
        "plan_version": PLAN_VERSION,
        "_editing_help": build_plan_editing_help(),
        "source_path": str(video_path),
        "source_filename": video_path.name,
        "analysis": {
            "original_width": crops.original_width,
            "original_height": crops.original_height,
            "crop_width": crops.crop_width,
            "crop_height": crops.crop_height,
            "aspect_ratio": list(aspect_ratio),
            **analysis_settings,
        },
        "render": build_render_plan_entry(
            video_path=video_path,
            render_preset=render_preset,
            output_width=output_width,
            output_height=output_height,
            output_name_mode=output_name_mode,
            output_suffix=output_suffix,
        ),
        "segments": [
            build_segment_plan_entry(segment=segment, index=index)
            for index, segment in enumerate(crops.segments)
        ],
    }


def normalize_segment_plan_entry(segment_data: dict, index: int) -> dict:
    """
    Fill in editable segment defaults while preserving existing segment values.
    """
    normalized_segment = dict(segment_data)
    normalized_segment.setdefault("segment_id", f"segment_{index + 1:04d}")
    normalized_segment.setdefault("enabled", True)
    normalized_segment.setdefault("notes", "")
    return normalized_segment


def normalize_plan_data(plan_data: dict) -> dict:
    """
    Upgrade older plan files in memory and fill in editable defaults.
    """
    if "source_path" not in plan_data:
        raise ValueError("Plan is missing required field: source_path")
    if "segments" not in plan_data:
        raise ValueError("Plan is missing required field: segments")

    normalized_plan = dict(plan_data)
    source_path = Path(normalized_plan["source_path"]).expanduser().resolve()
    analysis_data = dict(normalized_plan.get("analysis", {}))
    render_data = dict(normalized_plan.get("render", {}))
    inferred_output_name_mode, inferred_output_suffix = infer_output_name_settings(
        source_path=source_path,
        render_data=render_data,
    )

    analysis_data.setdefault("diarization_model", DEFAULT_DIARIZATION_MODEL)
    analysis_data.setdefault("num_speakers", None)
    analysis_data.setdefault("min_speakers", None)
    analysis_data.setdefault("max_speakers", None)
    analysis_data.setdefault("raw_diarization_path", None)
    analysis_data.setdefault(
        "face_detect_backend",
        get_default_face_detect_backend(analysis_data["diarization_model"]),
    )
    analysis_data.setdefault(
        "mediapipe_face_detect_model_selection",
        get_default_mediapipe_face_detect_model_selection(
            analysis_data["diarization_model"]
        ),
    )
    analysis_data.setdefault(
        "mediapipe_face_detect_min_detection_confidence",
        DEFAULT_MEDIAPIPE_FACE_DETECT_CONFIDENCE,
    )
    render_data.setdefault("mode", DEFAULT_RENDER_MODE)
    render_data.setdefault("preset_name", DEFAULT_RENDER_PRESET)
    render_data.setdefault("output_name_mode", inferred_output_name_mode)
    render_data.setdefault("output_suffix", inferred_output_suffix)
    render_data.setdefault("output_width", DEFAULT_OUTPUT_WIDTH)
    render_data.setdefault("output_height", DEFAULT_OUTPUT_HEIGHT)
    render_data.setdefault("overwrite", True)
    legacy_summary_flag = bool(render_data.get("export_summary_markdown", False))
    legacy_debug_flag = bool(render_data.get("export_result_debug", False))
    render_data.setdefault(
        "output_summary_and_logs",
        legacy_summary_flag or legacy_debug_flag,
    )
    render_data.pop("export_summary_markdown", None)
    render_data.pop("export_result_debug", None)

    normalized_plan["plan_version"] = max(
        int(normalized_plan.get("plan_version", 1)),
        PLAN_VERSION,
    )
    normalized_plan.setdefault("_editing_help", build_plan_editing_help())
    normalized_plan["source_path"] = str(source_path)
    normalized_plan.setdefault("source_filename", source_path.name)
    normalized_plan["analysis"] = analysis_data
    normalized_plan["render"] = render_data
    normalized_plan["segments"] = [
        normalize_segment_plan_entry(segment_data=segment, index=index)
        for index, segment in enumerate(normalized_plan["segments"])
    ]
    return normalized_plan


def validate_plan_data(plan_data: dict) -> dict:
    """
    Validate the core editable plan structure with user-facing errors.
    """
    for required_key in ("source_path", "analysis", "render", "segments"):
        if required_key not in plan_data:
            raise ValueError(f"Plan is missing required field: {required_key}")

    if not isinstance(plan_data["segments"], list) or not plan_data["segments"]:
        raise ValueError("Plan must contain at least one segment.")

    for segment in plan_data["segments"]:
        segment_id = segment.get("segment_id", "unknown-segment")
        for required_key in ("speakers", "start_time", "end_time", "x", "y"):
            if required_key not in segment:
                raise ValueError(
                    f"Segment '{segment_id}' is missing required field: {required_key}"
                )
        if float(segment["end_time"]) <= float(segment["start_time"]):
            raise ValueError(
                f"Segment '{segment_id}' has an invalid time range. "
                "end_time must be greater than start_time."
            )

    return plan_data


def get_enabled_segments(plan_data: dict) -> list[dict]:
    """
    Return only enabled plan segments and fail clearly if none remain.
    """
    enabled_segments = [
        segment for segment in plan_data["segments"] if segment.get("enabled", True)
    ]
    if not enabled_segments:
        raise ValueError("Plan has no enabled segments to render.")
    return enabled_segments


def create_crops_from_plan(plan_data: dict) -> Crops:
    """
    Recreate a Crops object from plan JSON data.
    """
    from clipsai.resize.crops import Crops
    from clipsai.resize.segment import Segment

    segments = [
        Segment(
            speakers=segment["speakers"],
            start_time=segment["start_time"],
            end_time=segment["end_time"],
            x=segment["x"],
            y=segment["y"],
        )
        for segment in get_enabled_segments(plan_data)
    ]
    analysis = plan_data["analysis"]
    return Crops(
        original_width=analysis["original_width"],
        original_height=analysis["original_height"],
        crop_width=analysis["crop_width"],
        crop_height=analysis["crop_height"],
        segments=segments,
    )


def default_plan_path(video_path: Path, plans_dir: Path) -> Path:
    """
    Return the default JSON plan path for a source video.
    """
    return plans_dir / f"{video_path.stem}{PLAN_FILE_SUFFIX}"


def default_output_path(source_path: Path, output_dir: Path) -> Path:
    """
    Return the default rendered output path for a source video.
    """
    return output_dir / default_output_filename(source_path)


def default_raw_diarization_path(video_path: Path, plans_dir: Path) -> Path:
    """
    Return the default raw diarization JSON path for one source video.
    """
    return (
        plans_dir
        / RAW_DIARIZATION_DIRNAME
        / f"{video_path.stem}{RAW_DIARIZATION_FILE_SUFFIX}"
    )


def store_plan(plan_path: Path, plan_data: dict) -> Path:
    """
    Write a plan JSON file to disk.
    """
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w", encoding="utf-8") as file_object:
        json.dump(plan_data, file_object, indent=2)
        file_object.write("\n")
    return plan_path


def load_plan(plan_path: Path) -> dict:
    """
    Read a plan JSON file from disk.
    """
    with plan_path.open("r", encoding="utf-8") as file_object:
        raw_plan_data = json.load(file_object)
    normalized_plan_data = validate_plan_data(normalize_plan_data(raw_plan_data))
    if normalized_plan_data != raw_plan_data:
        store_plan(plan_path, normalized_plan_data)
    return normalized_plan_data


def resolve_render_settings(
    plan_data: dict,
    render_preset_override: str | None = None,
    output_width_override: int | None = None,
    output_height_override: int | None = None,
    overwrite_override: bool | None = None,
) -> dict:
    """
    Resolve preset/custom render settings plus optional CLI overrides.
    """
    render_data = dict(plan_data["render"])
    source_path = Path(plan_data["source_path"])

    if render_preset_override is not None:
        render_data["mode"] = "preset"
        render_data["preset_name"] = render_preset_override

    mode = render_data.get("mode", DEFAULT_RENDER_MODE)
    if mode == "preset":
        preset_name = render_data.get("preset_name", DEFAULT_RENDER_PRESET)
        if preset_name not in RENDER_PRESETS:
            raise ValueError(f"Unknown render preset in plan: {preset_name}")
        resolved_settings = dict(RENDER_PRESETS[preset_name])
    elif mode == "custom":
        required_fields = [
            "video_codec",
            "audio_codec",
            "audio_bitrate",
            "preset",
            "scale_flags",
        ]
        missing_fields = [
            field_name for field_name in required_fields if field_name not in render_data
        ]
        if missing_fields:
            missing_fields_text = ", ".join(missing_fields)
            raise ValueError(
                "Custom render mode is missing required fields: "
                f"{missing_fields_text}"
            )
        resolved_settings = {field_name: render_data[field_name] for field_name in required_fields}
        if render_data["video_codec"] == "libx264":
            if "crf" not in render_data:
                raise ValueError("Custom render mode with libx264 requires a 'crf' value.")
            resolved_settings["crf"] = render_data["crf"]
        else:
            for required_field in ("cq", "bitrate"):
                if required_field not in render_data:
                    raise ValueError(
                        "Custom render mode for non-libx264 codecs requires "
                        f"'{required_field}'."
                    )
                resolved_settings[required_field] = render_data[required_field]
    else:
        raise ValueError(f"Unsupported render mode in plan: {mode}")

    resolved_settings["mode"] = mode
    resolved_settings["preset_name"] = render_data.get("preset_name", DEFAULT_RENDER_PRESET)
    resolved_settings["output_width"] = output_width_override or int(
        render_data.get("output_width", DEFAULT_OUTPUT_WIDTH)
    )
    resolved_settings["output_height"] = output_height_override or int(
        render_data.get("output_height", DEFAULT_OUTPUT_HEIGHT)
    )
    resolved_settings["output_name_mode"] = render_data.get(
        "output_name_mode",
        DEFAULT_OUTPUT_NAME_MODE,
    )
    resolved_settings["output_suffix"] = render_data.get(
        "output_suffix",
        DEFAULT_OUTPUT_SUFFIX,
    )
    resolved_settings["output_name"] = resolve_output_filename(source_path, render_data)
    resolved_settings["overwrite"] = (
        bool(render_data.get("overwrite", True))
        if overwrite_override is None
        else overwrite_override
    )
    resolved_settings["output_summary_and_logs"] = bool(
        render_data.get("output_summary_and_logs", False)
    )
    return resolved_settings


def format_summary_and_logs_timestamp(now: datetime) -> str:
    """
    Format a Windows-safe timestamp for result debug folder names.
    """
    return now.strftime("%I-%M%p_%m-%d").lstrip("0")


def sanitize_summary_and_logs_folder_name(name: str) -> str:
    """
    Convert a display name into a filesystem-safe folder name.
    """
    invalid_characters = '<>:"/\\|?*'
    sanitized = "".join("_" if character in invalid_characters else character for character in name)
    sanitized = sanitized.strip().rstrip(".")
    return sanitized or "batch"


def infer_batch_summary_and_logs_label(plan_files: list[Path], input_path: Path) -> str:
    """
    Infer a batch label from the common source folder when possible.
    """
    source_parent_names = []
    for plan_path in plan_files:
        plan_data = load_plan(plan_path)
        source_parent_names.append(Path(plan_data["source_path"]).parent.name)

    unique_parent_names = {name for name in source_parent_names if name}
    if len(unique_parent_names) == 1:
        return unique_parent_names.pop()

    return input_path.name or "batch"


def build_summary_and_logs_batch_root(
    output_dir: Path,
    batch_label: str,
    now: datetime,
) -> Path:
    """
    Build the parent summary/logs folder for one batch render run.
    """
    safe_label = sanitize_summary_and_logs_folder_name(batch_label)
    timestamp = format_summary_and_logs_timestamp(now)
    return output_dir / DEFAULT_SUMMARY_AND_LOGS_DIRNAME / f"{safe_label}_{timestamp}"


def build_summary_and_logs_paths(
    source_path: Path,
    output_dir: Path,
    batch_root: Path | None,
) -> dict[str, Path]:
    """
    Resolve the per-video summary/logs file paths.
    """
    video_root = (
        batch_root or (output_dir / DEFAULT_SUMMARY_AND_LOGS_DIRNAME)
    ) / sanitize_summary_and_logs_folder_name(source_path.stem)
    return {
        "video_root": video_root,
        "summary_path": video_root / "summary.md",
        "full_record_path": video_root / "full-record.json",
        "timeline_path": video_root / "timeline.csv",
    }


def build_render_summary_markdown(
    generated_at: str,
    plan_path: Path,
    source_path: Path,
    output_path: Path,
    summary_path: Path,
    analysis_data: dict,
    render_settings: dict,
    enabled_segments: list[dict],
    disabled_segments: list[dict],
) -> str:
    """
    Build a markdown summary of one render run.
    """
    def format_optional_value(value, empty_text: str = "not set") -> str:
        return empty_text if value is None else str(value)

    unique_speakers = sorted(
        {
            str(speaker)
            for segment in enabled_segments + disabled_segments
            for speaker in segment.get("speakers", [])
        }
    )
    timeline_rows = build_timeline_csv_rows(
        enabled_segments=enabled_segments,
        disabled_segments=disabled_segments,
    )
    lines = [
        "# Render Summary",
        "",
        f"Created: {generated_at}",
        "",
        "## Files",
        "",
        f"- Plan: `{plan_path}`",
        f"- Source video: `{source_path}`",
        f"- Rendered output: `{output_path}`",
        f"- Summary file: `{summary_path}`",
        "",
        "## Analysis Settings",
        "",
        f"- Diarization model: `{analysis_data.get('diarization_model', DEFAULT_DIARIZATION_MODEL)}`",
        f"- Exact speaker count: `{format_optional_value(analysis_data.get('num_speakers'))}`",
        f"- Min speakers: `{format_optional_value(analysis_data.get('min_speakers'))}`",
        f"- Max speakers: `{format_optional_value(analysis_data.get('max_speakers'))}`",
        f"- Face-detection backend: `{analysis_data.get('face_detect_backend', DEFAULT_FACE_DETECT_BACKEND_NAME)}`",
        f"- MediaPipe face-detect model selection: `{format_optional_value(analysis_data.get('mediapipe_face_detect_model_selection'))}`",
        f"- MediaPipe face-detect minimum confidence: `{format_optional_value(analysis_data.get('mediapipe_face_detect_min_detection_confidence'))}`",
        f"- Raw diarization path: `{format_optional_value(analysis_data.get('raw_diarization_path'), 'not saved')}`",
        "",
        "## Render Settings",
        "",
        f"- Mode: `{render_settings['mode']}`",
        f"- Preset: `{render_settings['preset_name']}`",
        f"- Output naming mode: `{render_settings['output_name_mode']}`",
        f"- Output suffix: `{render_settings['output_suffix']}`",
        f"- Output size: `{render_settings['output_width']}x{render_settings['output_height']}`",
        f"- Overwrite: `{render_settings['overwrite']}`",
        f"- Video codec: `{render_settings['video_codec']}`",
        f"- Audio codec: `{render_settings['audio_codec']}`",
        f"- Audio bitrate: `{render_settings['audio_bitrate']}`",
        f"- Scale flags: `{render_settings['scale_flags']}`",
        "",
        "## Segment Summary",
        "",
        f"- Enabled segments rendered: `{len(enabled_segments)}`",
        f"- Disabled segments skipped: `{len(disabled_segments)}`",
        f"- Total segments in plan: `{len(enabled_segments) + len(disabled_segments)}`",
        f"- Detected speaker count: `{len(unique_speakers)}`",
        f"- Detected speaker IDs: `{', '.join(unique_speakers) if unique_speakers else 'none'}`",
    ]

    if enabled_segments:
        first_start_time = min(float(segment["start_time"]) for segment in enabled_segments)
        last_end_time = max(float(segment["end_time"]) for segment in enabled_segments)
        lines.append(f"- Rendered timeline: `{first_start_time}` to `{last_end_time}` seconds")

    if disabled_segments:
        disabled_segment_ids = ", ".join(
            segment.get("segment_id", "unknown-segment") for segment in disabled_segments
        )
        lines.append(f"- Disabled segment IDs: `{disabled_segment_ids}`")

    lines.extend(
        [
            "",
            "## Timeline",
            "",
            "| Segment | Status | Start | End | Duration | Speakers | X | Y | Notes |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in timeline_rows:
        lines.append(
            "| "
            f"{row['segment_id']} | "
            f"{row['status']} | "
            f"{row['start_time']:.6f} | "
            f"{row['end_time']:.6f} | "
            f"{row['duration_seconds']:.6f} | "
            f"{row['speakers'] or '-'} | "
            f"{row['x']} | "
            f"{row['y']} | "
            f"{row['notes'] or '-'} |"
        )

    return "\n".join(lines) + "\n"


def build_timeline_csv_rows(
    enabled_segments: list[dict],
    disabled_segments: list[dict],
) -> list[dict]:
    """
    Build flat CSV rows for rendered and skipped segments.
    """
    rows = []
    for status, segments in (("enabled", enabled_segments), ("disabled", disabled_segments)):
        for segment in segments:
            start_time = float(segment["start_time"])
            end_time = float(segment["end_time"])
            rows.append(
                {
                    "segment_id": segment.get("segment_id", ""),
                    "status": status,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_seconds": round(end_time - start_time, 6),
                    "speakers": ",".join(str(speaker) for speaker in segment.get("speakers", [])),
                    "x": segment.get("x", ""),
                    "y": segment.get("y", ""),
                    "notes": segment.get("notes", ""),
                }
            )
    return rows


def build_summary_and_logs_payload(
    generated_at: str,
    plan_path: Path,
    source_path: Path,
    output_path: Path,
    export_paths: dict[str, Path],
    analysis_data: dict,
    render_settings: dict,
    enabled_segments: list[dict],
    disabled_segments: list[dict],
) -> dict:
    """
    Build the structured JSON payload for summary/log exports.
    """
    timeline_rows = build_timeline_csv_rows(
        enabled_segments=enabled_segments,
        disabled_segments=disabled_segments,
    )
    rendered_start_time = None
    rendered_end_time = None
    if enabled_segments:
        rendered_start_time = min(float(segment["start_time"]) for segment in enabled_segments)
        rendered_end_time = max(float(segment["end_time"]) for segment in enabled_segments)

    return {
        "created_at": generated_at,
        "files": {
            "plan_path": str(plan_path),
            "source_path": str(source_path),
            "output_path": str(output_path),
            "summary_and_logs_folder": str(export_paths["video_root"]),
            "summary_path": str(export_paths["summary_path"]),
            "full_record_path": str(export_paths["full_record_path"]),
            "timeline_path": str(export_paths["timeline_path"]),
        },
        "analysis": analysis_data,
        "render": {
            "mode": render_settings["mode"],
            "preset_name": render_settings["preset_name"],
            "output_name_mode": render_settings["output_name_mode"],
            "output_suffix": render_settings["output_suffix"],
            "output_name": render_settings["output_name"],
            "output_width": render_settings["output_width"],
            "output_height": render_settings["output_height"],
            "overwrite": render_settings["overwrite"],
            "video_codec": render_settings["video_codec"],
            "audio_codec": render_settings["audio_codec"],
            "audio_bitrate": render_settings["audio_bitrate"],
            "scale_flags": render_settings["scale_flags"],
        },
        "segments": {
            "enabled_count": len(enabled_segments),
            "disabled_count": len(disabled_segments),
            "rendered_start_time": rendered_start_time,
            "rendered_end_time": rendered_end_time,
            "timeline_rows": timeline_rows,
        },
    }


def write_summary_and_logs_exports(
    export_paths: dict[str, Path],
    summary_markdown: str,
    full_record_payload: dict,
    timeline_rows: list[dict],
    overwrite: bool,
) -> dict[str, Path]:
    """
    Write per-video summary/log exports.
    """
    for path_key in ("summary_path", "full_record_path", "timeline_path"):
        path = export_paths[path_key]
        if path.exists() and overwrite is False:
            raise FileExistsError(
                f"Summary/log file already exists and overwrite is disabled: {path}"
            )

    export_paths["video_root"].mkdir(parents=True, exist_ok=True)

    with export_paths["summary_path"].open("w", encoding="utf-8") as file_object:
        file_object.write(summary_markdown)

    with export_paths["full_record_path"].open("w", encoding="utf-8") as file_object:
        json.dump(full_record_payload, file_object, indent=2)
        file_object.write("\n")

    with export_paths["timeline_path"].open("w", encoding="utf-8", newline="") as file_object:
        writer = csv.DictWriter(
            file_object,
            fieldnames=[
                "segment_id",
                "status",
                "start_time",
                "end_time",
                "duration_seconds",
                "speakers",
                "x",
                "y",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(timeline_rows)

    return export_paths


def analyze_video(
    video_path: Path,
    plans_dir: Path,
    hf_token: str,
    render_preset: str,
    output_width: int,
    output_height: int,
    output_name_mode: str,
    output_suffix: str,
    min_segment_duration: float,
    samples_per_segment: int,
    face_detect_width: int,
    face_detect_backend: str | None,
    mediapipe_face_detect_model_selection: int | None,
    mediapipe_face_detect_min_detection_confidence: float,
    scene_merge_threshold: float,
    diarization_model: str,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    save_raw_diarization: bool,
) -> Path:
    """
    Analyze one video and write its editable plan JSON.
    """
    from clipsai.resize.resize import resize

    raw_diarization_path = None
    if save_raw_diarization:
        raw_diarization_path = default_raw_diarization_path(video_path, plans_dir)
    face_detect_backend = face_detect_backend or get_default_face_detect_backend(
        diarization_model
    )
    mediapipe_face_detect_model_selection = (
        mediapipe_face_detect_model_selection
        if mediapipe_face_detect_model_selection is not None
        else get_default_mediapipe_face_detect_model_selection(diarization_model)
    )

    crops = resize(
        video_file_path=str(video_path),
        pyannote_auth_token=hf_token,
        aspect_ratio=DEFAULT_ASPECT_RATIO,
        diarization_model=diarization_model,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        raw_diarization_output_path=(
            None if raw_diarization_path is None else str(raw_diarization_path)
        ),
        min_segment_duration=min_segment_duration,
        samples_per_segment=samples_per_segment,
        face_detect_width=face_detect_width,
        face_detect_backend=face_detect_backend,
        mediapipe_face_detect_model_selection=mediapipe_face_detect_model_selection,
        mediapipe_face_detect_min_detection_confidence=(
            mediapipe_face_detect_min_detection_confidence
        ),
        scene_merge_threshold=scene_merge_threshold,
    )
    analysis_settings = {
        "diarization_model": diarization_model,
        "num_speakers": num_speakers,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "min_segment_duration": min_segment_duration,
        "samples_per_segment": samples_per_segment,
        "face_detect_width": face_detect_width,
        "face_detect_backend": face_detect_backend,
        "mediapipe_face_detect_model_selection": mediapipe_face_detect_model_selection,
        "mediapipe_face_detect_min_detection_confidence": (
            mediapipe_face_detect_min_detection_confidence
        ),
        "scene_merge_threshold": scene_merge_threshold,
        "raw_diarization_path": (
            None if raw_diarization_path is None else str(raw_diarization_path)
        ),
    }
    plan_data = build_plan(
        video_path=video_path,
        crops=crops,
        aspect_ratio=DEFAULT_ASPECT_RATIO,
        output_width=output_width,
        output_height=output_height,
        render_preset=render_preset,
        analysis_settings=analysis_settings,
        output_name_mode=output_name_mode,
        output_suffix=output_suffix,
    )
    return store_plan(default_plan_path(video_path, plans_dir), plan_data)


def run_command(command: list[str]) -> None:
    """
    Run a subprocess command and raise a helpful error on failure.
    """
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Stdout: {result.stdout}\n"
            f"Stderr: {result.stderr}"
        )


def clamp_crop_value(value: int, maximum: int) -> int:
    """
    Keep a crop coordinate inside the frame.
    """
    return max(0, min(int(value), max(0, int(maximum))))


def build_render_media_file(
    source_path: Path,
    temporal_media_cls=None,
    video_file_cls=None,
    audiovideo_file_cls=None,
):
    """
    Open a render source as either a video-only file or an audio-video file.
    """
    if (
        temporal_media_cls is None
        or video_file_cls is None
        or audiovideo_file_cls is None
    ):
        from clipsai.media.audiovideo_file import AudioVideoFile
        from clipsai.media.temporal_media_file import TemporalMediaFile
        from clipsai.media.video_file import VideoFile

        temporal_media_cls = temporal_media_cls or TemporalMediaFile
        video_file_cls = video_file_cls or VideoFile
        audiovideo_file_cls = audiovideo_file_cls or AudioVideoFile

    media_file = temporal_media_cls(str(source_path))
    media_file.assert_exists()
    if media_file.has_video_stream() is False:
        raise ValueError(f"Source file does not contain a video stream: {source_path}")
    if media_file.has_audio_stream():
        return audiovideo_file_cls(str(source_path))
    return video_file_cls(str(source_path))


def render_plan(
    plan_path: Path,
    output_dir: Path,
    render_preset_override: str | None = None,
    output_width_override: int | None = None,
    output_height_override: int | None = None,
    overwrite_override: bool | None = None,
    summary_and_logs_batch_root: Path | None = None,
    render_started_at: datetime | None = None,
) -> Path:
    """
    Render one plan JSON file into a vertical video file.
    """
    plan_data = load_plan(plan_path)
    source_path = Path(plan_data["source_path"]).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source video from plan does not exist: {source_path}")

    crops = create_crops_from_plan(plan_data)
    enabled_segments = get_enabled_segments(plan_data)
    disabled_segments = [
        segment for segment in plan_data["segments"] if segment.get("enabled", True) is False
    ]
    render_settings = resolve_render_settings(
        plan_data=plan_data,
        render_preset_override=render_preset_override,
        output_width_override=output_width_override,
        output_height_override=output_height_override,
        overwrite_override=overwrite_override,
    )
    render_started_at = render_started_at or datetime.now()
    generated_at = render_started_at.strftime("%m/%d/%Y %I:%M %p")
    output_path = output_dir / render_settings["output_name"]
    analysis_data = plan_data["analysis"]
    export_paths = build_summary_and_logs_paths(
        source_path=source_path,
        output_dir=output_dir,
        batch_root=summary_and_logs_batch_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and render_settings["overwrite"] is False:
        raise FileExistsError(
            f"Output file already exists and overwrite is disabled: {output_path}"
        )
    if render_settings["output_summary_and_logs"] and render_settings["overwrite"] is False:
        for path_key in ("summary_path", "full_record_path", "timeline_path"):
            debug_path = export_paths[path_key]
            if debug_path.exists():
                raise FileExistsError(
                    f"Summary/log file already exists and overwrite is disabled: {debug_path}"
                )

    temp_dir = output_dir / f"{source_path.stem}_temp_segments"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    video = build_render_media_file(source_path)
    max_x = video.get_width_pixels() - crops.crop_width
    max_y = video.get_height_pixels() - crops.crop_height

    segment_paths: list[Path] = []
    for index, segment in enumerate(crops.segments):
        segment_path = temp_dir / f"segment_{index:04d}.mp4"
        segment_paths.append(segment_path)

        x = clamp_crop_value(segment.x, max_x)
        y = clamp_crop_value(segment.y, max_y)
        filter_chain = (
            f"crop={crops.crop_width}:{crops.crop_height}:{x}:{y},"
            f"scale={render_settings['output_width']}:{render_settings['output_height']}"
            f":flags={render_settings['scale_flags']},setsar=1"
        )

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            str(segment.start_time),
            "-to",
            str(segment.end_time),
            "-i",
            str(source_path),
            "-vf",
            filter_chain,
            "-c:v",
            render_settings["video_codec"],
            "-c:a",
            render_settings["audio_codec"],
            "-b:a",
            render_settings["audio_bitrate"],
        ]

        if render_settings["video_codec"] == "libx264":
            command.extend(
                [
                    "-preset",
                    render_settings["preset"],
                    "-crf",
                    render_settings["crf"],
                ]
            )
        else:
            command.extend(
                [
                    "-preset",
                    render_settings["preset"],
                    "-cq",
                    render_settings["cq"],
                    "-b:v",
                    render_settings["bitrate"],
                ]
            )

        command.append(str(segment_path))
        run_command(command)

    concat_file = temp_dir / "concat.txt"
    with concat_file.open("w", encoding="utf-8") as file_object:
        for segment_path in segment_paths:
            file_object.write(f"file '{segment_path.name}'\n")

    run_command(
        [
            "ffmpeg",
            "-y" if render_settings["overwrite"] else "-n",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(output_path),
        ]
    )

    if render_settings["output_summary_and_logs"]:
        summary_markdown = build_render_summary_markdown(
            generated_at=generated_at,
            plan_path=plan_path,
            source_path=source_path,
            output_path=output_path,
            summary_path=export_paths["summary_path"],
            analysis_data=analysis_data,
            render_settings=render_settings,
            enabled_segments=enabled_segments,
            disabled_segments=disabled_segments,
        )
        timeline_rows = build_timeline_csv_rows(
            enabled_segments=enabled_segments,
            disabled_segments=disabled_segments,
        )
        full_record_payload = build_summary_and_logs_payload(
            generated_at=generated_at,
            plan_path=plan_path,
            source_path=source_path,
            output_path=output_path,
            export_paths=export_paths,
            analysis_data=analysis_data,
            render_settings=render_settings,
            enabled_segments=enabled_segments,
            disabled_segments=disabled_segments,
        )
        write_summary_and_logs_exports(
            export_paths=export_paths,
            summary_markdown=summary_markdown,
            full_record_payload=full_record_payload,
            timeline_rows=timeline_rows,
            overwrite=render_settings["overwrite"],
        )

    shutil.rmtree(temp_dir)
    return output_path


def analyze_command(args: argparse.Namespace) -> int:
    """
    CLI handler for analysis-only mode.
    """
    hf_token = resolve_hf_token(args.hf_token)
    video_files = discover_video_files(args.input)
    plans_dir = Path(args.plans_dir).expanduser().resolve()
    for video_path in video_files:
        plan_path = analyze_video(
            video_path=video_path,
            plans_dir=plans_dir,
            hf_token=hf_token,
            render_preset=args.render_preset,
            output_width=args.output_width,
            output_height=args.output_height,
            output_name_mode=args.output_name_mode,
            output_suffix=args.output_suffix,
            min_segment_duration=args.min_segment_duration,
            samples_per_segment=args.samples_per_segment,
            face_detect_width=args.face_detect_width,
            face_detect_backend=args.face_detect_backend,
            mediapipe_face_detect_model_selection=(
                args.mediapipe_face_detect_model_selection
            ),
            mediapipe_face_detect_min_detection_confidence=(
                args.mediapipe_face_detect_min_detection_confidence
            ),
            scene_merge_threshold=args.scene_merge_threshold,
            diarization_model=args.diarization_model,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            save_raw_diarization=args.save_raw_diarization,
        )
        print(f"Created plan: {plan_path}")
    return 0


def render_command(args: argparse.Namespace) -> int:
    """
    CLI handler for render-only mode.
    """
    output_dir = Path(args.output_dir).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()
    plan_files = discover_plan_files(input_path)
    render_started_at = datetime.now()
    summary_and_logs_batch_root = None
    if input_path.is_dir():
        batch_label = infer_batch_summary_and_logs_label(
            plan_files=plan_files,
            input_path=input_path,
        )
        summary_and_logs_batch_root = build_summary_and_logs_batch_root(
            output_dir=output_dir,
            batch_label=batch_label,
            now=render_started_at,
        )
    for plan_path in plan_files:
        output_path = render_plan(
            plan_path=plan_path,
            output_dir=output_dir,
            render_preset_override=args.render_preset,
            output_width_override=args.output_width,
            output_height_override=args.output_height,
            overwrite_override=args.overwrite,
            summary_and_logs_batch_root=summary_and_logs_batch_root,
            render_started_at=render_started_at,
        )
        print(f"Rendered video: {output_path}")
    return 0


def run_command_handler(args: argparse.Namespace) -> int:
    """
    CLI handler for simple analyze-then-render mode.
    """
    analyze_command(args)
    render_args = argparse.Namespace(
        input=args.plans_dir,
        output_dir=args.output_dir,
        render_preset=None,
        output_width=None,
        output_height=None,
        overwrite=None,
    )
    return render_command(render_args)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the reframe workflow CLI parser.
    """
    parser = argparse.ArgumentParser(
        description="Analyze and render speaker-focused vertical videos."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_analyze_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--input", required=True, help="Video file or folder path.")
        subparser.add_argument(
            "--plans-dir",
            default="plans",
            help="Where editable JSON plans should be written.",
        )
        subparser.add_argument(
            "--output-width",
            type=int,
            default=DEFAULT_OUTPUT_WIDTH,
            help="Rendered output width.",
        )
        subparser.add_argument(
            "--output-height",
            type=int,
            default=DEFAULT_OUTPUT_HEIGHT,
            help="Rendered output height.",
        )
        subparser.add_argument(
            "--render-preset",
            choices=sorted(RENDER_PRESETS.keys()),
            default="high",
            help="Named render preset stored in each plan.",
        )
        subparser.add_argument(
            "--output-name-mode",
            choices=["suffix", "keep_original"],
            default=DEFAULT_OUTPUT_NAME_MODE,
            help="How rendered filenames should be generated in each plan.",
        )
        subparser.add_argument(
            "--output-suffix",
            default=DEFAULT_OUTPUT_SUFFIX,
            help="Suffix used when output-name-mode is set to 'suffix'.",
        )
        subparser.add_argument(
            "--hf-token",
            default=None,
            help="Optional Hugging Face token. If omitted, HF_TOKEN is used.",
        )
        subparser.add_argument(
            "--min-segment-duration",
            type=float,
            default=DEFAULT_MIN_SEGMENT_DURATION,
            help="Minimum speaker segment duration in seconds.",
        )
        subparser.add_argument(
            "--diarization-model",
            choices=get_supported_diarization_models(),
            default=DEFAULT_DIARIZATION_MODEL,
            help="Which supported pyannote diarization pipeline to use during analysis.",
        )
        subparser.add_argument(
            "--num-speakers",
            type=int,
            default=None,
            help="Exact number of speakers to force during diarization.",
        )
        subparser.add_argument(
            "--min-speakers",
            type=int,
            default=None,
            help="Lower speaker-count bound during diarization.",
        )
        subparser.add_argument(
            "--max-speakers",
            type=int,
            default=None,
            help="Upper speaker-count bound during diarization.",
        )
        subparser.add_argument(
            "--samples-per-segment",
            type=int,
            default=DEFAULT_SAMPLES_PER_SEGMENT,
            help="Frames sampled per segment during crop analysis.",
        )
        subparser.add_argument(
            "--face-detect-width",
            type=int,
            default=DEFAULT_FACE_DETECT_WIDTH,
            help="Downscaled width used for face detection.",
        )
        subparser.add_argument(
            "--face-detect-backend",
            choices=get_supported_face_detect_backends(),
            default=None,
            help=(
                "Optional face-detection backend override. If omitted, "
                "legacy-3.1 defaults to mtcnn and community-1 defaults to mediapipe."
            ),
        )
        subparser.add_argument(
            "--mediapipe-face-detect-model-selection",
            type=int,
            choices=[0, 1],
            default=None,
            help=(
                "MediaPipe face-detection model selection. 0 = short-range, "
                "1 = full-range. If omitted, legacy uses 0 and community-1 uses 1."
            ),
        )
        subparser.add_argument(
            "--mediapipe-face-detect-min-detection-confidence",
            type=float,
            default=DEFAULT_MEDIAPIPE_FACE_DETECT_CONFIDENCE,
            help="Minimum MediaPipe face-detection confidence used when that backend is selected.",
        )
        subparser.add_argument(
            "--scene-merge-threshold",
            type=float,
            default=DEFAULT_SCENE_MERGE_THRESHOLD,
            help="Seconds used when aligning scene boundaries to speaker segments.",
        )
        subparser.add_argument(
            "--save-raw-diarization",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Whether raw pyannote diarization JSON should be saved next to the plans.",
        )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Create editable JSON camera plans from one file or a folder of videos.",
    )
    add_shared_analyze_arguments(analyze_parser)
    analyze_parser.set_defaults(handler=analyze_command)

    render_parser = subparsers.add_parser(
        "render",
        help="Render one plan or a folder of plans into vertical videos.",
    )
    render_parser.add_argument("--input", required=True, help="Plan file or plans folder path.")
    render_parser.add_argument(
        "--output-dir",
        default="output",
        help="Where rendered videos should be written.",
    )
    render_parser.add_argument(
        "--render-preset",
        choices=sorted(RENDER_PRESETS.keys()),
        default=None,
        help="Optional preset override for this render run.",
    )
    render_parser.add_argument(
        "--output-width",
        type=int,
        default=None,
        help="Optional output width override for this render run.",
    )
    render_parser.add_argument(
        "--output-height",
        type=int,
        default=None,
        help="Optional output height override for this render run.",
    )
    render_parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional override for whether existing outputs may be replaced.",
    )
    render_parser.set_defaults(handler=render_command)

    run_parser = subparsers.add_parser(
        "run",
        help="Simple one-shot mode: analyze videos and then render them immediately.",
    )
    add_shared_analyze_arguments(run_parser)
    run_parser.add_argument(
        "--output-dir",
        default="output",
        help="Where rendered videos should be written.",
    )
    run_parser.set_defaults(handler=run_command_handler)

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Module entry point.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
