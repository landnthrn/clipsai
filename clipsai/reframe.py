"""
Simple analyze/render workflow for speaker-focused vertical reframing.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

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
DEFAULT_SCENE_MERGE_THRESHOLD = 0.25

RENDER_PRESETS = {
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


def build_plan(
    video_path: str | Path,
    crops,
    aspect_ratio: tuple[int, int],
    output_width: int,
    output_height: int,
    render_preset: str,
    analysis_settings: dict,
) -> dict:
    """
    Build a serializable plan from a Crops result.
    """
    video_path = Path(video_path).resolve()
    return {
        "plan_version": 1,
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
        "render": {
            "preset_name": render_preset,
            "output_width": output_width,
            "output_height": output_height,
            **RENDER_PRESETS[render_preset],
        },
        "segments": [segment.to_dict() for segment in crops.segments],
    }


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
        for segment in plan_data["segments"]
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
    return output_dir / f"{source_path.stem}_vertical.mp4"


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
        return json.load(file_object)


def analyze_video(
    video_path: Path,
    plans_dir: Path,
    hf_token: str,
    render_preset: str,
    output_width: int,
    output_height: int,
    min_segment_duration: float,
    samples_per_segment: int,
    face_detect_width: int,
    scene_merge_threshold: float,
) -> Path:
    """
    Analyze one video and write its editable plan JSON.
    """
    from clipsai.resize.resize import resize

    crops = resize(
        video_file_path=str(video_path),
        pyannote_auth_token=hf_token,
        aspect_ratio=DEFAULT_ASPECT_RATIO,
        min_segment_duration=min_segment_duration,
        samples_per_segment=samples_per_segment,
        face_detect_width=face_detect_width,
        scene_merge_threshold=scene_merge_threshold,
    )
    analysis_settings = {
        "min_segment_duration": min_segment_duration,
        "samples_per_segment": samples_per_segment,
        "face_detect_width": face_detect_width,
        "scene_merge_threshold": scene_merge_threshold,
    }
    plan_data = build_plan(
        video_path=video_path,
        crops=crops,
        aspect_ratio=DEFAULT_ASPECT_RATIO,
        output_width=output_width,
        output_height=output_height,
        render_preset=render_preset,
        analysis_settings=analysis_settings,
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


def render_plan(plan_path: Path, output_dir: Path) -> Path:
    """
    Render one plan JSON file into a vertical video file.
    """
    plan_data = load_plan(plan_path)
    source_path = Path(plan_data["source_path"]).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source video from plan does not exist: {source_path}")

    crops = create_crops_from_plan(plan_data)
    render_settings = plan_data["render"]
    output_path = default_output_path(source_path, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
            "-y",
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
            min_segment_duration=args.min_segment_duration,
            samples_per_segment=args.samples_per_segment,
            face_detect_width=args.face_detect_width,
            scene_merge_threshold=args.scene_merge_threshold,
        )
        print(f"Created plan: {plan_path}")
    return 0


def render_command(args: argparse.Namespace) -> int:
    """
    CLI handler for render-only mode.
    """
    output_dir = Path(args.output_dir).expanduser().resolve()
    plan_files = discover_plan_files(args.input)
    for plan_path in plan_files:
        output_path = render_plan(plan_path=plan_path, output_dir=output_dir)
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
            "--scene-merge-threshold",
            type=float,
            default=DEFAULT_SCENE_MERGE_THRESHOLD,
            help="Seconds used when aligning scene boundaries to speaker segments.",
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
