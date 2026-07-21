"""
FFmpeg discovery helpers for Windows audio-loading dependencies.
"""

# standard library imports
from dataclasses import dataclass
import os
from pathlib import Path
import platform
from typing import Callable, Mapping

FFMPEG_DLL_DIR_ENV_VAR = "CLIPSAI_FFMPEG_DLL_DIR"
REQUIRED_FFMPEG_DLL_PATTERNS = ("avcodec*.dll", "avformat*.dll", "avutil*.dll")

_DLL_DIRECTORY_HANDLES = []
_CONFIGURED_DLL_DIRS: set[str] = set()


class FfmpegDllDirectoryError(RuntimeError):
    """
    Raised when a required FFmpeg shared-DLL directory cannot be configured.
    """


@dataclass(frozen=True)
class FfmpegDllDirectoryConfig:
    """
    Result of a Windows FFmpeg DLL-directory setup attempt.
    """

    configured: bool
    path: str | None
    source: str
    reason: str


def has_ffmpeg_shared_dlls(directory: str | Path) -> bool:
    """
    Return whether a folder looks like a full/shared FFmpeg ``bin`` directory.
    """
    path = Path(directory)
    try:
        if not path.is_dir():
            return False
        return all(any(path.glob(pattern)) for pattern in REQUIRED_FFMPEG_DLL_PATTERNS)
    except OSError:
        return False


def _split_path_env(path_value: str, platform_system: str) -> list[str]:
    separator = ";" if platform_system == "Windows" else os.pathsep
    return [part.strip().strip('"') for part in path_value.split(separator) if part.strip()]


def _iter_common_windows_ffmpeg_dirs(env: Mapping[str, str]) -> list[Path]:
    candidates: list[Path] = []
    local_app_data = env.get("LOCALAPPDATA")
    if local_app_data:
        winget_root = (
            Path(local_app_data)
            / "Microsoft"
            / "WinGet"
            / "Packages"
            / "Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe"
        )
        try:
            candidates.extend(
                sorted(winget_root.glob("ffmpeg-*-full_build-shared/bin"), reverse=True)
            )
        except OSError:
            pass

    for base_env_name in ("ProgramFiles", "ProgramFiles(x86)", "ChocolateyInstall"):
        base_path = env.get(base_env_name)
        if not base_path:
            continue
        candidates.extend(
            [
                Path(base_path) / "ffmpeg" / "bin",
                Path(base_path) / "Gyan.FFmpeg" / "bin",
            ]
        )

    candidates.append(Path("C:/ffmpeg/bin"))
    return candidates


def discover_ffmpeg_dll_directory(
    env: Mapping[str, str] | None = None,
    platform_system: str | None = None,
) -> tuple[Path | None, str, str]:
    """
    Locate a usable FFmpeg shared-DLL directory on Windows.
    """
    platform_system = platform_system or platform.system()
    env = os.environ if env is None else env
    explicit_path = env.get(FFMPEG_DLL_DIR_ENV_VAR, "").strip().strip('"')

    if explicit_path:
        explicit_dir = Path(explicit_path).expanduser()
        if has_ffmpeg_shared_dlls(explicit_dir):
            return explicit_dir.resolve(), FFMPEG_DLL_DIR_ENV_VAR, "found configured path"
        return (
            None,
            FFMPEG_DLL_DIR_ENV_VAR,
            (
                f"{FFMPEG_DLL_DIR_ENV_VAR} points to '{explicit_dir}', but that "
                "folder does not contain avcodec*.dll, avformat*.dll, and avutil*.dll."
            ),
        )

    for path_entry in _split_path_env(env.get("PATH", ""), platform_system):
        path_dir = Path(path_entry).expanduser()
        if has_ffmpeg_shared_dlls(path_dir):
            return path_dir.resolve(), "PATH", "found FFmpeg shared DLLs on PATH"

    for candidate in _iter_common_windows_ffmpeg_dirs(env):
        if has_ffmpeg_shared_dlls(candidate):
            return candidate.resolve(), "auto-detect", "found common FFmpeg shared build"

    return (
        None,
        "not_found",
        (
            "No FFmpeg full/shared bin folder was found. Install a full/shared "
            f"FFmpeg build and set {FFMPEG_DLL_DIR_ENV_VAR} to its bin folder."
        ),
    )


def configure_ffmpeg_dll_directory(
    required: bool = False,
    env: Mapping[str, str] | None = None,
    platform_system: str | None = None,
    add_dll_directory: Callable[[str], object] | None = None,
) -> FfmpegDllDirectoryConfig:
    """
    Register a Windows FFmpeg shared-DLL directory for TorchCodec audio loading.
    """
    platform_system = platform_system or platform.system()
    if platform_system != "Windows":
        return FfmpegDllDirectoryConfig(
            configured=False,
            path=None,
            source="non_windows",
            reason="FFmpeg DLL directory setup is only needed on Windows.",
        )

    dll_dir, source, reason = discover_ffmpeg_dll_directory(
        env=env,
        platform_system=platform_system,
    )
    if dll_dir is None:
        if required:
            raise FfmpegDllDirectoryError(reason)
        return FfmpegDllDirectoryConfig(
            configured=False,
            path=None,
            source=source,
            reason=reason,
        )

    dll_dir_text = str(dll_dir)
    if dll_dir_text in _CONFIGURED_DLL_DIRS:
        return FfmpegDllDirectoryConfig(
            configured=True,
            path=dll_dir_text,
            source=source,
            reason="FFmpeg DLL directory was already registered.",
        )

    add_dll_directory = add_dll_directory or os.add_dll_directory
    try:
        handle = add_dll_directory(dll_dir_text)
    except (AttributeError, OSError) as error:
        message = (
            f"Could not register FFmpeg DLL directory '{dll_dir_text}'. "
            f"Set {FFMPEG_DLL_DIR_ENV_VAR} to a full/shared FFmpeg bin folder. "
            f"Original error: {error}"
        )
        if required:
            raise FfmpegDllDirectoryError(message) from error
        return FfmpegDllDirectoryConfig(
            configured=False,
            path=dll_dir_text,
            source=source,
            reason=message,
        )

    _DLL_DIRECTORY_HANDLES.append(handle)
    _CONFIGURED_DLL_DIRS.add(dll_dir_text)
    return FfmpegDllDirectoryConfig(
        configured=True,
        path=dll_dir_text,
        source=source,
        reason=reason,
    )
