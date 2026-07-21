from pathlib import Path

import pytest

from clipsai.utils import ffmpeg
from clipsai.utils.ffmpeg import FFMPEG_DLL_DIR_ENV_VAR
from clipsai.utils.ffmpeg import FfmpegDllDirectoryError
from clipsai.utils.ffmpeg import configure_ffmpeg_dll_directory
from clipsai.utils.ffmpeg import discover_ffmpeg_dll_directory
from clipsai.utils.ffmpeg import has_ffmpeg_shared_dlls


@pytest.fixture(autouse=True)
def reset_ffmpeg_dll_state(monkeypatch):
    monkeypatch.setattr(ffmpeg, "_DLL_DIRECTORY_HANDLES", [])
    monkeypatch.setattr(ffmpeg, "_CONFIGURED_DLL_DIRS", set())


def _write_required_ffmpeg_dlls(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename in ("avcodec-62.dll", "avformat-62.dll", "avutil-60.dll"):
        (directory / filename).write_text("", encoding="utf-8")


def test_has_ffmpeg_shared_dlls_rejects_exe_only_folder(tmp_path):
    ffmpeg_dir = tmp_path / "ffmpeg" / "bin"
    ffmpeg_dir.mkdir(parents=True)
    (ffmpeg_dir / "ffmpeg.exe").write_text("", encoding="utf-8")

    assert has_ffmpeg_shared_dlls(ffmpeg_dir) is False


def test_has_ffmpeg_shared_dlls_accepts_required_dll_patterns(tmp_path):
    ffmpeg_dir = tmp_path / "ffmpeg" / "bin"
    _write_required_ffmpeg_dlls(ffmpeg_dir)

    assert has_ffmpeg_shared_dlls(ffmpeg_dir) is True


def test_configure_ffmpeg_dll_directory_is_noop_off_windows():
    calls = []

    result = configure_ffmpeg_dll_directory(
        platform_system="Linux",
        env={},
        add_dll_directory=lambda path: calls.append(path),
    )

    assert result.configured is False
    assert result.source == "non_windows"
    assert calls == []


def test_configure_ffmpeg_dll_directory_prefers_explicit_env_path(tmp_path):
    env_dir = tmp_path / "env" / "bin"
    path_dir = tmp_path / "path" / "bin"
    _write_required_ffmpeg_dlls(env_dir)
    _write_required_ffmpeg_dlls(path_dir)
    (path_dir / "ffmpeg.exe").write_text("", encoding="utf-8")
    calls = []

    result = configure_ffmpeg_dll_directory(
        platform_system="Windows",
        env={
            FFMPEG_DLL_DIR_ENV_VAR: str(env_dir),
            "PATH": str(path_dir),
        },
        add_dll_directory=lambda path: calls.append(path) or object(),
    )

    assert result.configured is True
    assert result.source == FFMPEG_DLL_DIR_ENV_VAR
    assert calls == [str(env_dir.resolve())]


def test_configure_ffmpeg_dll_directory_rejects_bad_explicit_env_path(tmp_path):
    exe_only_dir = tmp_path / "ffmpeg" / "bin"
    exe_only_dir.mkdir(parents=True)
    (exe_only_dir / "ffmpeg.exe").write_text("", encoding="utf-8")

    with pytest.raises(FfmpegDllDirectoryError, match="avcodec"):
        configure_ffmpeg_dll_directory(
            required=True,
            platform_system="Windows",
            env={FFMPEG_DLL_DIR_ENV_VAR: str(exe_only_dir)},
            add_dll_directory=lambda path: object(),
        )


def test_discover_ffmpeg_dll_directory_ignores_static_path_entry(tmp_path):
    static_dir = tmp_path / "static" / "bin"
    shared_dir = tmp_path / "shared" / "bin"
    static_dir.mkdir(parents=True)
    (static_dir / "ffmpeg.exe").write_text("", encoding="utf-8")
    shared_dir.mkdir(parents=True)
    (shared_dir / "ffmpeg.exe").write_text("", encoding="utf-8")
    _write_required_ffmpeg_dlls(shared_dir)

    dll_dir, source, reason = discover_ffmpeg_dll_directory(
        platform_system="Windows",
        env={"PATH": f"{static_dir};{shared_dir}"},
    )

    assert dll_dir == shared_dir.resolve()
    assert source == "PATH"
    assert "shared DLLs" in reason


def test_configure_ffmpeg_dll_directory_keeps_directory_handle(tmp_path):
    ffmpeg_dir = tmp_path / "ffmpeg" / "bin"
    _write_required_ffmpeg_dlls(ffmpeg_dir)
    handle = object()

    configure_ffmpeg_dll_directory(
        platform_system="Windows",
        env={FFMPEG_DLL_DIR_ENV_VAR: str(ffmpeg_dir)},
        add_dll_directory=lambda path: handle,
    )

    assert ffmpeg._DLL_DIRECTORY_HANDLES == [handle]
    assert str(ffmpeg_dir.resolve()) in ffmpeg._CONFIGURED_DLL_DIRS
