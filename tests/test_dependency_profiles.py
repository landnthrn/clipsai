import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_requirement_lines(filename: str) -> list[str]:
    lines = []
    for line in (REPO_ROOT / filename).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


def _read_install_requires() -> list[str]:
    setup_ast = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    for node in ast.walk(setup_ast):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg != "install_requires":
                continue
            if not isinstance(keyword.value, ast.List):
                raise AssertionError("install_requires is no longer a plain list.")
            dependencies = []
            for item in keyword.value.elts:
                if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                    raise AssertionError("install_requires contains a non-string entry.")
                dependencies.append(item.value)
            return dependencies
    raise AssertionError("Could not find setup(... install_requires=...) in setup.py.")


def test_community_requirements_keep_torch_stack_out_of_pip_install():
    requirements = _read_requirement_lines("requirements-community.txt")

    assert "mediapipe==0.10.21" in requirements
    assert "pyannote.audio>=4,<5" in requirements
    assert "pyannote.core>=6,<7" in requirements
    assert all(not line.startswith("torch") for line in requirements)
    assert all(not line.startswith("torchaudio") for line in requirements)
    assert all(not line.startswith("torchvision") for line in requirements)
    assert all(not line.startswith("torchcodec") for line in requirements)
    assert "facenet-pytorch==2.6.0" not in requirements


def test_legacy_requirements_preserve_legacy_diarization_profile():
    requirements = _read_requirement_lines("requirements-legacy.txt")

    assert "facenet-pytorch==2.6.0" in requirements
    assert "mediapipe==0.10.21" in requirements
    assert "pyannote.audio==3.1.1" in requirements
    assert "pyannote.core<6" in requirements
    assert all(not line.startswith("torch") for line in requirements)
    assert all(not line.startswith("torchaudio") for line in requirements)
    assert all(not line.startswith("torchvision") for line in requirements)


def test_setup_install_requires_excludes_optional_ml_stacks():
    install_requires = _read_install_requires()

    assert "facenet-pytorch" not in install_requires
    assert "pyannote.audio" not in install_requires
    assert "pyannote.core" not in install_requires
    assert "sentence-transformers" not in install_requires
    assert "torch" not in install_requires
