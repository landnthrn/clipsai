import importlib
import sys
from types import SimpleNamespace


def test_filesys_file_import_works_with_windows_magic_provider_shape():
    original_magic = sys.modules.get("magic")
    sys.modules["magic"] = SimpleNamespace(Magic=object)
    sys.modules.pop("clipsai.filesys.file", None)

    try:
        module = importlib.import_module("clipsai.filesys.file")
    finally:
        sys.modules.pop("clipsai.filesys.file", None)
        if original_magic is None:
            sys.modules.pop("magic", None)
        else:
            sys.modules["magic"] = original_magic

    assert hasattr(module, "File")
