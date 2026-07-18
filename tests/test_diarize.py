# standard library imports
import json
from unittest.mock import patch, Mock
from types import ModuleType

# local package imports
from clipsai.diarize.config import DIARIZATION_MODELS
from clipsai.diarize.pyannote import PyannoteDiarizer
from clipsai.diarize.pyannote import _patch_speechbrain_lazy_modules
from clipsai.diarize.pyannote import build_pipeline_auth_kwargs
from clipsai.diarize.pyannote import build_speaker_count_kwargs
from clipsai.diarize.pyannote import extract_diarization_annotations
from clipsai.diarize.pyannote import get_pyannote_audio_major_version
from clipsai.diarize.pyannote import serialize_annotation

# third party imports
import pandas as pd
from pyannote.core import Segment, Annotation
import pytest


def test_patch_speechbrain_lazy_modules_adds_file_only_to_lazy_redirects():
    LazyLike = type("LazyLike", (ModuleType,), {"__module__": "speechbrain.utils.importutils"})
    lazy_module = LazyLike("speechbrain.k2_integration")
    normal_module = ModuleType("plain.module")
    already_patched_module = LazyLike("speechbrain.pretrained")
    already_patched_module.__file__ = "already-set"

    module_map = {
        "speechbrain.k2_integration": lazy_module,
        "plain.module": normal_module,
        "speechbrain.pretrained": already_patched_module,
    }

    patched = _patch_speechbrain_lazy_modules(module_map)

    assert patched == 1
    assert lazy_module.__file__ == "<lazy>"
    assert "__file__" not in normal_module.__dict__
    assert already_patched_module.__file__ == "already-set"


def test_get_pyannote_audio_major_version_parses_major_number():
    assert get_pyannote_audio_major_version("4.0.1") == 4


def test_build_pipeline_auth_kwargs_switches_by_major_version():
    with patch("clipsai.diarize.pyannote.get_pyannote_audio_major_version", return_value=4):
        assert build_pipeline_auth_kwargs("token-123") == {"token": "token-123"}


def test_build_speaker_count_kwargs_validates_conflicting_inputs():
    with pytest.raises(ValueError, match="cannot be combined"):
        build_speaker_count_kwargs(num_speakers=2, min_speakers=1)


def test_build_speaker_count_kwargs_returns_expected_values():
    kwargs = build_speaker_count_kwargs(min_speakers=2, max_speakers=4)

    assert kwargs == {"min_speakers": 2, "max_speakers": 4}


def test_extract_diarization_annotations_supports_wrapper_outputs():
    annotation = Annotation()

    class WrappedOutput:
        def __init__(self):
            self.speaker_diarization = annotation
            self.exclusive_speaker_diarization = annotation

    regular_annotation, exclusive_annotation = extract_diarization_annotations(
        WrappedOutput()
    )

    assert regular_annotation is annotation
    assert exclusive_annotation is annotation


@pytest.fixture
def mock_diarizer():
    with patch("pyannote.audio.Pipeline.from_pretrained", return_value=Mock()):
        diarizer = PyannoteDiarizer(auth_token="mock_token")
        diarizer.pipeline = Mock()
        return diarizer


@pytest.fixture
def mock_audio_file():
    mock_audio_file = Mock()
    mock_audio_file.path.return_value = "mock_audio.mp3"
    mock_audio_file.get_duration.return_value = 30.0
    return mock_audio_file


def test_pyannote_diarizer_uses_selected_pipeline_checkpoint():
    with patch("pyannote.audio.Pipeline.from_pretrained", return_value=Mock()) as pipeline_loader:
        PyannoteDiarizer(auth_token="mock_token", diarization_model="legacy-3.1")

    pipeline_loader.assert_called_once_with(
        DIARIZATION_MODELS["legacy-3.1"]["checkpoint"],
        use_auth_token="mock_token",
    )


def test_pyannote_diarizer_uses_community_pipeline_on_pyannote_4():
    pipeline_instance = Mock()
    pipeline_instance.to.return_value = pipeline_instance
    with patch(
        "clipsai.diarize.pyannote.get_pyannote_audio_major_version",
        return_value=4,
    ), patch(
        "pyannote.audio.__version__",
        "4.0.7",
    ), patch(
        "pyannote.audio.Pipeline.from_pretrained",
        return_value=pipeline_instance,
    ) as pipeline_loader:
        diarizer = PyannoteDiarizer(
            auth_token="mock_token",
            diarization_model="community-1",
        )

    assert diarizer.model_name == "community-1"
    pipeline_loader.assert_called_once_with(
        DIARIZATION_MODELS["community-1"]["checkpoint"],
        token="mock_token",
    )


def test_pyannote_diarizer_blocks_community_model_on_old_pyannote():
    with patch(
        "clipsai.diarize.pyannote.get_pyannote_audio_major_version",
        return_value=3,
    ):
        with pytest.raises(RuntimeError, match="requires pyannote.audio 4.x"):
            PyannoteDiarizer(auth_token="mock_token", diarization_model="community-1")


@pytest.mark.parametrize(
    "annotation_data, expected_output",
    [
        # Test 1: Segments with gaps between them
        (
            [
                {"segment": Segment(0, 10), "label": "speaker_0", "track": "_"},
                {"segment": Segment(12, 20), "label": "speaker_1", "track": "_"},
                {"segment": Segment(21, 30), "label": "speaker_0", "track": "_"},
            ],
            [
                {"speakers": [0], "start_time": 0, "end_time": 12},
                {"speakers": [1], "start_time": 12, "end_time": 21},
                {"speakers": [0], "start_time": 21, "end_time": 30},
            ],
        ),
        # Test 2: overlapping segments
        (
            [
                {"segment": Segment(0, 10), "label": "speaker_0", "track": "_"},
                {"segment": Segment(8, 12), "label": "speaker_2", "track": "_"},
                {"segment": Segment(10, 20), "label": "speaker_1", "track": "_"},
                {"segment": Segment(20, 30), "label": "speaker_0", "track": "_"},
            ],
            [
                {"speakers": [0], "start_time": 0, "end_time": 8},
                {"speakers": [2], "start_time": 8, "end_time": 10},
                {"speakers": [1], "start_time": 10, "end_time": 20},
                {"speakers": [0], "start_time": 20, "end_time": 30},
            ],
        ),
        # Test 3: discarding short segments
        (
            [
                {"segment": Segment(0, 10), "label": "speaker_0", "track": "_"},
                {"segment": Segment(11, 20), "label": "speaker_1", "track": "_"},
                {"segment": Segment(15, 16), "label": "speaker_1", "track": "_"},
                {"segment": Segment(21, 30), "label": "speaker_0", "track": "_"},
            ],
            [
                {"speakers": [0], "start_time": 0, "end_time": 11},
                {"speakers": [1], "start_time": 11, "end_time": 21},
                {"speakers": [0], "start_time": 21, "end_time": 30},
            ],
        ),
        # Test 4: merge contiguous segments with same speakers
        (
            [
                {"segment": Segment(0, 10), "label": "speaker_0", "track": "_"},
                {"segment": Segment(10, 12), "label": "speaker_1", "track": "_"},
                {"segment": Segment(12, 15), "label": "speaker_1", "track": "_"},
                {"segment": Segment(15, 20), "label": "speaker_1", "track": "_"},
                {"segment": Segment(20, 30), "label": "speaker_0", "track": "_"},
            ],
            [
                {"speakers": [0], "start_time": 0, "end_time": 10},
                {"speakers": [1], "start_time": 10, "end_time": 20},
                {"speakers": [0], "start_time": 20, "end_time": 30},
            ],
        ),
        # Test 5: handles empty annotation
        ([], [{"speakers": [], "start_time": 0, "end_time": 30}]),
        # Test 6: relabel speakers with discontiguous speaker labels
        (
            [
                {"segment": Segment(0, 10), "label": "speaker_2", "track": "_"},
                {"segment": Segment(10, 20), "label": "speaker_5", "track": "_"},
                {"segment": Segment(20, 30), "label": "speaker_2", "track": "_"},
            ],
            [
                {"speakers": [0], "start_time": 0, "end_time": 10},
                {"speakers": [1], "start_time": 10, "end_time": 20},
                {"speakers": [0], "start_time": 20, "end_time": 30},
            ],
        ),
        # Test 7: relabeling speakers not required with contiguous speaker labels
        (
            [
                {"segment": Segment(0, 10), "label": "speaker_0", "track": "_"},
                {"segment": Segment(10, 20), "label": "speaker_1", "track": "_"},
                {"segment": Segment(20, 30), "label": "speaker_0", "track": "_"},
            ],
            [
                {"speakers": [0], "start_time": 0, "end_time": 10},
                {"speakers": [1], "start_time": 10, "end_time": 20},
                {"speakers": [0], "start_time": 20, "end_time": 30},
            ],
        ),
        # Test 8: handles unlabeled speaker
        (
            [{"segment": Segment(0, 30), "label": "_", "track": "_"}],
            [{"speakers": [], "start_time": 0, "end_time": 30}],
        ),
    ],
)
def test_diarize(mock_diarizer, mock_audio_file, annotation_data, expected_output):
    # handle empty annotation to prevent KeyError
    if not annotation_data:
        annotation = Annotation()
    else:
        df = pd.DataFrame(annotation_data)
        annotation = Annotation().from_df(df)

    mock_diarizer.pipeline.return_value = annotation
    output_segments = mock_diarizer.diarize(mock_audio_file)

    assert output_segments == expected_output


def test_diarize_passes_speaker_count_kwargs_and_writes_raw_output(
    mock_diarizer,
    mock_audio_file,
    tmp_path,
):
    raw_output_path = tmp_path / "raw-diarization.json"
    df = pd.DataFrame(
        [
            {"segment": Segment(0, 10), "label": "speaker_0", "track": "_"},
            {"segment": Segment(10, 20), "label": "speaker_1", "track": "_"},
        ]
    )
    annotation = Annotation().from_df(df)
    mock_diarizer.pipeline.return_value = annotation

    output_segments = mock_diarizer.diarize(
        mock_audio_file,
        num_speakers=2,
        raw_output_path=raw_output_path,
    )

    mock_diarizer.pipeline.assert_called_once_with(mock_audio_file.path, num_speakers=2)
    assert output_segments == [
        {"speakers": [0], "start_time": 0, "end_time": 10},
        {"speakers": [1], "start_time": 10, "end_time": 30},
    ]
    saved_payload = json.loads(raw_output_path.read_text(encoding="utf-8"))
    assert saved_payload["model_name"] == "legacy-3.1"
    assert saved_payload["speaker_count_constraints"] == {"num_speakers": 2}
    assert len(saved_payload["speaker_diarization"]) == 2


def test_serialize_annotation_returns_plain_rows():
    df = pd.DataFrame(
        [{"segment": Segment(0, 1.23456), "label": "speaker_0", "track": "_"}]
    )
    annotation = Annotation().from_df(df)

    serialized = serialize_annotation(annotation, time_precision=3)

    assert serialized == [
        {
            "speaker_label": "speaker_0",
            "track": "_",
            "start_time": 0,
            "end_time": 1.235,
        }
    ]
