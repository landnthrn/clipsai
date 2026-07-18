import builtins
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import clipsai.resize.face_detection as face_detection


def test_importing_face_detection_module_does_not_require_facenet_pytorch():
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "facenet_pytorch":
            raise ImportError("blocked for test")
        return original_import(name, globals, locals, fromlist, level)

    sys.modules.pop("clipsai.resize.face_detection", None)
    with patch("builtins.__import__", side_effect=guarded_import):
        module = importlib.import_module("clipsai.resize.face_detection")

    assert hasattr(module, "CommunityMediaPipeFaceDetector")
    assert hasattr(module, "LegacyMediaPipeFaceDetector")
    assert hasattr(module, "MtcnnFaceDetector")


def test_build_face_detector_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported face-detection backend"):
        face_detection.build_face_detector(backend_name="unknown-backend")


def test_build_face_detector_selects_community_mediapipe_backend():
    with patch.object(face_detection, "CommunityMediaPipeFaceDetector") as detector_cls:
        sentinel = object()
        detector_cls.return_value = sentinel

        detector = face_detection.build_face_detector(
            backend_name="mediapipe",
            diarization_model="community-1",
        )

    assert detector is sentinel
    detector_cls.assert_called_once_with(
        model_selection=0,
        min_detection_confidence=0.5,
    )


def test_build_face_detector_selects_legacy_mediapipe_backend():
    with patch.object(face_detection, "LegacyMediaPipeFaceDetector") as detector_cls:
        sentinel = object()
        detector_cls.return_value = sentinel

        detector = face_detection.build_face_detector(
            backend_name="mediapipe",
            diarization_model="legacy-3.1",
        )

    assert detector is sentinel
    detector_cls.assert_called_once_with(
        model_selection=0,
        min_detection_confidence=0.5,
    )


def test_build_face_detector_creates_mediapipe_without_facenet_pytorch():
    with patch.object(
        face_detection,
        "_import_mtcnn",
        side_effect=AssertionError("facenet-pytorch should not be imported"),
    ), patch.object(face_detection, "CommunityMediaPipeFaceDetector") as detector_cls:
        sentinel = object()
        detector_cls.return_value = sentinel

        detector = face_detection.build_face_detector(
            backend_name="mediapipe",
            diarization_model="community-1",
        )

    assert detector is sentinel


def test_mtcnn_backend_raises_clear_error_when_facenet_pytorch_is_missing():
    with patch.object(
        face_detection,
        "_import_mtcnn",
        side_effect=RuntimeError(
            "Face-detection backend 'mtcnn' requires the optional "
            "'facenet-pytorch' package."
        ),
    ):
        with pytest.raises(RuntimeError, match="requires the optional 'facenet-pytorch'"):
            face_detection.build_face_detector(backend_name="mtcnn")


def test_legacy_mediapipe_face_detector_converts_relative_boxes_to_pixel_boxes():
    fake_face_detection = MagicMock()
    fake_face_detection_namespace = SimpleNamespace(
        FaceDetection=MagicMock(return_value=fake_face_detection)
    )
    with patch.object(
        face_detection,
        "_import_legacy_face_detection",
        return_value=fake_face_detection_namespace,
    ):
        detector = face_detection.LegacyMediaPipeFaceDetector(
            model_selection=1,
            min_detection_confidence=0.65,
        )

    fake_detection = MagicMock()
    fake_detection.location_data.relative_bounding_box.xmin = 0.25
    fake_detection.location_data.relative_bounding_box.ymin = 0.10
    fake_detection.location_data.relative_bounding_box.width = 0.50
    fake_detection.location_data.relative_bounding_box.height = 0.60
    fake_face_detection.process.return_value = MagicMock(detections=[fake_detection])

    frames = [np.zeros((400, 800, 3), dtype=np.uint8)]
    detections = detector.detect(frames=frames, face_detect_width=200)

    assert len(detections) == 1
    assert detections[0].tolist() == [[200, 40, 600, 280]]


def test_community_face_detector_initializes_with_tasks_api():
    created = {}

    class FakeBaseOptions:
        def __init__(self, model_asset_path):
            created["model_asset_path"] = model_asset_path

    class FakeFaceDetectorOptions:
        def __init__(self, **kwargs):
            created["options_kwargs"] = kwargs

    class FakeFaceDetector:
        @classmethod
        def create_from_options(cls, options):
            created["options_object"] = options
            return "detector-instance"

    fake_mp = SimpleNamespace(
        tasks=SimpleNamespace(BaseOptions=FakeBaseOptions),
    )
    fake_vision = SimpleNamespace(
        FaceDetectorOptions=FakeFaceDetectorOptions,
        FaceDetector=FakeFaceDetector,
        RunningMode=SimpleNamespace(IMAGE="IMAGE"),
    )

    with patch.object(face_detection, "_import_mediapipe", return_value=fake_mp), patch.object(
        face_detection,
        "_import_mediapipe_tasks_vision",
        return_value=fake_vision,
    ), patch.object(
        face_detection,
        "_resolve_face_detector_model_path",
        return_value=Path("C:/models/mediapipe/blaze_face_short_range.tflite"),
    ), patch.object(
        face_detection,
        "_assert_model_asset_exists",
        return_value=None,
    ):
        detector = face_detection.CommunityMediaPipeFaceDetector(
            model_selection=0,
            min_detection_confidence=0.65,
        )

    assert detector._detector == "detector-instance"
    assert Path(created["model_asset_path"]) == Path(
        "C:/models/mediapipe/blaze_face_short_range.tflite"
    )
    assert created["options_kwargs"]["running_mode"] == "IMAGE"
    assert created["options_kwargs"]["min_detection_confidence"] == 0.65


def test_community_face_detector_converts_absolute_boxes_to_pixel_boxes():
    detector = face_detection.CommunityMediaPipeFaceDetector.__new__(
        face_detection.CommunityMediaPipeFaceDetector
    )

    class FakeImage:
        def __init__(self, image_format, data):
            self.image_format = image_format
            self.data = data

    detector._mp = SimpleNamespace(
        Image=FakeImage,
        ImageFormat=SimpleNamespace(SRGB="SRGB"),
    )
    detector._detector = MagicMock()
    detector._detector.detect.return_value = SimpleNamespace(
        detections=[
            SimpleNamespace(
                bounding_box=SimpleNamespace(
                    origin_x=50,
                    origin_y=10,
                    width=100,
                    height=70,
                )
            )
        ]
    )

    frames = [np.zeros((400, 800, 3), dtype=np.uint8)]
    detections = detector.detect(frames=frames, face_detect_width=200)

    assert len(detections) == 1
    assert detections[0].tolist() == [[200, 40, 600, 320]]


def test_build_face_landmarker_selects_community_path():
    with patch.object(face_detection, "CommunityMediaPipeFaceLandmarker") as landmarker_cls:
        sentinel = object()
        landmarker_cls.return_value = sentinel

        landmarker = face_detection.build_face_landmarker(
            diarization_model="community-1"
        )

    assert landmarker is sentinel
    landmarker_cls.assert_called_once_with()


def test_build_face_landmarker_selects_legacy_path():
    with patch.object(face_detection, "LegacyMediaPipeFaceLandmarker") as landmarker_cls:
        sentinel = object()
        landmarker_cls.return_value = sentinel

        landmarker = face_detection.build_face_landmarker(
            diarization_model="legacy-3.1"
        )

    assert landmarker is sentinel
    landmarker_cls.assert_called_once_with()


def test_community_face_landmarker_initializes_with_tasks_api():
    created = {}

    class FakeBaseOptions:
        def __init__(self, model_asset_path):
            created["model_asset_path"] = model_asset_path

    class FakeFaceLandmarkerOptions:
        def __init__(self, **kwargs):
            created["options_kwargs"] = kwargs

    class FakeFaceLandmarker:
        @classmethod
        def create_from_options(cls, options):
            created["options_object"] = options
            return "landmarker-instance"

    fake_mp = SimpleNamespace(
        tasks=SimpleNamespace(BaseOptions=FakeBaseOptions),
    )
    fake_vision = SimpleNamespace(
        FaceLandmarkerOptions=FakeFaceLandmarkerOptions,
        FaceLandmarker=FakeFaceLandmarker,
        RunningMode=SimpleNamespace(IMAGE="IMAGE"),
    )

    with patch.object(face_detection, "_import_mediapipe", return_value=fake_mp), patch.object(
        face_detection,
        "_import_mediapipe_tasks_vision",
        return_value=fake_vision,
    ), patch.object(
        face_detection,
        "_resolve_face_landmarker_model_path",
        return_value=Path("C:/models/mediapipe/face_landmarker.task"),
    ), patch.object(
        face_detection,
        "_assert_model_asset_exists",
        return_value=None,
    ):
        landmarker = face_detection.CommunityMediaPipeFaceLandmarker()

    assert landmarker._landmarker == "landmarker-instance"
    assert Path(created["model_asset_path"]) == Path(
        "C:/models/mediapipe/face_landmarker.task"
    )
    assert created["options_kwargs"]["running_mode"] == "IMAGE"
    assert created["options_kwargs"]["num_faces"] == 1
    assert created["options_kwargs"]["output_face_blendshapes"] is False
    assert created["options_kwargs"]["output_facial_transformation_matrixes"] is False


def test_community_face_landmarker_detect_returns_pixel_landmarks():
    landmarker = face_detection.CommunityMediaPipeFaceLandmarker.__new__(
        face_detection.CommunityMediaPipeFaceLandmarker
    )

    class FakeImage:
        def __init__(self, image_format, data):
            self.image_format = image_format
            self.data = data

    landmarker._mp = SimpleNamespace(
        Image=FakeImage,
        ImageFormat=SimpleNamespace(SRGB="SRGB"),
    )
    landmarker._landmarker = MagicMock()
    landmarker._landmarker.detect.return_value = SimpleNamespace(
        face_landmarks=[
            [
                SimpleNamespace(x=0.50, y=0.25),
                SimpleNamespace(x=0.25, y=0.75),
            ]
        ]
    )

    face = np.zeros((200, 100, 3), dtype=np.uint8)
    landmarks = landmarker.detect(face)

    assert np.allclose(
        landmarks,
        np.array([[50.0, 50.0], [25.0, 150.0]], dtype=np.float32),
    )


def test_missing_model_assets_raise_clear_actionable_error(tmp_path: Path):
    missing_path = tmp_path / "missing" / "face_landmarker.task"

    with pytest.raises(
        RuntimeError,
        match="CLIPSAI_MEDIAPIPE_FACE_LANDMARKER_MODEL_PATH",
    ):
        face_detection._assert_model_asset_exists(
            task_name="MediaPipe Face Landmarker",
            model_path=missing_path,
            env_var_name="CLIPSAI_MEDIAPIPE_FACE_LANDMARKER_MODEL_PATH",
        )


def test_resolve_face_detector_model_path_keeps_community_profile_on_numpy_2_ready_defaults():
    resolved = face_detection._resolve_face_detector_model_path(model_selection=0)

    assert resolved.name == "blaze_face_short_range.tflite"


def test_resolve_face_detector_model_path_supports_full_range_filename():
    resolved = face_detection._resolve_face_detector_model_path(model_selection=1)

    assert resolved.name == "blaze_face_full_range_sparse.tflite"
