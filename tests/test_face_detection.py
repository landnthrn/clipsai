import builtins
import importlib
import sys
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

    assert hasattr(module, "MediaPipeFaceDetector")
    assert hasattr(module, "MtcnnFaceDetector")


def test_build_face_detector_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported face-detection backend"):
        face_detection.build_face_detector(backend_name="unknown-backend")


def test_build_face_detector_selects_mediapipe_backend():
    with patch.object(face_detection, "MediaPipeFaceDetector") as detector_cls:
        sentinel = object()
        detector_cls.return_value = sentinel

        detector = face_detection.build_face_detector(backend_name="mediapipe")

    assert detector is sentinel
    detector_cls.assert_called_once_with(
        model_selection=0,
        min_detection_confidence=0.5,
    )


def test_build_face_detector_creates_mediapipe_without_facenet_pytorch():
    with patch(
        "clipsai.resize.face_detection.import_module",
        side_effect=AssertionError("facenet-pytorch should not be imported"),
    ), patch.object(face_detection, "MediaPipeFaceDetector") as detector_cls:
        sentinel = object()
        detector_cls.return_value = sentinel

        detector = face_detection.build_face_detector(backend_name="mediapipe")

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


def test_mediapipe_face_detector_converts_relative_boxes_to_pixel_boxes():
    fake_face_detection = MagicMock()
    with patch(
        "clipsai.resize.face_detection.mp.solutions.face_detection.FaceDetection",
        return_value=fake_face_detection,
    ):
        detector = face_detection.MediaPipeFaceDetector(
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
