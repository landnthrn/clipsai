from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from clipsai.resize.face_detection import MediaPipeFaceDetector
from clipsai.resize.face_detection import build_face_detector


def test_build_face_detector_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported face-detection backend"):
        build_face_detector(backend_name="unknown-backend")


def test_build_face_detector_selects_mediapipe_backend():
    with patch("clipsai.resize.face_detection.MediaPipeFaceDetector") as detector_cls:
        sentinel = object()
        detector_cls.return_value = sentinel

        detector = build_face_detector(backend_name="mediapipe")

    assert detector is sentinel
    detector_cls.assert_called_once_with(
        model_selection=0,
        min_detection_confidence=0.5,
    )


def test_mediapipe_face_detector_converts_relative_boxes_to_pixel_boxes():
    fake_face_detection = MagicMock()
    with patch(
        "clipsai.resize.face_detection.mp.solutions.face_detection.FaceDetection",
        return_value=fake_face_detection,
    ):
        detector = MediaPipeFaceDetector(
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
