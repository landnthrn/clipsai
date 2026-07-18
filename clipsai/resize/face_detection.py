"""
Face-detection backend wrappers used by the resizer.
"""

from importlib import import_module
import logging

import cv2
import mediapipe as mp
import numpy as np
import torch

from .config import assert_supported_face_detect_backend
from .config import DEFAULT_FACE_DETECT_BACKEND
from .config import DEFAULT_MEDIAPIPE_FACE_DETECT_MIN_DETECTION_CONFIDENCE
from .config import DEFAULT_MEDIAPIPE_FACE_DETECT_MODEL_SELECTION


def _import_mtcnn():
    """
    Import the optional facenet-pytorch MTCNN class only when needed.
    """
    try:
        facenet_module = import_module("facenet_pytorch")
    except ImportError as exc:
        raise RuntimeError(
            "Face-detection backend 'mtcnn' requires the optional "
            "'facenet-pytorch' package. Install the legacy dependency profile or "
            "switch to the 'mediapipe' face-detection backend."
        ) from exc
    return facenet_module.MTCNN


def _resize_frames_for_detection(
    frames: list[np.ndarray],
    face_detect_width: int,
) -> tuple[list[np.ndarray], float, int]:
    """
    Resize frames to the requested detection width and return scaling info.
    """
    downsample_factor = max(frames[0].shape[1] / face_detect_width, 1)
    detect_height = int(frames[0].shape[0] / downsample_factor)
    resized_frames = [
        cv2.resize(frame, (face_detect_width, detect_height)) for frame in frames
    ]
    return resized_frames, downsample_factor, detect_height


class MtcnnFaceDetector:
    """
    Face-detection wrapper around facenet-pytorch MTCNN.
    """

    def __init__(
        self,
        face_detect_margin: int = 20,
        face_detect_post_process: bool = False,
        device: str = None,
    ) -> None:
        mtcnn_cls = _import_mtcnn()
        self._detector = mtcnn_cls(
            margin=face_detect_margin,
            post_process=face_detect_post_process,
            device=device,
        )

    def detect(
        self,
        frames: list[np.ndarray],
        face_detect_width: int,
    ) -> list[np.ndarray]:
        """
        Detect faces in frames and return MTCNN-style bounding boxes.
        """
        if len(frames) == 0:
            logging.debug("No frames to detect faces in.")
            return []

        logging.debug("Detecting faces in {} frames with MTCNN.".format(len(frames)))
        resized_frames, downsample_factor, _ = _resize_frames_for_detection(
            frames, face_detect_width
        )
        if torch.cuda.is_available():
            resized_frames = [
                torch.from_numpy(frame).to(device="cuda", dtype=torch.uint8)
                for frame in resized_frames
            ]
            resized_frames = torch.stack(resized_frames)

        detections, _ = self._detector.detect(resized_frames)

        face_detections = []
        for detection in detections:
            if detection is not None:
                detection[detection < 0] = 0
                detection = (detection * downsample_factor).astype(np.int16)
            face_detections.append(detection)

        logging.debug("Detected faces in {} frames.".format(len(face_detections)))
        return face_detections

    def cleanup(self) -> None:
        """
        Free detector resources.
        """
        del self._detector
        self._detector = None


class MediaPipeFaceDetector:
    """
    Face-detection wrapper around MediaPipe Face Detection.
    """

    def __init__(
        self,
        model_selection: int = DEFAULT_MEDIAPIPE_FACE_DETECT_MODEL_SELECTION,
        min_detection_confidence: float = (
            DEFAULT_MEDIAPIPE_FACE_DETECT_MIN_DETECTION_CONFIDENCE
        ),
    ) -> None:
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(
        self,
        frames: list[np.ndarray],
        face_detect_width: int,
    ) -> list[np.ndarray]:
        """
        Detect faces in frames and return bounding boxes shaped like the MTCNN path.
        """
        if len(frames) == 0:
            logging.debug("No frames to detect faces in.")
            return []

        logging.debug(
            "Detecting faces in {} frames with MediaPipe Face Detection.".format(
                len(frames)
            )
        )
        resized_frames, downsample_factor, detect_height = _resize_frames_for_detection(
            frames, face_detect_width
        )
        face_detections = []

        for frame in resized_frames:
            results = self._detector.process(frame)
            if not results.detections:
                face_detections.append(None)
                continue

            detections = []
            for detection in results.detections:
                bounding_box = detection.location_data.relative_bounding_box
                x1 = max(0.0, min(1.0, bounding_box.xmin))
                y1 = max(0.0, min(1.0, bounding_box.ymin))
                x2 = max(0.0, min(1.0, bounding_box.xmin + bounding_box.width))
                y2 = max(0.0, min(1.0, bounding_box.ymin + bounding_box.height))
                detections.append(
                    [
                        x1 * face_detect_width,
                        y1 * detect_height,
                        x2 * face_detect_width,
                        y2 * detect_height,
                    ]
                )

            face_detections.append(
                (np.array(detections, dtype=np.float32) * downsample_factor).astype(
                    np.int16
                )
            )

        logging.debug("Detected faces in {} frames.".format(len(face_detections)))
        return face_detections

    def cleanup(self) -> None:
        """
        Free detector resources.
        """
        self._detector.close()
        self._detector = None


def build_face_detector(
    backend_name: str = DEFAULT_FACE_DETECT_BACKEND,
    face_detect_margin: int = 20,
    face_detect_post_process: bool = False,
    device: str = None,
    mediapipe_face_detect_model_selection: int = (
        DEFAULT_MEDIAPIPE_FACE_DETECT_MODEL_SELECTION
    ),
    mediapipe_face_detect_min_detection_confidence: float = (
        DEFAULT_MEDIAPIPE_FACE_DETECT_MIN_DETECTION_CONFIDENCE
    ),
):
    """
    Build one supported face-detection backend instance.
    """
    assert_supported_face_detect_backend(backend_name)

    if backend_name == "mtcnn":
        return MtcnnFaceDetector(
            face_detect_margin=face_detect_margin,
            face_detect_post_process=face_detect_post_process,
            device=device,
        )

    return MediaPipeFaceDetector(
        model_selection=mediapipe_face_detect_model_selection,
        min_detection_confidence=mediapipe_face_detect_min_detection_confidence,
    )
