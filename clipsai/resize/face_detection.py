"""
Face-detection and face-landmark backend wrappers used by the resizer.
"""

from importlib import import_module
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from clipsai.diarize.config import DEFAULT_DIARIZATION_MODEL

from .config import assert_supported_face_detect_backend
from .config import DEFAULT_FACE_DETECT_BACKEND
from .config import DEFAULT_MEDIAPIPE_FACE_DETECT_MIN_DETECTION_CONFIDENCE
from .config import DEFAULT_MEDIAPIPE_FACE_DETECT_MODEL_SELECTION


COMMUNITY_DIARIZATION_MODEL = "community-1"
MEDIAPIPE_MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "mediapipe"
MEDIAPIPE_FACE_DETECTOR_DEFAULT_FILENAMES = {
    0: "blaze_face_short_range.tflite",
    1: "blaze_face_full_range_sparse.tflite",
}
MEDIAPIPE_FACE_DETECTOR_ENV_VARS = {
    0: "CLIPSAI_MEDIAPIPE_FACE_DETECTOR_MODEL_PATH",
    1: "CLIPSAI_MEDIAPIPE_FACE_DETECTOR_FULL_RANGE_MODEL_PATH",
}
MEDIAPIPE_FACE_LANDMARKER_DEFAULT_FILENAME = "face_landmarker.task"
MEDIAPIPE_FACE_LANDMARKER_ENV_VAR = "CLIPSAI_MEDIAPIPE_FACE_LANDMARKER_MODEL_PATH"


def _uses_community_mediapipe_runtime(diarization_model: str) -> bool:
    """
    Return whether this diarization mode should use the modern Tasks runtime.
    """
    return diarization_model == COMMUNITY_DIARIZATION_MODEL


def _import_mediapipe():
    """
    Import mediapipe only when a MediaPipe-backed path is actually requested.
    """
    try:
        return import_module("mediapipe")
    except ImportError as exc:
        raise RuntimeError(
            "The selected MediaPipe backend requires the optional 'mediapipe' "
            "package. Install the matching dependency profile for this run."
        ) from exc


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


def _import_legacy_face_detection():
    """
    Import the legacy MediaPipe Solutions face-detection API lazily.
    """
    return _import_mediapipe().solutions.face_detection


def _import_legacy_face_mesh():
    """
    Import the legacy MediaPipe Solutions face-mesh API lazily.
    """
    return _import_mediapipe().solutions.face_mesh


def _import_mediapipe_tasks_vision():
    """
    Import the modern MediaPipe Tasks vision namespace lazily.
    """
    return import_module("mediapipe.tasks.python.vision")


def _resolve_face_detector_model_path(
    model_selection: int,
    model_path: str | None = None,
) -> Path:
    """
    Resolve the configured MediaPipe Face Detector model path.
    """
    if model_selection not in MEDIAPIPE_FACE_DETECTOR_DEFAULT_FILENAMES:
        raise ValueError(
            "Unsupported MediaPipe face-detection model selection "
            f"'{model_selection}'. Expected 0 (short-range) or 1 (full-range)."
        )

    env_var_name = MEDIAPIPE_FACE_DETECTOR_ENV_VARS[model_selection]
    candidate = model_path or os.environ.get(env_var_name)
    if candidate:
        return Path(candidate).expanduser()

    return MEDIAPIPE_MODELS_DIR / MEDIAPIPE_FACE_DETECTOR_DEFAULT_FILENAMES[
        model_selection
    ]


def _resolve_face_landmarker_model_path(model_path: str | None = None) -> Path:
    """
    Resolve the configured MediaPipe Face Landmarker model path.
    """
    candidate = model_path or os.environ.get(MEDIAPIPE_FACE_LANDMARKER_ENV_VAR)
    if candidate:
        return Path(candidate).expanduser()
    return MEDIAPIPE_MODELS_DIR / MEDIAPIPE_FACE_LANDMARKER_DEFAULT_FILENAME


def _assert_model_asset_exists(
    task_name: str,
    model_path: Path,
    env_var_name: str,
) -> None:
    """
    Raise a clear actionable error when one required MediaPipe model is missing.
    """
    if model_path.exists():
        return

    raise RuntimeError(
        f"{task_name} requires a local MediaPipe Tasks model asset, but the file was "
        f"not found at '{model_path}'. Place the model at that path or set the "
        f"'{env_var_name}' environment variable to a valid local model file."
    )


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


def _scale_normalized_landmarks_to_pixels(
    landmarks,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """
    Convert normalized MediaPipe landmarks into image-pixel coordinates.
    """
    points = [[landmark.x, landmark.y] for landmark in landmarks]
    points = np.array(points, dtype=np.float32)
    points[:, 0] *= image_width
    points[:, 1] *= image_height
    return points


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

        logging.debug("Detecting faces in %s frames with MTCNN.", len(frames))
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

        logging.debug("Detected faces in %s frames.", len(face_detections))
        return face_detections

    def cleanup(self) -> None:
        """
        Free detector resources.
        """
        del self._detector
        self._detector = None


class LegacyMediaPipeFaceDetector:
    """
    Face-detection wrapper around legacy MediaPipe Solutions face detection.
    """

    def __init__(
        self,
        model_selection: int = DEFAULT_MEDIAPIPE_FACE_DETECT_MODEL_SELECTION,
        min_detection_confidence: float = (
            DEFAULT_MEDIAPIPE_FACE_DETECT_MIN_DETECTION_CONFIDENCE
        ),
    ) -> None:
        face_detection = _import_legacy_face_detection()
        self._detector = face_detection.FaceDetection(
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
            "Detecting faces in %s frames with legacy MediaPipe Face Detection.",
            len(frames),
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

        logging.debug("Detected faces in %s frames.", len(face_detections))
        return face_detections

    def cleanup(self) -> None:
        """
        Free detector resources.
        """
        self._detector.close()
        self._detector = None


class CommunityMediaPipeFaceDetector:
    """
    Face-detection wrapper around modern MediaPipe Tasks face detection.
    """

    def __init__(
        self,
        model_selection: int = DEFAULT_MEDIAPIPE_FACE_DETECT_MODEL_SELECTION,
        min_detection_confidence: float = (
            DEFAULT_MEDIAPIPE_FACE_DETECT_MIN_DETECTION_CONFIDENCE
        ),
        model_path: str | None = None,
    ) -> None:
        mp = _import_mediapipe()
        vision = _import_mediapipe_tasks_vision()
        resolved_model_path = _resolve_face_detector_model_path(
            model_selection=model_selection,
            model_path=model_path,
        )
        _assert_model_asset_exists(
            task_name="MediaPipe Face Detector",
            model_path=resolved_model_path,
            env_var_name=MEDIAPIPE_FACE_DETECTOR_ENV_VARS[model_selection],
        )

        options = vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(resolved_model_path)
            ),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=min_detection_confidence,
        )
        self._mp = mp
        self._detector = vision.FaceDetector.create_from_options(options)

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
            "Detecting faces in %s frames with MediaPipe Tasks Face Detector.",
            len(frames),
        )
        resized_frames, downsample_factor, detect_height = _resize_frames_for_detection(
            frames, face_detect_width
        )
        face_detections = []

        for frame in resized_frames:
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=frame,
            )
            results = self._detector.detect(mp_image)
            if not results.detections:
                face_detections.append(None)
                continue

            detections = []
            for detection in results.detections:
                bounding_box = detection.bounding_box
                x1 = max(0, int(bounding_box.origin_x))
                y1 = max(0, int(bounding_box.origin_y))
                x2 = min(
                    face_detect_width,
                    int(bounding_box.origin_x + bounding_box.width),
                )
                y2 = min(
                    detect_height,
                    int(bounding_box.origin_y + bounding_box.height),
                )
                detections.append([x1, y1, x2, y2])

            face_detections.append(
                (np.array(detections, dtype=np.float32) * downsample_factor).astype(
                    np.int16
                )
            )

        logging.debug("Detected faces in %s frames.", len(face_detections))
        return face_detections

    def cleanup(self) -> None:
        """
        Free detector resources.
        """
        self._detector.close()
        self._detector = None
        self._mp = None


class LegacyMediaPipeFaceLandmarker:
    """
    Face-landmark wrapper around legacy MediaPipe Solutions Face Mesh.
    """

    def __init__(self) -> None:
        face_mesh = _import_legacy_face_mesh()
        self._landmarker = face_mesh.FaceMesh()

    def detect(self, face: np.ndarray) -> np.ndarray | None:
        """
        Detect one face mesh and return pixel landmarks for the first face.
        """
        results = self._landmarker.process(face)
        if results.multi_face_landmarks is None:
            return None

        return _scale_normalized_landmarks_to_pixels(
            landmarks=results.multi_face_landmarks[0].landmark,
            image_width=face.shape[1],
            image_height=face.shape[0],
        )

    def close(self) -> None:
        """
        Free landmarker resources.
        """
        self._landmarker.close()
        self._landmarker = None


class CommunityMediaPipeFaceLandmarker:
    """
    Face-landmark wrapper around modern MediaPipe Tasks Face Landmarker.
    """

    def __init__(self, model_path: str | None = None) -> None:
        mp = _import_mediapipe()
        vision = _import_mediapipe_tasks_vision()
        resolved_model_path = _resolve_face_landmarker_model_path(model_path)
        _assert_model_asset_exists(
            task_name="MediaPipe Face Landmarker",
            model_path=resolved_model_path,
            env_var_name=MEDIAPIPE_FACE_LANDMARKER_ENV_VAR,
        )

        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(resolved_model_path)
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, face: np.ndarray) -> np.ndarray | None:
        """
        Detect one face mesh and return pixel landmarks for the first face.
        """
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=face,
        )
        results = self._landmarker.detect(mp_image)
        if not results.face_landmarks:
            return None

        return _scale_normalized_landmarks_to_pixels(
            landmarks=results.face_landmarks[0],
            image_width=face.shape[1],
            image_height=face.shape[0],
        )

    def close(self) -> None:
        """
        Free landmarker resources.
        """
        self._landmarker.close()
        self._landmarker = None
        self._mp = None


MediaPipeFaceDetector = LegacyMediaPipeFaceDetector


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
    diarization_model: str = DEFAULT_DIARIZATION_MODEL,
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

    detector_cls = LegacyMediaPipeFaceDetector
    if _uses_community_mediapipe_runtime(diarization_model):
        detector_cls = CommunityMediaPipeFaceDetector

    return detector_cls(
        model_selection=mediapipe_face_detect_model_selection,
        min_detection_confidence=(
            mediapipe_face_detect_min_detection_confidence
        ),
    )


def build_face_landmarker(
    diarization_model: str = DEFAULT_DIARIZATION_MODEL,
):
    """
    Build the matching MediaPipe face-landmarker implementation for one runtime.
    """
    if _uses_community_mediapipe_runtime(diarization_model):
        return CommunityMediaPipeFaceLandmarker()
    return LegacyMediaPipeFaceLandmarker()
