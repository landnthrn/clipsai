"""
Shared resize and face-detection configuration.
"""

DEFAULT_FACE_DETECT_BACKEND = "mtcnn"
DEFAULT_MEDIAPIPE_FACE_DETECT_MODEL_SELECTION = 0
DEFAULT_MEDIAPIPE_FACE_DETECT_MIN_DETECTION_CONFIDENCE = 0.5

FACE_DETECT_BACKENDS = {
    "mtcnn": {
        "label": "FaceNet MTCNN",
    },
    "mediapipe": {
        "label": "MediaPipe Face Detection",
    },
}


def get_supported_face_detect_backends() -> list[str]:
    """
    Return supported face-detection backend names.
    """
    return sorted(FACE_DETECT_BACKENDS.keys())


def assert_supported_face_detect_backend(backend_name: str) -> None:
    """
    Validate a supported face-detection backend name.
    """
    if backend_name not in FACE_DETECT_BACKENDS:
        supported_backends_text = ", ".join(sorted(FACE_DETECT_BACKENDS.keys()))
        raise ValueError(
            f"Unsupported face-detection backend '{backend_name}'. "
            f"Supported backends: {supported_backends_text}"
        )
