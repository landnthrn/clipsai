"""
Shared diarization model configuration.
"""

DEFAULT_DIARIZATION_MODEL = "legacy-3.1"

DIARIZATION_MODELS = {
    "legacy-3.1": {
        "checkpoint": "pyannote/speaker-diarization-3.1",
        "minimum_pyannote_audio_major": 3,
    },
    "community-1": {
        "checkpoint": "pyannote/speaker-diarization-community-1",
        "minimum_pyannote_audio_major": 4,
    },
}


def get_diarization_model_config(model_name: str) -> dict:
    """
    Return configuration for one supported diarization model.
    """
    if model_name not in DIARIZATION_MODELS:
        supported_models_text = ", ".join(sorted(DIARIZATION_MODELS.keys()))
        raise ValueError(
            f"Unsupported diarization model '{model_name}'. "
            f"Supported models: {supported_models_text}"
        )
    return DIARIZATION_MODELS[model_name]


def get_supported_diarization_models() -> list[str]:
    """
    Return supported diarization model names.
    """
    return sorted(DIARIZATION_MODELS.keys())
