from importlib import import_module


_EXPORTS = {
    "AudioFile": ("clipsai.media.audio_file", "AudioFile"),
    "AudioVideoFile": ("clipsai.media.audiovideo_file", "AudioVideoFile"),
    "Character": ("clipsai.transcribe.transcription_element", "Character"),
    "ClipFinder": ("clipsai.clip.clipfinder", "ClipFinder"),
    "Clip": ("clipsai.clip.clip", "Clip"),
    "Crops": ("clipsai.resize.crops", "Crops"),
    "MediaEditor": ("clipsai.media.editor", "MediaEditor"),
    "Segment": ("clipsai.resize.segment", "Segment"),
    "Sentence": ("clipsai.transcribe.transcription_element", "Sentence"),
    "Transcriber": ("clipsai.transcribe.transcriber", "Transcriber"),
    "Transcription": ("clipsai.transcribe.transcription", "Transcription"),
    "VideoFile": ("clipsai.media.video_file", "VideoFile"),
    "Word": ("clipsai.transcribe.transcription_element", "Word"),
    "resize": ("clipsai.resize.resize", "resize"),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'clipsai' has no attribute '{name}'")

    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
