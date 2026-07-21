"""
Diarize an audio file using supported pyannote speaker diarization pipelines.

Notes
-----
- Real-time factor is around 2.5% using one Nvidia Tesla V100 SXM2 GPU (for the neural
inference part) and one Intel Cascade Lake 6248 CPU (for the clustering part).
In other words, it takes approximately 1.5 minutes to process a one hour conversation.

- The legacy model details are described in
 https://huggingface.co/pyannote/speaker-diarization-3.1

- pyannote speaker diarization pipelines allow setting an exact or bounded number of
speakers to detect.
"""
# standard library imports
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import uuid

# local package imports
from .config import DEFAULT_DIARIZATION_MODEL
from .config import get_diarization_model_config
from clipsai.media.audio_file import AudioFile
from clipsai.utils.pytorch import get_compute_device, assert_compute_device_available

# third party imports
import pyannote.audio
from pyannote.audio import Pipeline
from pyannote.core.annotation import Annotation
import torch


def _patch_speechbrain_lazy_modules(module_map: dict | None = None) -> int:
    """
    Give SpeechBrain lazy redirect modules a harmless ``__file__`` value.

    Some newer SpeechBrain lazy modules trigger optional imports like ``k2`` when
    Python's inspect helpers ask whether the module has a ``__file__`` attribute.
    PyTorch Lightning calls into inspect while Pyannote is loading checkpoints,
    which can crash diarization even though k2 is unrelated to this workflow.
    """
    if module_map is None:
        try:
            import speechbrain  # noqa: F401
        except ImportError:
            return 0
        module_map = sys.modules

    patched = 0
    for module in list(module_map.values()):
        if module is None:
            continue

        module_type = type(module)
        if getattr(module_type, "__module__", "") != "speechbrain.utils.importutils":
            continue

        module_dict = getattr(module, "__dict__", None)
        if module_dict is None or "__file__" in module_dict:
            continue

        module.__file__ = "<lazy>"
        patched += 1

    return patched


def get_pyannote_audio_major_version(version_text: str | None = None) -> int:
    """
    Return the installed pyannote.audio major version number.
    """
    version_text = version_text or pyannote.audio.__version__
    return int(str(version_text).split(".")[0])


def build_pipeline_auth_kwargs(auth_token: str) -> dict:
    """
    Return the correct authentication keyword for the installed pyannote version.
    """
    if get_pyannote_audio_major_version() >= 4:
        return {"token": auth_token}
    return {"use_auth_token": auth_token}


def build_speaker_count_kwargs(
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict:
    """
    Validate and return supported pyannote speaker-count kwargs.
    """
    for field_name, value in (
        ("num_speakers", num_speakers),
        ("min_speakers", min_speakers),
        ("max_speakers", max_speakers),
    ):
        if value is not None and int(value) < 1:
            raise ValueError(f"{field_name} must be at least 1 when provided.")

    if num_speakers is not None and (
        min_speakers is not None or max_speakers is not None
    ):
        raise ValueError(
            "num_speakers cannot be combined with min_speakers or max_speakers."
        )

    if (
        min_speakers is not None
        and max_speakers is not None
        and int(min_speakers) > int(max_speakers)
    ):
        raise ValueError("min_speakers cannot be greater than max_speakers.")

    speaker_count_kwargs = {}
    if num_speakers is not None:
        speaker_count_kwargs["num_speakers"] = int(num_speakers)
    if min_speakers is not None:
        speaker_count_kwargs["min_speakers"] = int(min_speakers)
    if max_speakers is not None:
        speaker_count_kwargs["max_speakers"] = int(max_speakers)
    return speaker_count_kwargs


def extract_diarization_annotations(
    pipeline_output,
) -> tuple[Annotation, Annotation | None]:
    """
    Support both legacy Annotation outputs and newer wrapper-style outputs.
    """
    if isinstance(pipeline_output, Annotation):
        return pipeline_output, None

    regular_annotation = getattr(pipeline_output, "speaker_diarization", None)
    exclusive_annotation = getattr(pipeline_output, "exclusive_speaker_diarization", None)
    if isinstance(regular_annotation, Annotation):
        return regular_annotation, exclusive_annotation

    raise TypeError(
        "Unsupported pyannote pipeline output type. Expected an Annotation or an "
        "object with a speaker_diarization Annotation."
    )


def serialize_annotation(annotation: Annotation | None, time_precision: int) -> list[dict] | None:
    """
    Convert a pyannote Annotation into plain JSON-safe rows.
    """
    if annotation is None:
        return None

    serialized_segments = []
    for segment, track, speaker_label in annotation.itertracks(yield_label=True):
        serialized_segments.append(
            {
                "speaker_label": str(speaker_label),
                "track": str(track),
                "start_time": round(segment.start, time_precision),
                "end_time": round(segment.end, time_precision),
            }
        )
    return serialized_segments


def write_raw_diarization_payload(output_path: str | Path, payload: dict) -> Path:
    """
    Write raw diarization output JSON to disk.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_object:
        json.dump(payload, file_object, indent=2)
        file_object.write("\n")
    return output_path


class PyannoteDiarizer:
    """
    A class for diarizing audio files using supported pyannote pipelines.
    """

    def __init__(
        self,
        auth_token: str,
        device: str = None,
        diarization_model: str = DEFAULT_DIARIZATION_MODEL,
    ) -> None:
        """
        Initialize PyannoteDiarizer

        Parameters
        ----------
        auth_token: str
            Authentication token for Pyannote, obtained from HuggingFace.
        device: str
            PyTorch device to perform computations on. Ex: 'cpu', 'cuda'. Default is
            None (auto detects the correct device)
        diarization_model: str
            Which supported diarization pipeline should be loaded.

        Returns
        -------
        None
        """
        if device is None:
            device = get_compute_device()
        assert_compute_device_available(device)
        _patch_speechbrain_lazy_modules()
        self.model_name = diarization_model
        self.model_config = get_diarization_model_config(diarization_model)
        self.pipeline_checkpoint = self.model_config["checkpoint"]

        installed_pyannote_major = get_pyannote_audio_major_version()
        required_pyannote_major = self.model_config["minimum_pyannote_audio_major"]
        if installed_pyannote_major < required_pyannote_major:
            raise RuntimeError(
                f"Diarization model '{diarization_model}' requires pyannote.audio "
                f"{required_pyannote_major}.x or newer, but this environment has "
                f"{pyannote.audio.__version__}."
            )

        self.pipeline = Pipeline.from_pretrained(
            self.pipeline_checkpoint,
            **build_pipeline_auth_kwargs(auth_token),
        ).to(torch.device(device))
        logging.debug(
            "Pyannote using device: {} with model '{}' ({})".format(
                self.pipeline.device, self.model_name, self.pipeline_checkpoint
            )
        )

    def diarize(
        self,
        audio_file: AudioFile,
        min_segment_duration: float = 1.5,
        time_precision: int = 6,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        raw_output_path: str | Path | None = None,
    ) -> list[dict]:
        """
        Diarizes the audio file.

        Parameters
        ----------
        audio_file: AudioFile
            the audio file to diarize
        time_precision: int
            The number of decimal places for rounding the start and end times of
            segments.
        min_segment_duration: float
            The minimum duration (in seconds) for a segment to be considered valid.

        Returns
        -------
        speaker_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
        """
        speaker_count_kwargs = build_speaker_count_kwargs(
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        created_temp_wav = False
        if audio_file.has_file_extension("wav"):
            wav_file = audio_file
        else:
            created_temp_wav = True
            wav_file_path = os.path.join(
                audio_file.get_parent_dir_path(),
                "{}{}.wav".format(
                    audio_file.get_filename_without_extension(), str(uuid.uuid4().hex)
                ),
            )
            wav_file = audio_file.extract_audio(
                extracted_audio_file_path=wav_file_path,
                audio_codec="pcm_s16le",
                overwrite=False,
            )

        try:
            pipeline_output = self.pipeline(wav_file.path, **speaker_count_kwargs)
            pyannote_segments, exclusive_speaker_diarization = extract_diarization_annotations(
                pipeline_output
            )

            if raw_output_path is not None:
                raw_payload = {
                    "created_at": datetime.now().strftime("%m/%d/%Y %I:%M %p"),
                    "model_name": self.model_name,
                    "pipeline_checkpoint": self.pipeline_checkpoint,
                    "pyannote_audio_version": pyannote.audio.__version__,
                    "source_path": str(audio_file.path),
                    "analyzed_audio_path": str(wav_file.path),
                    "speaker_count_constraints": speaker_count_kwargs,
                    "speaker_diarization": serialize_annotation(
                        pyannote_segments,
                        time_precision,
                    ),
                    "exclusive_speaker_diarization": serialize_annotation(
                        exclusive_speaker_diarization,
                        time_precision,
                    ),
                }
                write_raw_diarization_payload(raw_output_path, raw_payload)

            adjusted_annotation = exclusive_speaker_diarization or pyannote_segments
            adjusted_speaker_segments = self._adjust_segments(
                pyannote_segments=adjusted_annotation,
                min_segment_duration=min_segment_duration,
                duration=audio_file.get_duration(),
                time_precision=time_precision,
            )
        finally:
            if created_temp_wav:
                wav_file.delete()

        return adjusted_speaker_segments

    def _adjust_segments(
        self,
        pyannote_segments: Annotation,
        min_segment_duration: float,
        duration: float,
        time_precision: int,
    ) -> list[dict]:
        """
        Adjusts and merges speaker segments to achieve an unbroken, non-overlapping
        sequence of speaker segments with at least one person speaking in each segment.

        Parameters
        ----------
        pyannote_segments: Annotation
            the pyannote speaker segments
        duration: float
            duration of the audio being diarized.
        time_precision: int
            The number of decimal places for rounding the start and end times of
            segments.
        min_segment_duration: float
            The minimum duration (in seconds) for a segment to be considered valid.

        Returns
        -------
        speaker_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
        """
        cur_end_time = None
        cur_speaker = None
        cur_start_time = 0.000
        adjusted_speaker_segments = []
        unique_speakers: set[int] = set()

        for segment, _, speaker_label in pyannote_segments.itertracks(True):
            next_start_time = segment.start
            next_end_time = segment.end
            if speaker_label.split("_")[1] == "":
                next_speaker = None
            else:
                next_speaker = int(speaker_label.split("_")[1])

            # skip segments that are too short
            if next_end_time - next_start_time < min_segment_duration:
                continue

            # first identified speaker
            if cur_speaker is None:
                cur_speaker = next_speaker
                cur_end_time = next_end_time
                continue

            # same speaker as next segment -> merge segments and continue
            if cur_speaker == next_speaker:
                cur_end_time = max(cur_end_time, next_end_time)
                continue

            # Different speaker than next segment
            # 1) The next speaker begins before the current speaker ends -> cut short
            # the end of the current speaker's segment be the start of the next
            # speaker segment.
            # 2) The next speaker begins after the current speaker ends -> extend the
            # current speaker's segment to end at the start of the next speaker segment.
            cur_end_time = next_start_time
            if cur_speaker is not None:
                speakers = [cur_speaker]
                unique_speakers.add(cur_speaker)
            else:
                speakers = []
            adjusted_speaker_segments.append(
                {
                    "speakers": speakers,
                    "start_time": round(cur_start_time, time_precision),
                    "end_time": round(cur_end_time, time_precision),
                }
            )

            cur_speaker = next_speaker
            cur_start_time = next_start_time
            cur_end_time = next_end_time

        # explicitly add the last segment
        if cur_speaker is not None:
            speakers = [cur_speaker]
            unique_speakers.add(cur_speaker)
        else:
            speakers = []
        adjusted_speaker_segments.append(
            {
                "speakers": speakers,
                "start_time": round(cur_start_time, time_precision),
                "end_time": round(duration, time_precision),
            }
        )

        adjusted_speaker_segments = self._relabel_speakers(
            adjusted_speaker_segments, unique_speakers
        )
        return adjusted_speaker_segments

    def _relabel_speakers(
        self, speaker_segments: list[dict], unique_speakers: set[int]
    ) -> list[dict]:
        """
        Relabels speaker segments so that the speaker labels are contiguous.

        Some speakers may have been skipped if their segments were too short. Thus,
        we could end up with a set of speaker labels like {0, 1, 3}. This function
        relabels the speakers to remove gaps so that our set of speaker labels would
        be contiguous, e.g. {0, 1, 2}.

        Parameters
        ----------
        speaker_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
        unique_speakers: set[int]
            set of unique speaker labels in the speaker segments

        Returns
        -------
        updated_speaker_segments: list[dict]
            list of speaker segments where the speakers are relabeled so that the
            speaker labels are contiguous. Each dictionary contains the following keys:
                speakers: list[int]
                    list of speaker numbers for the speakers talking in the segment
                start_time: float
                    start time of the segment in seconds
                end_time: float
                    end time of the segment in seconds
        """
        # no speakers
        if len(unique_speakers) == 0:
            return speaker_segments

        unique_speakers = sorted(list(unique_speakers))
        # speaker labels are already contiguous
        if len(unique_speakers) == unique_speakers[-1] + 1:
            return speaker_segments

        # create mapping from old speaker labels to new speaker labels
        relabel_speaker_map = {}
        for i in range(len(unique_speakers)):
            new_speaker_num = i
            old_speaker_num = unique_speakers[i]
            relabel_speaker_map[old_speaker_num] = new_speaker_num

        # relabel
        for segment in speaker_segments:
            relabeled_speakers = []
            for speaker in segment["speakers"]:
                relabeled_speakers.append(relabel_speaker_map[speaker])
            segment["speakers"] = relabeled_speakers

        return speaker_segments

    def cleanup(self) -> None:
        """
        Remove the diarization pipeline from memory and explicity free up GPU memory.
        """
        del self.pipeline
        self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
