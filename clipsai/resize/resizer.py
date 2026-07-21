"""
Resizing/cropping a media file to a different aspect ratio

Notes
-----
- ROI is "region of interest"
"""
# standard library imports
import logging

# current package imports
from .config import DEFAULT_FACE_DETECT_BACKEND
from .config import get_default_mediapipe_face_detect_min_detection_confidence
from .crops import Crops
from .exceptions import ResizerError
from .face_detection import build_face_detector
from .face_detection import build_face_landmarker
from .img_proc import calc_img_bytes
from .rect import Rect
from .segment import Segment
from .vid_proc import extract_frames

# local package imports
from clipsai.diarize.config import DEFAULT_DIARIZATION_MODEL
from clipsai.media.editor import MediaEditor
from clipsai.media.video_file import VideoFile
from clipsai.utils import pytorch
from clipsai.utils.conversions import bytes_to_gibibytes

# 3rd party imports
import cv2
import numpy as np
from sklearn.cluster import KMeans
import torch


MOUTH_ANALYSIS_CROP_MARGIN_RATIO = 0.75
MOUTH_ANALYSIS_MIN_FACE_SIZE_PIXELS = 256
MOUTH_MOVEMENT_MIN_SCORE = 0.01
MOUTH_MOVEMENT_MIN_LANDMARK_FRAMES = 2
MOUTH_MOVEMENT_MIN_RATIO = 1.25
MOUTH_MOVEMENT_MIN_DELTA = 0.015


class Resizer:
    """
    A class for calculating the initial coordinates for resizing by using
    segmentation and face detection.
    """

    def __init__(
        self,
        face_detect_margin: int = 20,
        face_detect_post_process: bool = False,
        face_detect_backend: str = DEFAULT_FACE_DETECT_BACKEND,
        mediapipe_face_detect_model_selection: int | None = None,
        mediapipe_face_detect_min_detection_confidence: float | None = None,
        diarization_model: str = DEFAULT_DIARIZATION_MODEL,
        device: str = None,
    ) -> None:
        """
        Initializes the Resizer with specific configurations for face
        detection. This class uses a configurable face-detection backend plus a
        MediaPipe landmark runtime for analyzing mouth movement to determine who is
        speaking within video frames.

        Parameters
        ----------
        face_detect_margin: int, optional
            The margin around detected faces, specified in pixels. Increasing this
            value results in a larger area around each detected face being included.
            Default is 20 pixels.
        face_detect_post_process: bool, optional
            Determines whether to apply post-processing on the detected faces. Setting
            this to False prevents normalization of output images, making them appear
            more natural to the human eye. Default is False (no post-processing).
        face_detect_backend: str, optional
            Which supported face-detection backend should be used.
        mediapipe_face_detect_model_selection: int, optional
            MediaPipe Face Detection model selection. `0` is short-range, `1` is
            full-range, and omitted values use the default for the diarization mode.
        mediapipe_face_detect_min_detection_confidence: float, optional
            Minimum MediaPipe face-detection confidence.
        diarization_model: str, optional
            Which diarization profile this resize run belongs to. Community runs use
            the modern MediaPipe Tasks path while legacy runs keep the older
            MediaPipe Solutions path.
        device: str, optional
            PyTorch device to perform computations on. Ex: 'cpu', 'cuda'. Default is
            None (auto detects the correct device)
        """
        if device is None:
            device = pytorch.get_compute_device()
        pytorch.assert_compute_device_available(device)
        logging.debug("Face-detection backend '{}' using device: {}".format(face_detect_backend, device))

        self._face_detect_backend = face_detect_backend
        if mediapipe_face_detect_min_detection_confidence is None:
            mediapipe_face_detect_min_detection_confidence = (
                get_default_mediapipe_face_detect_min_detection_confidence(
                    diarization_model
                )
            )
        self._face_detector = build_face_detector(
            backend_name=face_detect_backend,
            face_detect_margin=face_detect_margin,
            face_detect_post_process=face_detect_post_process,
            device=device,
            mediapipe_face_detect_model_selection=mediapipe_face_detect_model_selection,
            mediapipe_face_detect_min_detection_confidence=(
                mediapipe_face_detect_min_detection_confidence
            ),
            diarization_model=diarization_model,
        )
        self._face_landmarker = build_face_landmarker(
            diarization_model=diarization_model
        )
        self._media_editor = MediaEditor()

    def resize(
        self,
        video_file: VideoFile,
        speaker_segments: list[dict],
        scene_changes: list[float],
        aspect_ratio: tuple = (9, 16),
        samples_per_segment: int = 13,
        face_detect_width: int = 960,
        n_face_detect_batches: int = 8,
        scene_merge_threshold: float = 0.25,
    ) -> Crops:
        """
        Calculates the coordinates to resize the video to for different
        segments given the diarized speaker segments and the desired aspect
        ratio.

        Parameters
        ----------
        video_file: VideoFile
            The video file to resize
        speaker_segments: list[dict]
            speakers: list[int]
                list of speakers (represented by int) talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
        scene_changes: list[float]
            List of scene change times in seconds
        aspect_ratio: tuple[int, int]
            The (width,height) aspect ratio to resize the video to
        samples_per_segment: int
            Number of frames to sample per segment for face detection.
        face_detect_width: int
            The width to use for face detection
        n_face_detect_batches: int
            Number of batches for GPU face detection in a video file
        scene_merge_threshold: float
            The threshold in seconds for merging scene changes with speaker segments.
            Scene changes within this threshold of a segment's start or end time will
            cause the segment to be adjusted.

        Returns
        -------
        Crops
            the resized speaker segments
        """
        logging.debug(
            "Video Resolution: {}x{}".format(
                video_file.get_width_pixels(), video_file.get_height_pixels()
            )
        )
        # calculate resize dimensions
        resize_width, resize_height = self._calc_resize_width_and_height_pixels(
            original_width_pixels=video_file.get_width_pixels(),
            original_height_pixels=video_file.get_height_pixels(),
            resize_aspect_ratio=aspect_ratio,
        )

        logging.debug(
            "Merging {} speaker segments with {} scene changes.".format(
                len(speaker_segments), len(scene_changes)
            )
        )
        segments = self._merge_scene_change_and_speaker_segments(
            speaker_segments, scene_changes, scene_merge_threshold
        )
        logging.debug("Video has {} distinct segments.".format(len(segments)))

        logging.debug("Determining the first second with a face for each segment.")
        segments = self._find_first_sec_with_face_for_each_segment(
            segments, video_file, face_detect_width, n_face_detect_batches
        )

        logging.debug(
            "Determining the region of interest for {} segments.".format(len(segments))
        )
        segments = self._add_x_y_coords_to_each_segment(
            segments,
            video_file,
            resize_width,
            resize_height,
            samples_per_segment,
            face_detect_width,
            n_face_detect_batches,
        )

        logging.debug("Merging identical segments together.")
        unmerge_segments_length = len(segments)
        segments = self._merge_identical_segments(segments, video_file)
        logging.debug(
            "Merged {} identical segments.".format(
                unmerge_segments_length - len(segments)
            )
        )

        crop_segments = []
        for segment in segments:
            crop_segments.append(
                Segment(
                    speakers=segment["speakers"],
                    start_time=segment["start_time"],
                    end_time=segment["end_time"],
                    x=segment["x"],
                    y=segment["y"],
                    crop_selection=segment.get("crop_selection"),
                )
            )

        crops = Crops(
            original_width=video_file.get_width_pixels(),
            original_height=video_file.get_height_pixels(),
            crop_width=resize_width,
            crop_height=resize_height,
            segments=crop_segments,
        )

        return crops

    def _calc_resize_width_and_height_pixels(
        self,
        original_width_pixels: int,
        original_height_pixels: int,
        resize_aspect_ratio: tuple[int, int],
    ) -> tuple[int, int]:
        """
        Calculate the number of pixels along the width and height to resize the video
        to based on the desired aspect ratio.

        Parameters
        ----------
        original_pixels_width: int
            Number of pixels along the width of the original video.
        original_pixels_height: int
            Number of pixels along the height of the original video
        resize_aspect_ratio: tuple[int, int]
            The width:height aspect ratio to resize the video to

        Returns
        -------
        tuple[int, int]
            The number of pixels along the width and height to resize the video to
        """
        resize_ar_width, resize_ar_height = resize_aspect_ratio
        desired_aspect_ratio = resize_ar_width / resize_ar_height
        original_aspect_ratio = original_width_pixels / original_height_pixels

        # original aspect ratio is wider than desired aspect ratio
        if original_aspect_ratio > desired_aspect_ratio:
            resize_height_pixels = original_height_pixels
            resize_width_pixels = int(
                resize_height_pixels * resize_ar_width / resize_ar_height
            )
        # original aspect ratio is taller than desired aspect ratio
        else:
            resize_width_pixels = original_width_pixels
            resize_height_pixels = int(
                resize_width_pixels * resize_ar_height / resize_ar_width
            )

        return resize_width_pixels, resize_height_pixels

    def _merge_scene_change_and_speaker_segments(
        self,
        speaker_segments: list[dict],
        scene_changes: list[float],
        scene_merge_threshold: float,
    ) -> list[dict]:
        """
        Merge scene change segments with speaker segments based on a specified
        threshold.

        Parameters
        ----------
        speaker_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
        scene_changes: list[float]
            List of scene change times in seconds.
        scene_merge_threshold: float
            The threshold in seconds for merging scene changes with speaker segments.
            Scene changes within this threshold of a segment's start or end time will
            cause the segment to be adjusted.

        Returns
        -------
        updated_speaker_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
        """
        segments_idx = 0
        for scene_change_sec in scene_changes:
            segment = speaker_segments[segments_idx]
            while scene_change_sec > (segment["end_time"]):
                segments_idx += 1
                segment = speaker_segments[segments_idx]
            # scene change is close to speaker segment end -> merge the two
            if 0 < (segment["end_time"] - scene_change_sec) < scene_merge_threshold:
                segment["end_time"] = scene_change_sec
                if segments_idx == len(speaker_segments) - 1:
                    continue
                next_segment = speaker_segments[segments_idx + 1]
                next_segment["start_time"] = scene_change_sec
                continue
            # scene change is close to speaker segment start -> merge the two
            if 0 < (scene_change_sec - segment["start_time"]) < scene_merge_threshold:
                segment["start_time"] = scene_change_sec
                if segments_idx == 0:
                    continue
                prev_segment = speaker_segments[segments_idx - 1]
                prev_segment["end_time"] = scene_change_sec
                continue
            # scene change already exists
            if scene_change_sec == segment["end_time"]:
                continue
            # add scene change to segments
            new_segment = {
                "start_time": scene_change_sec,
                "speakers": segment["speakers"],
                "end_time": segment["end_time"],
            }
            segment["end_time"] = scene_change_sec
            speaker_segments = (
                speaker_segments[: segments_idx + 1]
                + [new_segment]
                + speaker_segments[segments_idx + 1 :]
            )

        return speaker_segments

    def _find_first_sec_with_face_for_each_segment(
        self,
        segments: list[dict],
        video_file: VideoFile,
        face_detect_width: int,
        n_face_detect_batches: int,
    ) -> list[dict]:
        """
        Find the first frame in a segment with a face.

        Parameters
        ----------
        segments: list[dict]
            List of speaker segments (dictionaries), each with the following keys
                speakers: list[int]
                    list of speaker numbers for the speakers talking in the segment
                start_time: float
                    start time of the segment in seconds
                end_time: float
                    end time of the segment in seconds
        video_file: VideoFile
            The video file to analyze.
        n_face_detect_batches: int
            The number of batches to use for identifyinng faces from a video file

        Returns
        -------
        list[dict]
            List of speaker segments (dictionaries), each with the following keys
                speakers: list[int]
                    list of speaker numbers for the speakers talking in the segment
                start_time: float
                    start time of the segment in seconds
                end_time: float
                    end time of the segment in seconds
                first_face_sec: float
                    the first second in the segment with a face
                found_face: bool
                    whether or not a face was found in the segment
        """
        for segment in segments:
            start_time = segment["start_time"]
            end_time = segment["end_time"]
            # start looking for faces an eighth of the way through the segment
            segment["first_face_sec"] = start_time + (end_time - start_time) / 8
            segment["found_face"] = False
            segment["is_analyzed"] = False

        batch_period = 1  # interval length to sample each segment at each iteration
        sample_period = 1  # interval between consecutive samples
        analyzed_segments = 0
        while analyzed_segments < len(segments):
            # select times to detect faces from
            detect_secs = []
            for segment in segments:
                if segment["is_analyzed"] is True:
                    continue
                segment_secs_left = segment["end_time"] - segment["first_face_sec"]
                num_samples = min(batch_period, segment_secs_left) // sample_period
                num_samples = max(1, int(num_samples))
                segment["num_samples"] = num_samples
                for i in range(num_samples):
                    detect_secs.append(segment["first_face_sec"] + i * sample_period)

            # detect faces
            n_batches = self._calc_n_batches(
                video_file=video_file,
                num_frames=len(detect_secs),
                face_detect_width=face_detect_width,
                n_face_detect_batches=n_face_detect_batches,
            )
            frames_per_batch = int(len(detect_secs) // n_batches + 1)
            face_detections = []
            for i in range(n_batches):
                frames = extract_frames(
                    video_file,
                    detect_secs[
                        i
                        * frames_per_batch : min(
                            (i + 1) * frames_per_batch, len(detect_secs)
                        )
                    ],
                )
                face_detections += self._detect_faces(frames, face_detect_width)

            # check if any faces were found for each segment
            idx = 0
            for segment in segments:
                # segment already analyzed
                if segment["is_analyzed"] is True:
                    continue
                segment_idx = idx
                # check if any faces were found
                for _ in range(segment["num_samples"]):
                    faces = face_detections[idx]
                    if faces is not None:
                        segment["found_face"] = True
                        break
                    segment["first_face_sec"] += sample_period
                    idx += 1
                # update segment analyzation status
                is_analyzed = (
                    segment["found_face"] is True
                    or segment["first_face_sec"] >= segment["end_time"] - 0.25
                )
                if is_analyzed:
                    segment["is_analyzed"] = True
                    analyzed_segments += 1
                idx = segment_idx + segment["num_samples"]

            # increase period for next iteration
            batch_period = (batch_period + 3) * 2

        for segment in segments:
            del segment["num_samples"]
            del segment["is_analyzed"]

        return segments

    def _calc_n_batches(
        self,
        video_file: VideoFile,
        num_frames: int,
        face_detect_width: int,
        n_face_detect_batches: int,
    ) -> int:
        """
        Calculate the number of batches to use for extracting frames from a video file
        and detecting the face in each frame.

        Parameters
        ----------
        video_file: VideoFile
            The video file to analyze.
        num_frames: int
            The number of frames to analyze.
        face_detect_width: int
            The width to use for face detection.
        n_face_detect_batches: int
            Number of batches for GPU face detection in a video file.

        Returns
        -------
        int
            The number of batches to use.
        """
        # calculate memory needed to extract frames to CPU
        vid_height = video_file.get_height_pixels()
        vid_width = video_file.get_width_pixels()
        num_color_channels = 3
        bytes_per_frame = calc_img_bytes(vid_height, vid_width, num_color_channels)
        total_extract_bytes = num_frames * bytes_per_frame
        logging.debug(
            "Need {:.3f} GiB to extract (at most) {} frames".format(
                bytes_to_gibibytes(total_extract_bytes), num_frames
            )
        )

        # calculate memory needed to detect faces -> could be CPU or GPU
        downsample_factor = max(vid_width / face_detect_width, 1)
        face_detect_height = int(vid_height // downsample_factor)
        logging.debug(
            "Face detection dimensions: {}x{}".format(
                face_detect_height, face_detect_width
            )
        )
        bytes_per_frame = calc_img_bytes(
            face_detect_height, face_detect_width, num_color_channels
        )
        total_face_detect_bytes = num_frames * bytes_per_frame
        logging.debug(
            "Need {:.3f} GiB to detect faces from (at most) {} frames".format(
                bytes_to_gibibytes(total_face_detect_bytes), num_frames
            )
        )

        # calculate number of batches to use
        free_cpu_memory = pytorch.get_free_cpu_memory()
        if self._face_detect_backend != DEFAULT_FACE_DETECT_BACKEND:
            n_face_detect_batches = 0
        if torch.cuda.is_available():
            n_extract_batches = int((total_extract_bytes // free_cpu_memory) + 1)
        else:
            total_extract_bytes += total_face_detect_bytes
            n_extract_batches = int((total_extract_bytes // free_cpu_memory) + 1)
            n_face_detect_batches = 0

        n_batches = int(max(n_extract_batches, n_face_detect_batches))
        cpu_mem_per_batch = bytes_to_gibibytes(total_extract_bytes // n_batches)
        if n_face_detect_batches == 0:
            gpu_mem_per_batch = 0
        else:
            gpu_mem_per_batch = bytes_to_gibibytes(total_face_detect_bytes // n_batches)
        logging.debug(
            "Using {} batches to extract and detect frames. Need {:.3f} GiB of CPU "
            "memory per batch and {:.3f} GiB of GPU memory per batch".format(
                n_batches,
                cpu_mem_per_batch,
                gpu_mem_per_batch,
            )
        )
        return n_batches

    def _detect_faces(
        self,
        frames: list[np.ndarray],
        face_detect_width: int,
    ) -> list[np.ndarray]:
        """
        Detect faces in a list of frames.

        Parameters
        ----------
        frames: list[np.ndarray]
            The frames to detect faces in.
        face_detect_width: int
            The width to use for face detection.

        Returns
        -------
        list[np.ndarray]
            The face detections for each frame.
        """
        return self._face_detector.detect(frames, face_detect_width)

    def _add_x_y_coords_to_each_segment(
        self,
        segments: list[dict],
        video_file: VideoFile,
        resize_width: int,
        resize_height: int,
        samples_per_segment: int,
        face_detect_width: int,
        n_face_detect_batches: int,
    ) -> list[dict]:
        """
        Add the x and y coordinates to resize each segment to.

        Parameters
        ----------
        segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
            first_face_sec: float
                the first second in the segment with a face
            found_face: bool
                whether or not a face was found in the segment
        video_file: VideoFile
            The video file to analyze.
        resize_width: int
            The width to resize the video to.
        resize_height: int
            The height to resize the video to.
        samples_per_segment: int
            Number of samples to take per segment for face detection.
        face_detect_width: int
            Width to resize the frames to for face detection.
        n_face_detect_batches: int
            Number of batches to process for face detection.


        Returns
        -------
        updated_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
            x: int
                x-coordinate of the top left corner of the resized segment
            y: int
                y-coordinate of the top left corner of the resized segment
        """
        num_segments = len(segments)
        num_frames = num_segments * samples_per_segment
        n_batches = self._calc_n_batches(
            video_file, num_frames, face_detect_width, n_face_detect_batches
        )
        segments_per_batch = int(num_segments // n_batches + 1)
        segments_with_xy_coords = []
        speaker_face_centers: dict[int, float] = {}
        for i in range(n_batches):
            logging.debug("Analyzing batch {} of {}.".format(i, n_batches))
            cur_segments = segments[
                i
                * segments_per_batch : min((i + 1) * segments_per_batch, len(segments))
            ]
            if len(cur_segments) == 0:
                logging.debug("No segments left to analyze. (Batch {})".format(i))
                break
            segments_with_xy_coords += self._add_x_y_coords_to_each_segment_batch(
                segments=cur_segments,
                video_file=video_file,
                resize_width=resize_width,
                resize_height=resize_height,
                samples_per_segment=samples_per_segment,
                face_detect_width=face_detect_width,
                speaker_face_centers=speaker_face_centers,
            )
        segments_with_xy_coords = self._reconcile_speaker_face_choices(
            segments=segments_with_xy_coords,
            resize_width=resize_width,
            resize_height=resize_height,
            frame_width=video_file.get_width_pixels(),
        )
        for segment in segments_with_xy_coords:
            segment.pop("_roi_candidates", None)
        return segments_with_xy_coords

    def _add_x_y_coords_to_each_segment_batch(
        self,
        segments: list[dict],
        video_file: VideoFile,
        resize_width: int,
        resize_height: int,
        samples_per_segment: int,
        face_detect_width: int,
        speaker_face_centers: dict[int, float],
    ) -> list[dict]:
        """
        Add the x and y coordinates to resize each segment to for a given batch.

        Parameters
        ----------
        segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
            first_face_sec: float
                the first second in the segment with a face
            found_face: bool
                whether or not a face was found in the segment
        video_file: VideoFile
            The video file to analyze.
        resize_width: int
            The width to resize the video to.
        resize_height: int
            The height to resize the video to.
        samples_per_segment: int
            Number of samples to take per segment for analyzing face locations.
        face_detect_width: int
            Width to which the video frames are resized for face detection.

        Returns
        -------
        updated_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
            x: int
                x-coordinate of the top left corner of the resized segment
            y: int
                y-coordinate of the top left corner of the resized segment
        """
        fps = video_file.get_frame_rate()

        # define frames to analyze from each segment
        detect_secs = []
        for segment in segments:
            if segment["found_face"] is False:
                continue
            # define interval over which to analyze faces
            end_time = segment["end_time"]
            first_face_sec = segment["first_face_sec"]
            analyze_end_time = end_time - (end_time - first_face_sec) / 8
            # get sample locations
            frames_left = int((analyze_end_time - first_face_sec) * fps + 1)
            num_samples = min(frames_left, samples_per_segment)
            segment["num_samples"] = num_samples
            # add first face, sample the rest
            detect_secs.append(first_face_sec)
            sample_frames = np.sort(
                np.random.choice(range(1, frames_left), num_samples - 1, replace=False)
            )
            for sample_frame in sample_frames:
                detect_secs.append(first_face_sec + sample_frame / fps)

        # detect faces from each segment
        logging.debug("Extracting {} frames".format(len(detect_secs)))
        frames = extract_frames(video_file, detect_secs)
        logging.debug("Extracted {} frames".format(len(detect_secs)))
        face_detections = self._detect_faces(frames, face_detect_width)

        logging.debug("Calculating ROI for {} segments.".format(len(segments)))
        # find roi for each segment
        idx = 0
        for segment in segments:
            # find segment roi
            if segment["found_face"] is True:
                roi_result = self._calc_segment_roi(
                    frames=frames[idx : idx + segment["num_samples"]],
                    face_detections=face_detections[idx : idx + segment["num_samples"]],
                    speakers=segment.get("speakers", []),
                    speaker_face_centers=speaker_face_centers,
                )
                roi = roi_result["roi"]
                segment["crop_selection"] = roi_result["crop_selection"]
                segment["_roi_candidates"] = roi_result.get("roi_candidates", [])
                idx += segment["num_samples"]
                del segment["num_samples"]
            else:
                logging.debug("Using default ROI for segment {}".format(segment))
                roi = Rect(
                    x=(video_file.get_width_pixels()) // 4,
                    y=(video_file.get_height_pixels()) // 4,
                    width=(video_file.get_width_pixels()) // 2,
                    height=(video_file.get_height_pixels()) // 2,
                )
                segment["crop_selection"] = {
                    "reason": "fallback_default_center",
                    "face_side": "center",
                    "face_center_x": self._roi_center_x(roi),
                    "mouth_movement": 0.0,
                    "landmark_count": 0,
                    "face_sample_count": 0,
                    "speaker_mapping_locked": False,
                }
                segment["_roi_candidates"] = []
            del segment["found_face"]
            del segment["first_face_sec"]

            # add crop coordinates to segment
            crop = self._calc_crop(roi, resize_width, resize_height)
            segment["x"] = int(crop.x)
            segment["y"] = int(crop.y)
        logging.debug("Calculated ROI for {} segments.".format(len(segments)))

        return segments

    def _calc_segment_roi(
        self,
        frames: list[np.ndarray],
        face_detections: list[np.ndarray],
        speakers: list[int] | None = None,
        speaker_face_centers: dict[int, float] | None = None,
    ) -> dict:
        """
        Find the region of interest (ROI) for a given segment.

        Parameters
        ----------
        frames: np.ndarray
            The frames to analyze.
        face_detections: np.ndarray
            The face detection outputs for each frame

        Returns
        -------
        dict
            The selected ROI plus crop-selection evidence for the segment.
        """
        speakers = speakers or []
        speaker_face_centers = (
            speaker_face_centers if speaker_face_centers is not None else {}
        )

        # preprocessing for kmeans
        bounding_boxes: list[np.ndarray] = []
        k = 0
        for face_detection in face_detections:
            if face_detection is None:
                continue
            k = max(k, len(face_detection))
            for bounding_box in face_detection:
                bounding_boxes.append(bounding_box)

        # no faces detected
        if k == 0:
            raise ResizerError("No faces detected in segment.")
        bounding_boxes = np.stack(bounding_boxes)
        frame_width = frames[0].shape[1]

        # single face detected
        if k == 1:
            box = np.mean(bounding_boxes, axis=0).astype(np.int16)
            x1, y1, x2, y2 = box
            segment_roi = Rect(x1, y1, x2 - x1, y2 - y1)
            self._remember_single_speaker_face(
                speakers=speakers,
                speaker_face_centers=speaker_face_centers,
                roi=segment_roi,
            )
            candidate = {
                "mouth_movement": 0.0,
                "landmark_count": 0,
                "frame_count": len(bounding_boxes),
                "roi": segment_roi,
                "selection_reason": "single_face",
                "speaker_mapping_locked": len(speakers) == 1,
            }
            return {
                "roi": segment_roi,
                "crop_selection": self._build_crop_selection_metadata(
                    candidate,
                    frame_width,
                ),
                "roi_candidates": [candidate],
            }

        # use kmeans to group the same bounding boxes together
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=2, random_state=0).fit(
            bounding_boxes
        )
        bounding_box_labels = kmeans.labels_
        bounding_box_groups: list[list[dict]] = [[] for _ in range(k)]
        kmeans_idx = 0
        for i, face_detection in enumerate(face_detections):
            if face_detection is None:
                continue
            for bounding_box in face_detection:
                assert np.sum(bounding_box < 0) == 0
                bounding_box_label = bounding_box_labels[kmeans_idx]
                bounding_box_groups[bounding_box_label].append(
                    {"bounding_box": bounding_box, "frame": i}
                )
                kmeans_idx += 1

        roi_candidates = []
        for bounding_box_group in bounding_box_groups:
            if not bounding_box_group:
                continue
            roi_candidates.append(self._calc_mouth_movement(bounding_box_group, frames))
        if not roi_candidates:
            raise ResizerError("No usable face groups detected in segment.")

        selected_candidate = self._select_segment_roi_candidate(
            roi_candidates=roi_candidates,
            speakers=speakers,
            speaker_face_centers=speaker_face_centers,
        )

        return {
            "roi": selected_candidate["roi"],
            "crop_selection": self._build_crop_selection_metadata(
                selected_candidate,
                frame_width,
            ),
            "roi_candidates": roi_candidates,
        }

    def _select_segment_roi_candidate(
        self,
        roi_candidates: list[dict],
        speakers: list[int],
        speaker_face_centers: dict[int, float],
    ) -> dict:
        """
        Choose a face ROI using mouth movement first and speaker continuity second.
        """
        if self._has_confident_mouth_movement_candidate(roi_candidates):
            selected_candidate = max(
                roi_candidates,
                key=lambda candidate: candidate["mouth_movement"],
            )
            selected_candidate["selection_reason"] = "mouth_movement"
            selected_candidate["speaker_mapping_locked"] = len(speakers) == 1
        else:
            logging.debug("No mouth movement detected for segment.")
            selected_candidate = self._select_no_mouth_movement_roi_candidate(
                roi_candidates=roi_candidates,
                speakers=speakers,
                speaker_face_centers=speaker_face_centers,
            )

        selected_roi = selected_candidate["roi"]
        if selected_candidate.get("speaker_mapping_locked") is True:
            self._remember_single_speaker_face(
                speakers=speakers,
                speaker_face_centers=speaker_face_centers,
                roi=selected_roi,
            )
        return selected_candidate

    def _has_confident_mouth_movement_candidate(
        self,
        roi_candidates: list[dict],
    ) -> bool:
        """
        Return whether the strongest mouth signal is reliable enough to use.
        """
        if not roi_candidates:
            return False

        ordered_candidates = sorted(
            roi_candidates,
            key=lambda candidate: candidate["mouth_movement"],
            reverse=True,
        )
        strongest = ordered_candidates[0]
        if strongest["landmark_count"] < MOUTH_MOVEMENT_MIN_LANDMARK_FRAMES:
            return False
        if strongest["mouth_movement"] < MOUTH_MOVEMENT_MIN_SCORE:
            return False
        if len(ordered_candidates) == 1:
            return True

        runner_up = ordered_candidates[1]
        strongest_score = strongest["mouth_movement"]
        runner_up_score = runner_up["mouth_movement"]
        if runner_up_score <= 0:
            return True
        if strongest_score - runner_up_score >= MOUTH_MOVEMENT_MIN_DELTA:
            return True
        return strongest_score >= runner_up_score * MOUTH_MOVEMENT_MIN_RATIO

    def _select_no_mouth_movement_roi_candidate(
        self,
        roi_candidates: list[dict],
        speakers: list[int],
        speaker_face_centers: dict[int, float],
    ) -> dict:
        """
        Pick a stable face when landmarks cannot identify the active speaker.
        """
        if len(speakers) == 1:
            speaker = int(speakers[0])
            if speaker in speaker_face_centers:
                selected_candidate = min(
                    roi_candidates,
                    key=lambda candidate: abs(
                        self._roi_center_x(candidate["roi"])
                        - speaker_face_centers[speaker]
                    ),
                )
                selected_candidate["selection_reason"] = "speaker_continuity"
                selected_candidate["speaker_mapping_locked"] = False
                return selected_candidate

            if speaker_face_centers:
                known_centers = list(speaker_face_centers.values())
                selected_candidate = max(
                    roi_candidates,
                    key=lambda candidate: (
                        min(
                            abs(self._roi_center_x(candidate["roi"]) - known_center)
                            for known_center in known_centers
                        ),
                        candidate["frame_count"],
                    ),
                )
                selected_candidate["selection_reason"] = "fallback_unclaimed_face"
                selected_candidate["speaker_mapping_locked"] = False
                return selected_candidate

        selected_candidate = max(
            roi_candidates,
            key=lambda candidate: candidate["frame_count"],
        )
        selected_candidate["selection_reason"] = "fallback_most_frames"
        selected_candidate["speaker_mapping_locked"] = False
        return selected_candidate

    def _remember_single_speaker_face(
        self,
        speakers: list[int],
        speaker_face_centers: dict[int, float],
        roi: Rect,
    ) -> None:
        """
        Store the selected face position for single-speaker segments.
        """
        if len(speakers) != 1:
            return
        speaker_face_centers[int(speakers[0])] = self._roi_center_x(roi)

    def _roi_center_x(self, roi: Rect) -> float:
        """
        Return the horizontal center of one ROI.
        """
        return float(roi.x + roi.width / 2)

    def _face_side(self, center_x: float, frame_width: int) -> str:
        """
        Return a readable horizontal face location label.
        """
        if center_x < frame_width * 0.45:
            return "left"
        if center_x > frame_width * 0.55:
            return "right"
        return "center"

    def _build_crop_selection_metadata(
        self,
        candidate: dict,
        frame_width: int,
    ) -> dict:
        """
        Build JSON-safe evidence for the selected face crop.
        """
        center_x = self._roi_center_x(candidate["roi"])
        return {
            "reason": candidate.get("selection_reason", "unknown"),
            "face_side": self._face_side(center_x, frame_width),
            "face_center_x": round(center_x, 3),
            "mouth_movement": round(float(candidate.get("mouth_movement", 0.0)), 6),
            "landmark_count": int(candidate.get("landmark_count", 0)),
            "face_sample_count": int(candidate.get("frame_count", 0)),
            "speaker_mapping_locked": bool(
                candidate.get("speaker_mapping_locked", False)
            ),
        }

    def _reconcile_speaker_face_choices(
        self,
        segments: list[dict],
        resize_width: int,
        resize_height: int,
        frame_width: int,
    ) -> list[dict]:
        """
        Let strong mouth evidence correct earlier fallback-only speaker choices.
        """
        speaker_face_centers: dict[int, float] = {}
        speaker_face_scores: dict[int, float] = {}

        for segment in segments:
            speakers = segment.get("speakers", [])
            crop_selection = segment.get("crop_selection", {})
            if len(speakers) != 1:
                continue
            if crop_selection.get("reason") not in ("mouth_movement", "single_face"):
                continue
            score = float(crop_selection.get("mouth_movement", 0.0)) + (
                int(crop_selection.get("landmark_count", 0)) * 0.001
            )
            speaker = int(speakers[0])
            if score >= speaker_face_scores.get(speaker, -1.0):
                speaker_face_scores[speaker] = score
                speaker_face_centers[speaker] = float(crop_selection["face_center_x"])

        for segment in segments:
            speakers = segment.get("speakers", [])
            if len(speakers) != 1:
                continue
            speaker = int(speakers[0])
            target_center = speaker_face_centers.get(speaker)
            if target_center is None:
                continue

            crop_selection = segment.get("crop_selection", {})
            if crop_selection.get("reason") in ("mouth_movement", "single_face"):
                continue

            candidates = segment.get("_roi_candidates", [])
            if not candidates:
                continue

            selected_candidate = min(
                candidates,
                key=lambda candidate: abs(
                    self._roi_center_x(candidate["roi"]) - target_center
                ),
            )
            previous_reason = crop_selection.get("reason", "unknown")
            previous_x = segment.get("x")
            previous_y = segment.get("y")
            selected_candidate["selection_reason"] = "reconciled_speaker_mapping"
            selected_candidate["speaker_mapping_locked"] = True
            roi = selected_candidate["roi"]
            crop = self._calc_crop(roi, resize_width, resize_height)
            segment["x"] = int(crop.x)
            segment["y"] = int(crop.y)
            segment["crop_selection"] = self._build_crop_selection_metadata(
                selected_candidate,
                frame_width,
            )
            segment["crop_selection"]["previous_reason"] = previous_reason
            segment["crop_selection"]["previous_x"] = previous_x
            segment["crop_selection"]["previous_y"] = previous_y
            segment["crop_selection"]["matched_speaker"] = speaker

        return segments

    def _prepare_face_for_mouth_analysis(
        self,
        frame: np.ndarray,
        bounding_box: np.ndarray,
        margin_ratio: float = MOUTH_ANALYSIS_CROP_MARGIN_RATIO,
        min_face_size_pixels: int = MOUTH_ANALYSIS_MIN_FACE_SIZE_PIXELS,
    ) -> np.ndarray:
        """
        Expand and upscale a face crop before running mouth landmarks.
        """
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in bounding_box[:4]]
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(x1 + 1, min(x2, frame_width))
        y2 = max(y1 + 1, min(y2, frame_height))

        box_width = x2 - x1
        box_height = y2 - y1
        margin_x = int(box_width * margin_ratio)
        margin_y = int(box_height * margin_ratio)
        crop_x1 = max(0, x1 - margin_x)
        crop_y1 = max(0, y1 - margin_y)
        crop_x2 = min(frame_width, x2 + margin_x)
        crop_y2 = min(frame_height, y2 + margin_y)

        face = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]
        if face.size == 0:
            return face

        face_height, face_width = face.shape[:2]
        largest_side = max(face_width, face_height)
        if largest_side < min_face_size_pixels:
            scale = min_face_size_pixels / largest_side
            face = cv2.resize(
                face,
                (
                    max(1, int(face_width * scale)),
                    max(1, int(face_height * scale)),
                ),
                interpolation=cv2.INTER_CUBIC,
            )

        return np.ascontiguousarray(face)

    def _calc_mouth_movement(
        self,
        bounding_box_group: list[dict[np.ndarray, int]],
        frames: list[np.ndarray],
    ) -> dict:
        """
        Calculates the mouth movement for a group of faces. These faces are assumed to
        all be the same person in different frames of the source video. Further, the
        frames are assumed to be in order of occurrence (earlier frames first).

        Parameters
        ----------
        bounding_box_group: list[dict[np.ndarray, int]]
            The faces to analyze. A list of dictionaries, each with the following keys:
                bounding_box: np.ndarray
                    The bounding box of the face to analyze. The array contains four
                    values: [x1, y1, x2, y2]
                frame: int
                    The frame the bounding box of the face is associated with.
        frames: list[np.ndarray]
            The frames to analyze.

        Returns
        -------
        dict
            Mouth movement, landmark count, and averaged ROI for the face group.
        """
        roi = Rect(0, 0, 0, 0)
        prev_mar = None
        mouth_movement = 0.0
        landmark_count = 0

        for bounding_box_data in bounding_box_group:
            # roi
            box = bounding_box_data["bounding_box"]
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            # sum all roi's, average after loop
            roi += Rect(x1, y1, x2 - x1, y2 - y1)
            frame = frames[bounding_box_data["frame"]]
            face = self._prepare_face_for_mouth_analysis(frame, box)

            # mouth movement
            mar = self._calc_mouth_aspect_ratio(face)
            if mar is None:
                continue
            landmark_count += 1
            if prev_mar is None:
                prev_mar = mar
                continue
            mouth_movement += abs(mar - prev_mar)
            prev_mar = mar

        return {
            "mouth_movement": float(mouth_movement),
            "landmark_count": landmark_count,
            "frame_count": len(bounding_box_group),
            "roi": roi / len(bounding_box_group),
        }

    def _calc_mouth_aspect_ratio(self, face: np.ndarray) -> float | None:
        """
        Calculate the mouth aspect ratio using dlib shape predictor.

        Parameters
        ----------
        face: np.ndarray
            Pytorch array of a face

        Returns
        -------
        mar: float
            The mouth aspect ratio.
        """
        if face.size == 0:
            return None

        landmarks = self._face_landmarker.detect(face)
        if landmarks is None:
            return None

        # inner lip
        upper_lip = landmarks[[95, 88, 178, 87, 14, 317, 402, 318, 324], :]
        lower_lip = landmarks[[191, 80, 81, 82, 13, 312, 311, 310, 415], :]
        avg_mouth_height = np.mean(np.abs(upper_lip - lower_lip))
        mouth_width = np.sum(np.abs(landmarks[[308], :] - landmarks[[78], :]))
        mar = avg_mouth_height / mouth_width

        return mar

    def _calc_crop(
        self,
        roi: Rect,
        resize_width: int,
        resize_height: int,
    ) -> Rect:
        """
        Calculate the crop given the ROI location.

        Parameters
        ----------
        roi: Rect
            The rectangle containing the region of interest (ROI).

        Returns
        -------
        Rect
            The crop rectangle.
        """
        roi_x_center = roi.x + roi.width // 2
        roi_y_center = roi.y + roi.height // 2
        crop = Rect(
            x=max(roi_x_center - (resize_width // 2), 0),
            y=max(roi_y_center - (resize_height // 2), 0),
            width=resize_width,
            height=resize_height,
        )
        return crop

    def _merge_identical_segments(
        self,
        segments: list[dict],
        video_file: VideoFile,
    ) -> list[dict]:
        """
        Merge identical segments that are next to each other.

        Parameters
        ----------
        segments: list[dict]
            speakers: list[int]
                the speaker labels of the speakers talking in the segment
            start_time: float
                the start time of the segment
            end_time: float
                the end time of the segment
            x: int
                x-coordinate of the top left corner of the resized segment
            y: int
                y-coordinate of the top left corner of the resized segment
        video_file: VideoFile
            The video file that the segments are from

        Returns
        -------
        list[dict]
            The merged segments.
        """
        idx = 0
        max_position_difference_ratio = 0.04
        video_width = video_file.get_width_pixels()
        video_height = video_file.get_height_pixels()

        for _ in range(len(segments) - 1):
            cur_speakers = segments[idx].get("speakers")
            next_speakers = segments[idx + 1].get("speakers")
            if cur_speakers is not None and next_speakers is not None:
                if cur_speakers != next_speakers:
                    idx += 1
                    continue

            cur_x = segments[idx]["x"]
            next_x = segments[idx + 1]["x"]
            x_diff = abs(cur_x - next_x)
            if (x_diff / video_width) < max_position_difference_ratio:
                same_x = True
                segments[idx]["x"] = int((cur_x + next_x) // 2)
            else:
                same_x = False

            curr_y = segments[idx]["y"]
            next_y = segments[idx + 1]["y"]
            y_diff = abs(curr_y - next_y)
            if (y_diff / video_height) < max_position_difference_ratio:
                same_y = True
                segments[idx]["y"] = int((curr_y + next_y) // 2)
            else:
                same_y = False

            if same_x and same_y:
                segments[idx]["end_time"] = segments[idx + 1]["end_time"]
                segments = segments[: idx + 1] + segments[idx + 2 :]
            else:
                idx += 1
        return segments

    def cleanup(self) -> None:
        """
        Remove the face detector from memory and explicity free up GPU memory.
        """
        self._face_detector.cleanup()
        self._face_detector = None
        self._face_landmarker.close()
        self._face_landmarker = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
