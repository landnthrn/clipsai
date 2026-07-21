# standard library imports
from unittest.mock import patch, MagicMock

# local package imports
from clipsai.media.video_file import VideoFile
from clipsai.resize.resizer import Resizer
from clipsai.resize.rect import Rect


# third party imports
import numpy as np
import pytest


def build_test_resizer(**kwargs):
    with patch(
        "clipsai.resize.resizer.pytorch.get_compute_device",
        return_value="cpu",
    ), patch(
        "clipsai.resize.resizer.pytorch.assert_compute_device_available",
        return_value=None,
    ), patch(
        "clipsai.resize.resizer.build_face_detector",
        return_value=MagicMock(),
    ), patch(
        "clipsai.resize.resizer.build_face_landmarker",
        return_value=MagicMock(),
    ):
        return Resizer(**kwargs)


@pytest.mark.parametrize(
    "original_width, original_height, aspect_ratio, expected",
    [
        # Wider aspect ratio
        (1920, 1080, (9, 16), (607, 1080)),
        (1280, 720, (9, 16), (405, 720)),
        # Taller aspect ratio
        (1080, 1920, (16, 9), (1080, 607)),
        (720, 1280, (16, 9), (720, 405)),
        # Extreme aspect ratios
        (1920, 1080, (1, 100), (10, 1080)),
        (1920, 1080, (100, 1), (1920, 19)),
        # Equal aspect ratio
        (1920, 1080, (16, 9), (1920, 1080)),
        (1280, 720, (16, 9), (1280, 720)),
        # Equal aspect ratio = Small dimensions
        (320, 240, (4, 3), (320, 240)),
        (10, 10, (1, 1), (10, 10)),
        # Equal aspect ratio = Large dimensions
        (8000, 4500, (16, 9), (8000, 4500)),
        (4500, 8000, (9, 16), (4500, 8000)),
    ],
)
def test_calc_resize_width_and_height_pixels(
    original_width: int,
    original_height: int,
    aspect_ratio: tuple[int, int],
    expected: tuple[int, int],
):
    resizer = build_test_resizer()
    result = resizer._calc_resize_width_and_height_pixels(
        original_width_pixels=original_width,
        original_height_pixels=original_height,
        resize_aspect_ratio=aspect_ratio,
    )
    assert result == expected


# Test cases
@pytest.mark.parametrize(
    "speaker_segments, scene_changes, expected",
    [
        # Test with no scene changes
        (
            [{"speakers": [0], "start_time": 0, "end_time": 10}],
            [],
            [{"speakers": [0], "start_time": 0, "end_time": 10}],
        ),
        # Test with scene change matching the end of a segment
        (
            [{"speakers": [0], "start_time": 0, "end_time": 5}],
            [5],
            [{"speakers": [0], "start_time": 0, "end_time": 5}],
        ),
        # Test with scene change within a segment
        (
            [{"speakers": [0], "start_time": 0, "end_time": 10}],
            [5],
            [
                {"speakers": [0], "start_time": 0, "end_time": 5},
                {"speakers": [0], "start_time": 5, "end_time": 10},
            ],
        ),
        # Test with multiple segments and scene changes
        (
            [
                {"speakers": [0], "start_time": 0, "end_time": 5},
                {"speakers": [1], "start_time": 5, "end_time": 10},
            ],
            [3, 8],
            [
                {"speakers": [0], "start_time": 0, "end_time": 3},
                {"speakers": [0], "start_time": 3, "end_time": 5},
                {"speakers": [1], "start_time": 5, "end_time": 8},
                {"speakers": [1], "start_time": 8, "end_time": 10},
            ],
        ),
        # Test with scene changes at segment boundaries
        (
            [
                {"speakers": [0], "start_time": 0, "end_time": 5},
                {"speakers": [1], "start_time": 5, "end_time": 10},
            ],
            [5],
            [
                {"speakers": [0], "start_time": 0, "end_time": 5},
                {"speakers": [1], "start_time": 5, "end_time": 10},
            ],
        ),
        # Test with scene change very close to segment start
        (
            [
                {"speakers": [0], "start_time": 0, "end_time": 5},
                {"speakers": [1], "start_time": 5, "end_time": 10},
            ],
            [4.8],
            [
                {"speakers": [0], "start_time": 0, "end_time": 4.8},
                {"speakers": [1], "start_time": 4.8, "end_time": 10},
            ],
        ),
        # Test with scene change very close to segment end
        (
            [
                {"speakers": [0], "start_time": 0, "end_time": 5},
                {"speakers": [1], "start_time": 5, "end_time": 10},
            ],
            [5.1],
            [
                {"speakers": [0], "start_time": 0, "end_time": 5.1},
                {"speakers": [1], "start_time": 5.1, "end_time": 10},
            ],
        ),
    ],
)
def test_merge_scene_change_and_speaker_segments(
    speaker_segments: list[dict], scene_changes: list[float], expected: list[dict]
):
    resizer = build_test_resizer()
    result = resizer._merge_scene_change_and_speaker_segments(
        speaker_segments=speaker_segments,
        scene_changes=scene_changes,
        scene_merge_threshold=0.25,
    )
    assert result == expected


@pytest.mark.parametrize(
    (
        "width, height, num_frames, gpu_available, face_detect_width,"
        "n_face_detect_batches, expected_batches"
    ),
    [
        # Scenario 1: CPU only, small video
        (640, 480, 100, False, 960, 8, 1),
        # Scenario 2: CPU only, large video
        (1920, 1080, 100, False, 960, 8, 1),
        # Scenario 3: GPU available, small video
        (640, 480, 100, True, 960, 8, 8),
        # Scenario 4: GPU available, large video
        (1920, 1080, 100, True, 960, 8, 8),
    ],
)
def test_calc_n_batches(
    width: int,
    height: int,
    num_frames: int,
    gpu_available: bool,
    face_detect_width: int,
    n_face_detect_batches: int,
    expected_batches: int,
):
    # Setup the mock video file object
    mock_video_file = MagicMock(spec=VideoFile)
    mock_video_file.get_width_pixels.return_value = width
    mock_video_file.get_height_pixels.return_value = height

    resizer = build_test_resizer()

    # Mock pytorch.get_free_cpu_memory ~7.5 GiB
    with patch("torch.cuda.is_available", return_value=gpu_available), patch(
        "clipsai.resize.resizer.pytorch.get_free_cpu_memory",
        return_value=8000000000,
    ):
        n_batches = resizer._calc_n_batches(
            video_file=mock_video_file,
            num_frames=num_frames,
            face_detect_width=face_detect_width,
            n_face_detect_batches=n_face_detect_batches,
        )

        assert n_batches == expected_batches


def test_resizer_builds_requested_face_detection_backend():
    with patch(
        "clipsai.resize.resizer.pytorch.get_compute_device",
        return_value="cpu",
    ), patch(
        "clipsai.resize.resizer.pytorch.assert_compute_device_available",
        return_value=None,
    ), patch(
        "clipsai.resize.resizer.build_face_detector",
        return_value=MagicMock(),
    ) as detector_builder, patch(
        "clipsai.resize.resizer.build_face_landmarker",
        return_value=MagicMock(),
    ) as landmarker_builder:
        Resizer(
            face_detect_backend="mediapipe",
            mediapipe_face_detect_model_selection=1,
            mediapipe_face_detect_min_detection_confidence=0.65,
            diarization_model="community-1",
        )

    detector_builder.assert_called_once_with(
        backend_name="mediapipe",
        face_detect_margin=20,
        face_detect_post_process=False,
        device="cpu",
        mediapipe_face_detect_model_selection=1,
        mediapipe_face_detect_min_detection_confidence=0.65,
        diarization_model="community-1",
    )
    landmarker_builder.assert_called_once_with(diarization_model="community-1")


def test_calc_n_batches_treats_mediapipe_as_cpu_side_detection():
    mock_video_file = MagicMock(spec=VideoFile)
    mock_video_file.get_width_pixels.return_value = 1920
    mock_video_file.get_height_pixels.return_value = 1080

    with patch(
        "clipsai.resize.resizer.pytorch.get_compute_device",
        return_value="cpu",
    ), patch(
        "clipsai.resize.resizer.pytorch.assert_compute_device_available",
        return_value=None,
    ), patch(
        "clipsai.resize.resizer.build_face_detector",
        return_value=MagicMock(),
    ), patch(
        "clipsai.resize.resizer.build_face_landmarker",
        return_value=MagicMock(),
    ):
        resizer = Resizer(
            face_detect_backend="mediapipe",
            diarization_model="community-1",
        )

    with patch("torch.cuda.is_available", return_value=True), patch(
        "clipsai.resize.resizer.pytorch.get_free_cpu_memory",
        return_value=8000000000,
    ):
        n_batches = resizer._calc_n_batches(
            video_file=mock_video_file,
            num_frames=100,
            face_detect_width=960,
            n_face_detect_batches=8,
        )

    assert n_batches == 1


def test_no_mouth_movement_prefers_known_face_for_same_speaker():
    resizer = build_test_resizer()
    left_roi = Rect(300, 100, 100, 100)
    right_roi = Rect(1000, 100, 100, 100)

    selected = resizer._select_no_mouth_movement_roi_candidate(
        roi_candidates=[
            {"roi": right_roi, "frame_count": 8, "mouth_movement": 0},
            {"roi": left_roi, "frame_count": 4, "mouth_movement": 0},
        ],
        speakers=[1],
        speaker_face_centers={1: 350},
    )

    assert selected["roi"] == left_roi


def test_no_mouth_movement_prefers_unclaimed_face_for_new_speaker():
    resizer = build_test_resizer()
    left_roi = Rect(300, 100, 100, 100)
    right_roi = Rect(1000, 100, 100, 100)

    selected = resizer._select_no_mouth_movement_roi_candidate(
        roi_candidates=[
            {"roi": left_roi, "frame_count": 20, "mouth_movement": 0},
            {"roi": right_roi, "frame_count": 5, "mouth_movement": 0},
        ],
        speakers=[0],
        speaker_face_centers={1: 350},
    )

    assert selected["roi"] == right_roi


def test_prepare_face_for_mouth_analysis_adds_margin_and_upscales():
    resizer = build_test_resizer()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    face = resizer._prepare_face_for_mouth_analysis(
        frame=frame,
        bounding_box=np.array([40, 40, 50, 50]),
        margin_ratio=0.5,
        min_face_size_pixels=64,
    )

    assert face.shape[:2] == (64, 64)
    assert face.flags["C_CONTIGUOUS"] is True


def test_mouth_evidence_locks_speaker_face_mapping():
    resizer = build_test_resizer()
    left_roi = Rect(300, 100, 100, 100)
    right_roi = Rect(1000, 100, 100, 100)
    speaker_face_centers = {}

    selected = resizer._select_segment_roi_candidate(
        roi_candidates=[
            {
                "roi": left_roi,
                "frame_count": 8,
                "mouth_movement": 0.05,
                "landmark_count": 4,
            },
            {
                "roi": right_roi,
                "frame_count": 8,
                "mouth_movement": 0.01,
                "landmark_count": 4,
            },
        ],
        speakers=[1],
        speaker_face_centers=speaker_face_centers,
    )

    assert selected["roi"] == left_roi
    assert selected["selection_reason"] == "mouth_movement"
    assert selected["speaker_mapping_locked"] is True
    assert speaker_face_centers[1] == 350


def test_fallback_only_selection_does_not_lock_speaker_face_mapping():
    resizer = build_test_resizer()
    left_roi = Rect(300, 100, 100, 100)
    right_roi = Rect(1000, 100, 100, 100)
    speaker_face_centers = {}

    selected = resizer._select_segment_roi_candidate(
        roi_candidates=[
            {
                "roi": left_roi,
                "frame_count": 8,
                "mouth_movement": 0,
                "landmark_count": 0,
            },
            {
                "roi": right_roi,
                "frame_count": 4,
                "mouth_movement": 0,
                "landmark_count": 0,
            },
        ],
        speakers=[1],
        speaker_face_centers=speaker_face_centers,
    )

    assert selected["roi"] == left_roi
    assert selected["selection_reason"] == "fallback_most_frames"
    assert selected["speaker_mapping_locked"] is False
    assert speaker_face_centers == {}


def test_reconcile_speaker_face_choices_corrects_earlier_fallback_segment():
    resizer = build_test_resizer()
    left_roi = Rect(300, 100, 100, 100)
    right_roi = Rect(1000, 100, 100, 100)
    left_crop = resizer._calc_crop(left_roi, resize_width=600, resize_height=1080)
    right_crop = resizer._calc_crop(right_roi, resize_width=600, resize_height=1080)
    segments = [
        {
            "speakers": [1],
            "x": int(left_crop.x),
            "y": int(left_crop.y),
            "crop_selection": {
                "reason": "fallback_most_frames",
                "face_center_x": 350,
            },
            "_roi_candidates": [
                {
                    "roi": left_roi,
                    "frame_count": 8,
                    "mouth_movement": 0,
                    "landmark_count": 0,
                },
                {
                    "roi": right_roi,
                    "frame_count": 4,
                    "mouth_movement": 0,
                    "landmark_count": 0,
                },
            ],
        },
        {
            "speakers": [0],
            "x": int(left_crop.x),
            "y": int(left_crop.y),
            "crop_selection": {
                "reason": "mouth_movement",
                "face_center_x": 350,
                "mouth_movement": 0.12,
                "landmark_count": 5,
            },
            "_roi_candidates": [],
        },
        {
            "speakers": [1],
            "x": int(right_crop.x),
            "y": int(right_crop.y),
            "crop_selection": {
                "reason": "mouth_movement",
                "face_center_x": 1050,
                "mouth_movement": 0.07,
                "landmark_count": 5,
            },
            "_roi_candidates": [],
        },
    ]

    reconciled = resizer._reconcile_speaker_face_choices(
        segments=segments,
        resize_width=600,
        resize_height=1080,
        frame_width=1920,
    )

    assert reconciled[0]["x"] == int(right_crop.x)
    assert reconciled[0]["crop_selection"]["reason"] == "reconciled_speaker_mapping"
    assert reconciled[0]["crop_selection"]["previous_reason"] == "fallback_most_frames"
    assert reconciled[0]["crop_selection"]["matched_speaker"] == 1
    assert reconciled[1]["x"] == int(left_crop.x)
    assert reconciled[1]["crop_selection"]["reason"] == "mouth_movement"


@pytest.mark.parametrize(
    "roi, resize_width, resize_height, expected_crop",
    [
        # Test case 1
        (Rect(400, 300, 200, 200), 200, 200, Rect(400, 300, 200, 200)),
        # Test case 2
        (Rect(0, 0, 100, 100), 200, 200, Rect(0, 0, 200, 200)),
        # Test case 3
        (Rect(800, 600, 100, 100), 200, 200, Rect(750, 550, 200, 200)),
        # Test case 4
        (Rect(800, 600, 100, 100), 200, 400, Rect(750, 450, 200, 400)),
    ],
)
def test_calc_crop(roi, resize_width, resize_height, expected_crop):
    resizer = build_test_resizer()
    actual_crop = resizer._calc_crop(roi, resize_width, resize_height)
    assert actual_crop == expected_crop


@pytest.mark.parametrize(
    "segments, expected",
    [
        # Test case 1: No identical segments
        (
            [
                {"x": 100, "y": 0, "start_time": 0, "end_time": 10},
                {"x": 200, "y": 0, "start_time": 10, "end_time": 20},
            ],
            [
                {"x": 100, "y": 0, "start_time": 0, "end_time": 10},
                {"x": 200, "y": 0, "start_time": 10, "end_time": 20},
            ],
        ),
        # Test case 2: Two identical segments
        (
            [
                {"x": 100, "y": 0, "start_time": 0, "end_time": 10},
                {"x": 100, "y": 0, "start_time": 10, "end_time": 20},
            ],
            [{"x": 100, "y": 0, "start_time": 0, "end_time": 20}],
        ),
        # Test case 3: Multiple identical segments
        (
            [
                {"x": 100, "y": 0, "start_time": 0, "end_time": 10},
                {"x": 100, "y": 0, "start_time": 10, "end_time": 20},
                {"x": 100, "y": 0, "start_time": 20, "end_time": 30},
            ],
            [{"x": 100, "y": 0, "start_time": 0, "end_time": 30}],
        ),
        # Test case 4: Identical X but different Y
        (
            [
                {"x": 100, "y": 0, "start_time": 0, "end_time": 10},
                {"x": 100, "y": 50, "start_time": 10, "end_time": 20},
            ],
            [
                {"x": 100, "y": 0, "start_time": 0, "end_time": 10},
                {"x": 100, "y": 50, "start_time": 10, "end_time": 20},
            ],
        ),
        # Test case 5: Single segment
        (
            [{"x": 100, "y": 0, "start_time": 0, "end_time": 10}],
            [{"x": 100, "y": 0, "start_time": 0, "end_time": 10}],
        ),
        # Test case 6: Empty list
        ([], []),
        # Test case 7: Segments with very slight differences in X
        (
            [
                {"x": 100, "y": 0, "start_time": 0, "end_time": 10},
                {"x": 101, "y": 0, "start_time": 10, "end_time": 20},
            ],
            [
                {"x": 100, "y": 0, "start_time": 0, "end_time": 20},
            ],
        ),
    ],
)
def test_merge_identical_segments(segments, expected):
    mock_video_file = MagicMock(spec=VideoFile)
    mock_video_file.get_width_pixels.return_value = 1000
    mock_video_file.get_height_pixels.return_value = 1000

    resizer = build_test_resizer()
    merged_segments = resizer._merge_identical_segments(segments, mock_video_file)
    assert merged_segments == expected


def test_merge_identical_segments_preserves_different_speakers():
    mock_video_file = MagicMock(spec=VideoFile)
    mock_video_file.get_width_pixels.return_value = 1000
    mock_video_file.get_height_pixels.return_value = 1000
    segments = [
        {"speakers": [1], "x": 100, "y": 0, "start_time": 0, "end_time": 10},
        {"speakers": [0], "x": 101, "y": 0, "start_time": 10, "end_time": 20},
    ]

    resizer = build_test_resizer()
    merged_segments = resizer._merge_identical_segments(segments, mock_video_file)

    assert merged_segments == segments
