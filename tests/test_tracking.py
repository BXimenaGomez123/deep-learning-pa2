import numpy as np
import pytest

from pa2.tracking import IoUTracker, gated_hungarian


def test_gated_hungarian_maximizes_number_of_valid_matches():
    similarities = np.array([[0.9, 0.5], [0.5, 0.49]])

    assert set(gated_hungarian(similarities, threshold=0.5)) == {
        (0, 1),
        (1, 0),
    }


def test_iou_tracker_only_reconnects_to_the_previous_frame():
    tracker = IoUTracker(iou_threshold=0.3, max_missed_frames=2)
    box = (10, 10, 10, 10)

    first = tracker.update(1, [box])
    assert first.detections == [(box, 1)]
    tracker.update(2, [])
    assert 1 in tracker.tracks

    after_gap = tracker.update(3, [box])
    assert after_gap.detections == [(box, 2)]
    assert 1 not in tracker.tracks


def test_iou_tracker_requires_consecutive_frames():
    tracker = IoUTracker()
    tracker.update(1, [])

    with pytest.raises(ValueError, match="consecutive"):
        tracker.update(3, [])
