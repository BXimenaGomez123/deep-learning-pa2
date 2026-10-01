import pytest
import torch
from torch import float32

from pa2.box_corruption import corrupt_track_boxes
from pa2.synthetic_dataset import SyntheticTrackDataset
from utils import tensor_like


def make_track_sample():
    boxes = torch.tensor(
        [
            [10.0, 12.0, 8.0, 14.0],
            [12.0, 13.0, 8.0, 14.0],
            [14.0, 14.0, 8.0, 14.0],
            [16.0, 15.0, 8.0, 14.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float32,
    )
    mask = torch.tensor([1, 1, 1, 1, 0, 0], dtype=float32)
    return boxes, mask


def test_drop_fraction_drops_boxes_and_preserves_dataset_output_types():
    boxes, mask = make_track_sample()
    original_boxes, original_mask = boxes.clone(), mask.clone()

    corrupted_boxes, corrupted_mask = corrupt_track_boxes(
        boxes,
        mask,
        drop_fraction=0.5,
        generator=torch.Generator().manual_seed(1),
    )

    assert tensor_like(corrupted_boxes, (6, 4), float32)
    assert tensor_like(corrupted_mask, (6,), float32)
    assert corrupted_mask.sum() == 2
    assert torch.all(corrupted_boxes[corrupted_mask == 0] == 0)
    assert torch.equal(boxes, original_boxes)
    assert torch.equal(mask, original_mask)


def test_noise_changes_retained_boxes_without_invalidating_them():
    boxes, mask = make_track_sample()

    corrupted_boxes, corrupted_mask = corrupt_track_boxes(
        boxes,
        mask,
        noise_std=2.0,
        image_size=(100, 100),
        generator=torch.Generator().manual_seed(2),
    )

    assert torch.equal(corrupted_mask, mask)
    assert not torch.equal(corrupted_boxes[:4], boxes[:4])
    assert torch.all(corrupted_boxes[:4, 2:] > 0)
    assert torch.all(corrupted_boxes[:4, 0] >= 0)
    assert torch.all(corrupted_boxes[:4, 1] >= 0)
    assert torch.all(corrupted_boxes[:4, 0] + corrupted_boxes[:4, 2] <= 100)
    assert torch.all(corrupted_boxes[:4, 1] + corrupted_boxes[:4, 3] <= 100)


def test_false_positives_use_invalid_mask_slots():
    boxes, mask = make_track_sample()

    corrupted_boxes, corrupted_mask = corrupt_track_boxes(
        boxes,
        mask,
        false_positive_count=2,
        image_size=(100, 100),
        generator=torch.Generator().manual_seed(3),
    )

    assert torch.equal(corrupted_mask, torch.ones_like(mask))
    assert torch.all(corrupted_boxes[4:, 2:] > 0)
    assert torch.all(corrupted_boxes[4:, 0] + corrupted_boxes[4:, 2] <= 100)
    assert torch.all(corrupted_boxes[4:, 1] + corrupted_boxes[4:, 3] <= 100)


def test_false_positives_cannot_exceed_available_invalid_slots():
    boxes, mask = make_track_sample()

    with pytest.raises(ValueError, match="originally invalid slots"):
        corrupt_track_boxes(boxes, mask, false_positive_count=3)


def test_corrupt_track_boxes_accepts_synthetic_dataset_sample():
    dataset = SyntheticTrackDataset(
        n_videos=1,
        n_frames=20,
        n_objects=3,
        img_size=32,
        T=8,
        stride=4,
    )
    boxes, mask = dataset[0]

    corrupted_boxes, corrupted_mask = corrupt_track_boxes(
        boxes,
        mask,
        drop_fraction=0.25,
        noise_std=1.0,
        generator=torch.Generator().manual_seed(4),
    )

    assert tensor_like(corrupted_boxes, (8, 4), float32)
    assert tensor_like(corrupted_mask, (8,), float32)
    assert torch.all((corrupted_mask == 0) | (corrupted_mask == 1))
