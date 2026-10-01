import pytest
import torch

from pa2.mot17_dataset import MOT17FrameDataset, MOT17TrackDataset
from utils import make_fake_mot17


def test_mot17_frame_shapes_and_types(tmp_path):
    root = make_fake_mot17(
        tmp_path, sequence_numbers=("02",), n_frames=2, image_size=(6, 8), include_distractor=True
    )
    dataset = MOT17FrameDataset(root=root)

    image, boxes, ids = dataset[0]

    assert image.shape == (3, 6, 8)
    assert image.dtype == torch.float32
    assert boxes.shape == (1, 4)
    assert boxes.dtype == torch.float32
    assert ids.shape == (1,)
    assert ids.dtype == torch.int32
    assert ids.item() == 7


def test_mot17_track_shapes_and_types(tmp_path):
    root = make_fake_mot17(tmp_path, sequence_numbers=("02",), n_frames=2)
    dataset = MOT17TrackDataset(root=root, T=2, stride=1)

    boxes, valid = dataset[0]

    assert boxes.shape == (2, 4)
    assert boxes.dtype == torch.float32
    assert valid.shape == (2,)
    assert valid.dtype == torch.float32
    assert torch.equal(valid, torch.ones(2))


def test_mot17_missing_data_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="MOT17 data directory not found"):
        MOT17FrameDataset(root=tmp_path / "missing")
