import numpy as np
import pytest
import torch
from PIL import Image

from pa2.mot17_dataset import MOT17FrameDataset, MOT17TrackDataset


def make_tiny_mot17(root):
    sequence = root / "MOT17" / "train" / "MOT17-02-SDP"
    image_dir = sequence / "img1"
    image_dir.mkdir(parents=True)
    (sequence / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT17-02-SDP\nimDir=img1\nimExt=.jpg\n"
    )
    for frame in (1, 2):
        image = np.full((6, 8, 3), frame * 40, dtype=np.uint8)
        Image.fromarray(image).save(image_dir / f"{frame:06d}.jpg")

    gt_dir = root / "MOT17Labels" / "train" / "MOT17-02-SDP" / "gt"
    gt_dir.mkdir(parents=True)
    # Valid person track plus a non-pedestrian class that should be ignored.
    (gt_dir / "gt.txt").write_text(
        "1,7,1,2,3,4,1,1,1\n"
        "1,99,2,2,3,4,1,7,1\n"
        "2,7,2,3,3,4,1,1,1\n"
    )
    return root


def test_mot17_frame_shapes_and_types(tmp_path):
    dataset = MOT17FrameDataset(root=make_tiny_mot17(tmp_path))

    image, boxes, ids = dataset[0]

    assert image.shape == (3, 6, 8)
    assert image.dtype == torch.float32
    assert boxes.shape == (1, 4)
    assert boxes.dtype == torch.float32
    assert ids.shape == (1,)
    assert ids.dtype == torch.int32
    assert ids.item() == 7


def test_mot17_track_shapes_and_types(tmp_path):
    dataset = MOT17TrackDataset(root=make_tiny_mot17(tmp_path), T=2, stride=1)

    boxes, valid = dataset[0]

    assert boxes.shape == (2, 4)
    assert boxes.dtype == torch.float32
    assert valid.shape == (2,)
    assert valid.dtype == torch.float32
    assert torch.equal(valid, torch.ones(2))


def test_mot17_missing_data_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="MOT17 data directory not found"):
        MOT17FrameDataset(root=tmp_path / "missing")
