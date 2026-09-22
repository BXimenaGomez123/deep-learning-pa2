import torch
from pytest import fixture, mark

from pa2.synthetic_dataset import SyntheticFrameDataset, SyntheticTrackDataset
from utils import tensor_like


def test_frame_getitem():
    S = 32 # image size
    ds = SyntheticFrameDataset(n_videos=1, n_frames=10, n_objects=4, img_size=S)

    image, target = ds[0]

    assert tensor_like(image, (1, S, S), torch.float32)


def test_track_getitem():
    T = 8 # video section span
    ds = SyntheticTrackDataset(n_videos=1, n_frames=30, n_objects=4, img_size=32, T=T, stride=4)

    boxes, mask = ds[0]

    assert tensor_like(boxes, (T, 4), torch.float32)
    assert tensor_like(mask, (T,), torch.float32)
    assert torch.all((mask == 0) | (mask == 1))
