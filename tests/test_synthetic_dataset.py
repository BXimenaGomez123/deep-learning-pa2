import torch
from torch import int32, float32
from pytest import fixture, mark

from pa2.synthetic_dataset import SyntheticFrameDataset, SyntheticTrackDataset
from utils import tensor_like


def test_frame_getitem():
    S = 32 # image size
    N = 4 # number of objects
    ds = SyntheticFrameDataset(n_videos=1, n_frames=10, n_objects=N, img_size=S)

    image, boxes, ids = ds[0]

    assert tensor_like(image, (3, S, S), float32)
    assert tensor_like(boxes, (N, 4), float32)
    assert tensor_like(ids, (N,), int32)
    # 'ids' tem todos os números inteiros de 0 a N
    assert torch.all(torch.sort(ids)[0] == torch.arange(0,N))


def test_track_getitem():
    T = 8 # video section span
    ds = SyntheticTrackDataset(n_videos=1, n_frames=30, n_objects=4, img_size=32, T=T, stride=4)

    boxes, mask = ds[0]

    assert tensor_like(boxes, (T, 4), float32)
    assert tensor_like(mask, (T,), float32)
    # 'mask' só contém 0s e 1s
    assert torch.all((mask == 0) | (mask == 1))
