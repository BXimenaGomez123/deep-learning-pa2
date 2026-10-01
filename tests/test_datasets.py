from typing import cast

from pa2.datasets import (
    MOT17_SEQUENCE_SPLITS,
    make_mot17_frame_splits,
    make_mot17_track_splits,
)
from pa2.mot17_dataset import MOT17FrameDataset, MOT17TrackDataset
from utils import make_fake_mot17


def test_frame_and_track_factories_share_whole_sequence_splits(tmp_path):
    root = make_fake_mot17(tmp_path)
    frame_splits = make_mot17_frame_splits(root)
    track_splits = make_mot17_track_splits(root, T=2, stride=1)

    for split, expected_ids in MOT17_SEQUENCE_SPLITS.items():
        frame_subset = frame_splits[split]
        frame_dataset = cast(MOT17FrameDataset, frame_subset.dataset)
        frame_ids = {
            frame_dataset.sequences[frame_dataset.index[index][0]].name.split("-")[1]
            for index in frame_subset.indices
        }

        track_subset = track_splits[split]
        track_dataset = cast(MOT17TrackDataset, track_subset.dataset)
        track_ids = {
            track_dataset.sample_sequences[index].split("-")[1]
            for index in track_subset.indices
        }

        assert frame_ids == set(expected_ids)
        assert track_ids == set(expected_ids)
