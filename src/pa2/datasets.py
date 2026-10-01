
"""Deterministic, sequence-level MOT17 train/validation/test splits."""

from torch.utils.data import Subset

from pa2.mot17_dataset import MOT17FrameDataset, MOT17TrackDataset


# Split by source sequence number, so DPM/FRCNN/SDP variants stay together.
# MOT17-05 is the low-resolution (640x480), low-frame-rate outlier and is
# reserved for validation. MOT17-13 is held out for final testing.
MOT17_SEQUENCE_SPLITS = {
    "train": ("02", "04", "09", "10", "11"),
    "validation": ("05",),
    "test": ("13",),
}


def _sequence_number(name: str) -> str:
    """Extract the source sequence ID from names such as MOT17-02-SDP."""
    parts = name.split("-")
    if len(parts) < 3 or parts[0] != "MOT17":
        raise ValueError(f"Unexpected MOT17 sequence name: {name}")
    return parts[1]


def _split_for_sequence(name: str) -> str:
    number = _sequence_number(name)
    for split, sequence_numbers in MOT17_SEQUENCE_SPLITS.items():
        if number in sequence_numbers:
            return split
    raise ValueError(f"No dataset split configured for MOT17 sequence {name}")


def make_mot17_frame_splits(root="data") -> dict[str, Subset[MOT17FrameDataset]]:
    """Return frame datasets split by whole source sequence, never by frame."""
    dataset = MOT17FrameDataset(root=root, split="train")
    split_indices = {split: [] for split in MOT17_SEQUENCE_SPLITS}

    for index, (sequence_index, _) in enumerate(dataset.index):
        sequence = dataset.sequences[sequence_index]
        split_indices[_split_for_sequence(sequence.name)].append(index)

    return {split: Subset(dataset, indices) for split, indices in split_indices.items()}


def make_mot17_track_splits(root="data", T=16, stride=4) -> dict[str, Subset[MOT17TrackDataset]]:
    """Return track windows split by source sequence, never by frame."""
    dataset = MOT17TrackDataset(root=root, split="train", T=T, stride=stride)
    split_indices = {split: [] for split in MOT17_SEQUENCE_SPLITS}
    for index, sequence_name in enumerate(dataset.sample_sequences):
        split_indices[_split_for_sequence(sequence_name)].append(index)
    return {split: Subset(dataset, indices) for split, indices in split_indices.items()}
