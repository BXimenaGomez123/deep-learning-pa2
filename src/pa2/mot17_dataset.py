"""PyTorch datasets for MOT17 in MOTChallenge directory layout."""

from __future__ import annotations

import configparser
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class _Sequence:
    """Loaded MOT17 sequence metadata and per-frame annotations."""

    name: str
    image_paths: list[Path]
    frame_numbers: list[int]
    boxes: dict[int, np.ndarray]
    ids: dict[int, np.ndarray]


def _sequence_dirs(root: Path, split: str) -> list[Path]:
    images_root = root / "MOT17" / split
    return sorted(p.parent for p in images_root.glob("*/seqinfo.ini"))


def _find_gt(root: Path, split: str, sequence_dir: Path) -> Path | None:
    candidates = (
        sequence_dir / "gt" / "gt.txt",
        root / "MOT17Labels" / split / sequence_dir.name / "gt" / "gt.txt",
    )
    return next((path for path in candidates if path.is_file()), None)


def _read_annotations(path: Path) -> tuple[dict[int, list[list[float]]], dict[int, list[int]]]:
    boxes: dict[int, list[list[float]]] = {}
    ids: dict[int, list[int]] = {}

    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                values = [float(value) for value in row]
            except ValueError as exc:
                raise ValueError(f"Invalid MOT annotation row in {path}: {row}") from exc

            if len(values) < 6:
                raise ValueError(f"Expected at least 6 columns in {path}, got: {row}")

            frame, object_id = int(values[0]), int(values[1])

            # MOT17 rows are frame,id,x,y,w,h,mark,class,visibility. Keep
            # marked pedestrian annotations and ignore distractor classes.
            if len(values) >= 8 and (values[6] != 1 or values[7] != 1):
                continue

            boxes.setdefault(frame, []).append(values[2:6])
            ids.setdefault(frame, []).append(object_id)

    return boxes, ids


def _load_sequences(root: Path, split: str) -> list[_Sequence]:
    if not root.is_dir():
        raise FileNotFoundError(
            f"MOT17 data directory not found: {root}. "
            "Place the dataset under data/MOT17 (and optionally data/MOT17Labels)."
        )

    dirs = _sequence_dirs(root, split)
    if not dirs:
        raise FileNotFoundError(
            f"No MOT17 sequences found under {root / 'MOT17' / split}. "
            "Expected folders containing seqinfo.ini and an image directory."
        )

    sequences = []
    missing_gt = []
    for sequence_dir in dirs:
        info_path = sequence_dir / "seqinfo.ini"
        config = configparser.ConfigParser()
        config.read(info_path)
        if not config.has_section("Sequence"):
            raise ValueError(f"Missing [Sequence] section in {info_path}")

        info = config["Sequence"]
        image_dir = sequence_dir / info.get("imDir", "img1")
        image_ext = info.get("imExt", ".jpg")
        if not image_ext.startswith("."):
            image_ext = f".{image_ext}"

        # Use the numbered image files actually present rather than assuming
        # every sequence has all frames (useful for partial local datasets).
        image_by_frame = {}
        for image_path in image_dir.glob(f"*{image_ext}"):
            try:
                image_by_frame[int(image_path.stem)] = image_path
            except ValueError:
                continue
        frame_numbers = sorted(image_by_frame)
        if not frame_numbers:
            continue

        gt_path = _find_gt(root, split, sequence_dir)
        if gt_path is None:
            missing_gt.append(sequence_dir.name)
            continue

        raw_boxes, raw_ids = _read_annotations(gt_path)
        sequences.append(
            _Sequence(
                name=sequence_dir.name,
                image_paths=[image_by_frame[frame] for frame in frame_numbers],
                frame_numbers=frame_numbers,
                boxes={
                    frame: np.asarray(raw_boxes.get(frame, []), dtype=np.float32).reshape(-1, 4)
                    for frame in frame_numbers
                },
                ids={
                    frame: np.asarray(raw_ids.get(frame, []), dtype=np.int32)
                    for frame in frame_numbers
                },
            )
        )

    if not sequences:
        details = f" Sequences without ground-truth labels: {', '.join(missing_gt)}." if missing_gt else ""
        raise FileNotFoundError(
            f"No usable labeled MOT17 sequences found for split '{split}' under {root}."
            f"{details} Expected labels at <sequence>/gt/gt.txt or "
            "MOT17Labels/<split>/<sequence>/gt/gt.txt."
        )

    return sequences


class MOT17FrameDataset(Dataset):
    """One sample = one MOT17 frame: ``(image, boxes, ids)``.

    Images are RGB float tensors with shape ``[3, H, W]`` and values in
    ``[0, 1]``. Boxes are float ``[N, 4]`` arrays in ``(x, y, w, h)`` format;
    IDs are int32 ``[N]`` tensors. Image resolution is kept as stored in MOT17.
    """

    def __init__(self, root="data", split="train"):
        self.sequences = _load_sequences(Path(root), split)
        self.index = [
            (sequence_index, image_index)
            for sequence_index, sequence in enumerate(self.sequences)
            for image_index in range(len(sequence.image_paths))
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        sequence_index, image_index = self.index[index]
        sequence = self.sequences[sequence_index]
        frame = sequence.frame_numbers[image_index]

        with Image.open(sequence.image_paths[image_index]) as image_file:
            image = np.asarray(image_file.convert("RGB"))
        image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        boxes = torch.from_numpy(sequence.boxes[frame]).float()
        ids = torch.from_numpy(sequence.ids[frame]).int()
        return image_tensor, boxes, ids


class MOT17TrackDataset(Dataset):
    """One sample = one identity's box sequence over a sliding frame window.

    Returns ``(boxes, valid)`` with float tensors of shapes ``[T, 4]`` and
    ``[T]``. Missing annotations are zero-filled and marked invalid, matching
    :class:`pa2.synthetic_dataset.SyntheticTrackDataset`.
    """

    def __init__(self, root="data", split="train", T=16, stride=4):
        if T <= 0:
            raise ValueError("T must be a positive integer")
        if stride <= 0:
            raise ValueError("stride must be a positive integer")

        self.T = T
        self.samples = []
        for sequence in _load_sequences(Path(root), split):
            frame_to_index = {frame: index for index, frame in enumerate(sequence.frame_numbers)}
            all_ids = sorted({int(object_id) for values in sequence.ids.values() for object_id in values})

            for object_id in all_ids:
                track = [None] * len(sequence.frame_numbers)
                for frame in sequence.frame_numbers:
                    matching = np.flatnonzero(sequence.ids[frame] == object_id)
                    if len(matching):
                        track[frame_to_index[frame]] = sequence.boxes[frame][matching[0]]

                for start in range(0, len(track) - T + 1, stride):
                    box_seq = np.zeros((T, 4), dtype=np.float32)
                    valid = np.zeros((T,), dtype=np.float32)
                    for offset, box in enumerate(track[start : start + T]):
                        if box is not None:
                            box_seq[offset] = box
                            valid[offset] = 1.0
                    if valid.sum() >= 2:
                        self.samples.append((box_seq, valid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        box_seq, valid = self.samples[index]
        return torch.from_numpy(box_seq), torch.from_numpy(valid)
