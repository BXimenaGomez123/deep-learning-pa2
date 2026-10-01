"""Animate samples from the synthetic and MOT17 frame datasets."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from pa2.box_corruption import corrupt_track_boxes
from pa2.mot17_dataset import MOT17FrameDataset
from pa2.plotting import plot_video
from pa2.synthetic_dataset import SyntheticFrameDataset


def _to_video_tensors(dataset, indices):
    frames, boxes_per_frame, ids_per_frame = [], [], []
    for index in indices:
        image, boxes, ids = dataset[index]
        # Dataset image tensors are [C, H, W]; plot_video expects [H, W, C].
        frames.append((image.permute(1, 2, 0) * 255).byte().cpu().numpy())
        boxes_per_frame.append(boxes.cpu().numpy())
        ids_per_frame.append(ids.cpu().numpy())
    return frames, boxes_per_frame, ids_per_frame


def _corrupt_synthetic_video(video, drop_fraction, noise_std, false_positives, seed):
    frames, boxes_per_frame, ids_per_frame = video
    n_frames, height, width = frames.shape[:3]
    object_ids = sorted({int(object_id) for frame_ids in ids_per_frame for object_id in frame_ids})
    corrupted_boxes = [[] for _ in range(n_frames)]
    corrupted_ids = [[] for _ in range(n_frames)]
    next_false_positive_id = max(object_ids, default=-1) + 1
    generator = torch.Generator().manual_seed(seed)

    for object_id in object_ids:
        track_boxes = torch.zeros((n_frames, 4), dtype=torch.float32)
        valid = torch.zeros(n_frames, dtype=torch.float32)

        for frame_index, (frame_boxes, frame_ids) in enumerate(
            zip(boxes_per_frame, ids_per_frame)
        ):
            matches = np.flatnonzero(frame_ids == object_id)
            if len(matches):
                track_boxes[frame_index] = torch.from_numpy(frame_boxes[matches[0]])
                valid[frame_index] = 1.0

        available_false_positive_slots = int((valid == 0).sum())
        corrupted_track, corrupted_valid = corrupt_track_boxes(
            track_boxes,
            valid,
            drop_fraction=drop_fraction,
            noise_std=noise_std,
            false_positive_count=min(false_positives, available_false_positive_slots),
            image_size=(height, width),
            generator=generator,
        )

        for frame_index in torch.where(corrupted_valid > 0)[0].tolist():
            corrupted_boxes[frame_index].append(corrupted_track[frame_index].numpy())
            if valid[frame_index] > 0:
                corrupted_ids[frame_index].append(object_id)
            else:
                corrupted_ids[frame_index].append(next_false_positive_id)
                next_false_positive_id += 1

    return frames, corrupted_boxes, corrupted_ids


def _show_video(title, frames, boxes, ids, interval):
    animation = plot_video(frames, boxes, ids, interval=interval)
    figure = plt.gcf()
    figure.suptitle(title)
    # plot/show one animation at a time; keep it alive until the window closes.
    plt.show()
    plt.close(figure)
    return animation


def main():
    parser = argparse.ArgumentParser(
        description="Plot frame samples from the synthetic and MOT17 datasets."
    )
    parser.add_argument("--data-root", default="data", help="MOT17 data directory (default: data)")
    parser.add_argument("--split", default="train", help="MOT17 split (default: train)")
    parser.add_argument("--sequence", help="MOT17 sequence name; defaults to the first available")
    parser.add_argument("--start", type=int, default=0, help="Zero-based start frame within each video")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to animate")
    parser.add_argument("--interval", type=int, default=50, help="Animation interval in milliseconds")
    parser.add_argument("--drop-fraction", type=float, default=0.15)
    parser.add_argument("--noise-std", type=float, default=2.0)
    parser.add_argument("--false-positives", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if not 0.0 <= args.drop_fraction <= 1.0:
        parser.error("--drop-fraction must be between 0 and 1")
    if args.noise_std < 0:
        parser.error("--noise-std must be non-negative")
    if args.false_positives < 0:
        parser.error("--false-positives must be non-negative")

    # The synthetic frame dataset flattens videos into frame samples. Generate
    # enough frames to support the requested start offset and slice.
    synthetic = SyntheticFrameDataset(
        n_videos=1,
        n_frames=max(50, args.start + args.frames),
    )
    synthetic_indices = range(args.start, args.start + args.frames)
    frames, boxes, ids = _to_video_tensors(synthetic, synthetic_indices)
    _show_video("SyntheticFrameDataset — ellipse ground truth", frames, boxes, ids, args.interval)

    mot17 = MOT17FrameDataset(root=args.data_root, split=args.split)
    if args.sequence is None:
        sequence_name = mot17.sequences[0].name
    else:
        sequence_name = args.sequence

    sequence_index = next(
        (i for i, sequence in enumerate(mot17.sequences) if sequence.name == sequence_name),
        None,
    )
    if sequence_index is None:
        available = ", ".join(sequence.name for sequence in mot17.sequences)
        parser.error(f"Sequence {sequence_name!r} not found. Available: {available}")

    mot17_indices = [
        dataset_index
        for dataset_index, (seq_index, _) in enumerate(mot17.index)
        if seq_index == sequence_index
    ]
    mot17_indices = mot17_indices[args.start : args.start + args.frames]
    if not mot17_indices:
        parser.error(
            f"No frames available in {sequence_name!r} starting at offset {args.start}."
        )

    frames, boxes, ids = _to_video_tensors(mot17, mot17_indices)
    _show_video(f"MOT17FrameDataset — {sequence_name}", frames, boxes, ids, args.interval)

    corrupted_video = _corrupt_synthetic_video(
        synthetic.videos[0],
        drop_fraction=args.drop_fraction,
        noise_std=args.noise_std,
        false_positives=args.false_positives,
        seed=args.seed,
    )
    corrupted_frames, corrupted_boxes, corrupted_ids = corrupted_video
    frame_slice = slice(args.start, args.start + args.frames)
    _show_video(
        "Synthetic ellipses — deliberately corrupted boxes",
        corrupted_frames[frame_slice],
        corrupted_boxes[frame_slice],
        corrupted_ids[frame_slice],
        args.interval,
    )


if __name__ == "__main__":
    main()
