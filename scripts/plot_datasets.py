"""Animate samples from the synthetic and MOT17 frame datasets."""

import argparse

import matplotlib.pyplot as plt

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
    args = parser.parse_args()

    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.frames <= 0:
        parser.error("--frames must be positive")

    # The synthetic frame dataset flattens videos into frame samples. Generate
    # enough frames to support the requested start offset and slice.
    synthetic = SyntheticFrameDataset(
        n_videos=1,
        n_frames=max(50, args.start + args.frames),
    )
    synthetic_indices = range(args.start, args.start + args.frames)
    frames, boxes, ids = _to_video_tensors(synthetic, synthetic_indices)
    synthetic_animation = plot_video(frames, boxes, ids, interval=args.interval)
    plt.gcf().suptitle("SyntheticFrameDataset")

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
    mot17_animation = plot_video(frames, boxes, ids, interval=args.interval)
    plt.gcf().suptitle(f"MOT17FrameDataset — {sequence_name}")

    # Keep animation objects alive until the GUI event loop is running.
    _animations = (synthetic_animation, mot17_animation)
    plt.show()


if __name__ == "__main__":
    main()
