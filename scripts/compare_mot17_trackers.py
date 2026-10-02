"""Compare consecutive-frame IoU tracking with GRU box prediction on MOT17."""

from pathlib import Path
import csv
from typing import cast

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from tqdm import tqdm
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

from pa2.datasets import MOT17_SEQUENCE_SPLITS, make_mot17_frame_splits
from pa2.gru_model import GRUSequenceModel
from pa2.metrics import Box, compute_idf1, count_id_switches
from pa2.mot17_dataset import MOT17FrameDataset
from pa2.tracking import GRUTracker, IoUTracker, TrackingResult
from predict_mot17_tracks import (
    HISTORY_LENGTH,
    IOU_THRESHOLD,
    MAX_MISSED_FRAMES,
    detect_people,
)


VIDEO_SEQUENCE = "MOT17-13-SDP"
VIDEO_MAX_FRAMES = 60
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def evaluate_tracker(ground_truth, tracked_detections):
    gt_boxes, gt_ids = ground_truth
    pred_boxes = [[box for box, _ in frame] for frame in tracked_detections]
    pred_ids = [[track_id for _, track_id in frame] for frame in tracked_detections]
    idf1, counts = compute_idf1(gt_boxes, gt_ids, pred_boxes, pred_ids)
    switches = count_id_switches(gt_boxes, gt_ids, pred_boxes, pred_ids)
    return idf1, switches, counts


def show_comparison(
    frames,
    frame_numbers,
    iou_tracks: list[TrackingResult],
    gru_tracks: list[TrackingResult],
):
    """Animate each tracker's detection IDs side by side."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    figure.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.88, wspace=0.02)
    images = [axis.imshow(frames[0]) for axis in axes]
    for axis in axes:
        axis.set_axis_off()

    rectangles = [[], []]
    labels = [[], []]

    def update(frame_index):
        for image in images:
            image.set_data(frames[frame_index])
        for panel in range(2):
            for artist in rectangles[panel] + labels[panel]:
                artist.remove()
            rectangles[panel].clear()
            labels[panel].clear()

        for panel, (axis, tracked) in enumerate(zip(axes, (iou_tracks, gru_tracks))):
            for box, track_id in tracked[frame_index].detections:
                x, y, width, height = box
                color = plt.cm.tab20(track_id % 20)
                rectangle = patches.Rectangle(
                    (x, y), width, height, fill=False, edgecolor=color, linewidth=2
                )
                label = axis.text(x, y, str(track_id), color=color, fontsize=9)
                axis.add_patch(rectangle)
                rectangles[panel].append(rectangle)
                labels[panel].append(label)

        # Dashed boxes on the GRU panel show its next-frame motion predictions.
        for box, track_id in gru_tracks[frame_index].predictions:
            x, y, width, height = box
            rectangle = patches.Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                edgecolor=plt.cm.tab20(track_id % 20),
                linestyle="--",
            )
            axes[1].add_patch(rectangle)
            rectangles[1].append(rectangle)

        axes[0].set_title(f"Consecutive-frame IoU | frame {frame_numbers[frame_index]}")
        axes[1].set_title(f"GRU prediction + IoU | frame {frame_numbers[frame_index]}")
        return [
            *images,
            *(artist for panel in rectangles + labels for artist in panel),
        ]

    video = animation.FuncAnimation(
        figure, update, frames=len(frames), interval=50, blit=False
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "mot17_tracker_comparison.mp4"
    print(f"Saving compact comparison video to {output_path}...", flush=True)
    writer = animation.FFMpegWriter(
        fps=10,
        codec="libx264",
        bitrate=1400,
        extra_args=["-pix_fmt", "yuv420p", "-crf", "25"],
    )
    video.save(output_path, writer=writer, dpi=100)
    print(f"Saved video: {output_path}", flush=True)
    plt.show()
    return video


def mean_longest_gt_gap(frame_numbers, ids_per_frame):
    """Mean per-person longest absence between GT annotations (occlusion proxy)."""
    appearances: dict[int, list[int]] = {}
    for frame_number, ids in zip(frame_numbers, ids_per_frame):
        for identity in ids:
            appearances.setdefault(identity, []).append(frame_number)
    longest_gaps = [
        max((later - earlier - 1 for earlier, later in zip(frames, frames[1:])), default=0)
        for frames in appearances.values()
    ]
    return sum(longest_gaps) / len(longest_gaps) if longest_gaps else 0.0


def evaluate_sequence(dataset, sequence_name, split, detector, gru, device):
    sequence_index = next(
        (
            index
            for index, sequence in enumerate(dataset.sequences)
            if sequence.name == sequence_name
        ),
        None,
    )
    if sequence_index is None:
        raise FileNotFoundError(f"Missing sequence {sequence_name}")

    dataset_indices = [
        index
        for index, (source_index, _) in enumerate(dataset.index)
        if source_index == sequence_index
    ]
    sequence = dataset.sequences[sequence_index]
    frame_numbers = [sequence.frame_numbers[dataset.index[index][1]] for index in dataset_indices]
    if any(later != earlier + 1 for earlier, later in zip(frame_numbers, frame_numbers[1:])):
        raise ValueError(f"{sequence_name} must contain consecutive frames")

    iou_tracker = IoUTracker(IOU_THRESHOLD, MAX_MISSED_FRAMES)
    gru_tracker = GRUTracker(
        gru,
        device,
        iou_threshold=IOU_THRESHOLD,
        max_missed_frames=MAX_MISSED_FRAMES,
        history_length=HISTORY_LENGTH,
    )
    iou_results, gru_results = [], []
    gt_boxes: list[list[Box]] = []
    gt_ids: list[list[int]] = []
    video_frames, video_iou, video_gru = [], [], []
    video_numbers = []

    for position, dataset_index in enumerate(
        tqdm(dataset_indices, desc=sequence_name, unit="frame", leave=False)
    ):
        image, boxes, ids = dataset[dataset_index]
        detections = detect_people(detector, image, device)
        image_height, image_width = image.shape[1:]
        iou_result = iou_tracker.update(frame_numbers[position], detections)
        gru_result = gru_tracker.update(detections, (image_height, image_width))
        iou_results.append(iou_result)
        gru_results.append(gru_result)
        gt_boxes.append(
            [(int(box[0]), int(box[1]), int(box[2]), int(box[3])) for box in boxes]
        )
        gt_ids.append([int(identity) for identity in ids.tolist()])

        if sequence_name == VIDEO_SEQUENCE and len(video_frames) < VIDEO_MAX_FRAMES:
            video_frames.append(image.permute(1, 2, 0).mul(255).byte().cpu().numpy())
            video_iou.append(iou_result)
            video_gru.append(gru_result)
            video_numbers.append(frame_numbers[position])

    ground_truth = (gt_boxes, gt_ids)
    iou_idf1, iou_switches, _ = evaluate_tracker(
        ground_truth, [result.detections for result in iou_results]
    )
    gru_idf1, gru_switches, _ = evaluate_tracker(
        ground_truth, [result.detections for result in gru_results]
    )
    return {
        "sequence": sequence_name,
        "split": split,
        "mean_longest_gt_gap": mean_longest_gt_gap(frame_numbers, gt_ids),
        "iou_idf1": iou_idf1,
        "gru_idf1": gru_idf1,
        "idf1_delta": gru_idf1 - iou_idf1,
        "iou_id_switches": iou_switches,
        "gru_id_switches": gru_switches,
    }, (video_frames, video_numbers, video_iou, video_gru)


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    print("Loading pretrained Faster R-CNN detector...", flush=True)
    detector = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    ).to(device).eval()
    print("Faster R-CNN is ready.", flush=True)

    gru = GRUSequenceModel(input_size=4, hidden_size=32, output_size=4).to(device)
    weights_path = Path(__file__).resolve().parents[1] / "weights" / "mot17_gru.pt"
    print(f"Loading trained GRU from {weights_path}...", flush=True)
    gru.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    gru.eval()

    print("Preparing the shared MOT17 sequence splits...", flush=True)
    splits = make_mot17_frame_splits()
    dataset = cast(MOT17FrameDataset, splits["train"].dataset)
    source_splits = {
        number: split
        for split, numbers in MOT17_SEQUENCE_SPLITS.items()
        for number in numbers
    }
    rows = []
    video_data = None

    with torch.inference_mode():
        for number, split in source_splits.items():
            sequence_name = f"MOT17-{number}-SDP"
            print(f"Evaluating {sequence_name} ({split})...", flush=True)
            row, candidate_video = evaluate_sequence(
                dataset, sequence_name, split, detector, gru, device
            )
            rows.append(row)
            print(
                f"{sequence_name}: IoU IDF1={row['iou_idf1']:.3f}, "
                f"GRU IDF1={row['gru_idf1']:.3f}, "
                f"delta={row['idf1_delta']:+.3f}"
            )
            if sequence_name == VIDEO_SEQUENCE:
                video_data = candidate_video

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "mot17_tracker_metrics.csv"
    with csv_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    ordered = sorted(rows, key=lambda row: row["mean_longest_gt_gap"])
    gaps = [row["mean_longest_gt_gap"] for row in ordered]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(gaps, [row["iou_idf1"] for row in ordered], "o-", label="Consecutive-frame IoU")
    axis.plot(gaps, [row["gru_idf1"] for row in ordered], "s-", label="GRU prediction + IoU")
    for row in ordered:
        marker = "*" if row["split"] == "train" else ""
        axis.annotate(
            f"{row['sequence'][-2:]}{marker}",
            (row["mean_longest_gt_gap"], row["iou_idf1"]),
            xytext=(3, -12),
            textcoords="offset points",
            fontsize=8,
        )
        axis.annotate(
            f"{row['sequence'][-2:]}{marker}",
            (row["mean_longest_gt_gap"], row["gru_idf1"]),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Mean per-person longest GT annotation gap (frames; occlusion proxy)")
    axis.set_ylabel("IDF1")
    axis.set_ylim(0, 1.05)
    axis.set_title("Tracking IDF1 versus sequence difficulty (* = GRU training sequence)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    plot_path = OUTPUT_DIR / "mot17_idf1_vs_annotation_gap.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    mean_iou_idf1 = sum(row["iou_idf1"] for row in rows) / len(rows)
    mean_gru_idf1 = sum(row["gru_idf1"] for row in rows) / len(rows)
    print("\nPer-sequence summary (IDF1 higher is better):")
    print("Sequence  Split       GT-gap  IoU IDF1  GRU IDF1  Delta   IoU switches  GRU switches")
    for row in ordered:
        print(
            f"{row['sequence']:<9} {row['split']:<11} "
            f"{row['mean_longest_gt_gap']:>6.1f}  "
            f"{row['iou_idf1']:>8.3f}  {row['gru_idf1']:>8.3f}  "
            f"{row['idf1_delta']:>+6.3f}  "
            f"{row['iou_id_switches']:>12}  {row['gru_id_switches']:>12}"
        )
    print(
        f"\nMacro-average IDF1: IoU={mean_iou_idf1:.3f}, "
        f"GRU={mean_gru_idf1:.3f}, delta={mean_gru_idf1 - mean_iou_idf1:+.3f}"
    )
    print("GT-gap is mean per-person longest annotation absence (an occlusion proxy).")
    print(f"Saved metrics to {csv_path} and plot to {plot_path}")

    if video_data is None:
        raise RuntimeError(f"No visualization frames collected for {VIDEO_SEQUENCE}")
    frames, frame_numbers, iou_results, gru_results = video_data
    print("Saving the compact side-by-side video...", flush=True)
    show_comparison(frames, frame_numbers, iou_results, gru_results)


if __name__ == "__main__":
    main()
