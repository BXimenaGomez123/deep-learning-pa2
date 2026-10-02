"""Rank GRU tracking failures on training MOT17 sequences and save a case video."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

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
from pa2.metrics import Box
from pa2.mot17_dataset import MOT17FrameDataset
from pa2.tracking import GRUTracker, TrackingResult
from pa2.tracking_analysis import IdentityError, analyze_identity_errors
from predict_mot17_tracks import HISTORY_LENGTH, IOU_THRESHOLD, MAX_MISSED_FRAMES, detect_people


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
IOU_EVALUATION_THRESHOLD = 0.5
LONG_GAP_THRESHOLD = 10


@torch.inference_mode()
def evaluate_training_sequences(detector, model, device):
    """Run detector and GRU once per frame, retaining compact tracking results."""
    train_subset = make_mot17_frame_splits(root=ROOT / "data")["train"]
    dataset = cast(MOT17FrameDataset, train_subset.dataset)
    source_numbers = set(MOT17_SEQUENCE_SPLITS["train"])
    sequence_indices = [
        index
        for index, sequence in enumerate(dataset.sequences)
        if sequence.name.endswith("-SDP")
        and sequence.name.split("-")[1] in source_numbers
    ]
    if not sequence_indices:
        raise FileNotFoundError("No training SDP sequences were found in data/MOT17/train")

    evaluated = {}
    rows = []
    total_frames = sum(
        len(dataset.sequences[index].image_paths) for index in sequence_indices
    )
    progress = tqdm(total=total_frames, desc="Scoring GRU tracks", unit="frame")

    for sequence_index in sequence_indices:
        sequence = dataset.sequences[sequence_index]
        dataset_indices = [
            index
            for index, (source_index, _) in enumerate(dataset.index)
            if source_index == sequence_index
        ]
        frame_numbers = list(sequence.frame_numbers)
        gt_boxes: list[list[Box]] = []
        gt_ids: list[list[int]] = []
        pred_boxes: list[list[Box]] = []
        pred_ids: list[list[int]] = []
        image_sizes: list[tuple[int, int]] = []
        tracking_results: list[TrackingResult] = []

        tracker = GRUTracker(
            model,
            device,
            iou_threshold=IOU_THRESHOLD,
            max_missed_frames=MAX_MISSED_FRAMES,
            history_length=HISTORY_LENGTH,
        )
        for dataset_index in dataset_indices:
            image, boxes, ids = dataset[dataset_index]
            detections = detect_people(detector, image, device)
            image_height, image_width = image.shape[1:]
            result = tracker.update(detections, (image_height, image_width))
            tracking_results.append(result)
            gt_boxes.append(
                [(int(box[0]), int(box[1]), int(box[2]), int(box[3])) for box in boxes]
            )
            gt_ids.append([int(identity) for identity in ids.tolist()])
            pred_boxes.append([box for box, _ in result.detections])
            pred_ids.append([track_id for _, track_id in result.detections])
            image_sizes.append((image_height, image_width))
            progress.update(1)
            progress.set_postfix(sequence=sequence.name, detections=len(detections))

        identities = analyze_identity_errors(
            gt_boxes,
            gt_ids,
            pred_boxes,
            pred_ids,
            frame_numbers,
            image_sizes,
            iou_threshold=IOU_EVALUATION_THRESHOLD,
            long_gap_threshold=LONG_GAP_THRESHOLD,
        )
        evaluated[sequence.name] = {
            "dataset_indices": dataset_indices,
            "frame_numbers": frame_numbers,
            "gt_boxes": gt_boxes,
            "gt_ids": gt_ids,
            "pred_boxes": pred_boxes,
            "pred_ids": pred_ids,
            "tracking_results": tracking_results,
            "identities": identities,
        }
        for identity in identities:
            row = asdict(identity)
            row.update(sequence=sequence.name, error_score=identity.error_score)
            rows.append(row)

    progress.close()
    rows.sort(
        key=lambda row: (
            row["error_score"],
            row["id_switches"],
            row["fragmentations"],
            row["misses"],
        ),
        reverse=True,
    )
    return dataset, evaluated, rows


def _category_count(identity: IdentityError, category: str) -> int:
    return {
        "long_annotation_gap_proxy": identity.long_gap_events,
        "pedestrian_overlap_at_failure": identity.crossing_events,
        "scene_entry_exit": identity.boundary_events,
    }[category]


def select_distinct_candidates(evaluated) -> list[dict[str, Any]]:
    """Select the strongest metric-ranked candidate for each failure context."""
    categories = (
        "long_annotation_gap_proxy",
        "pedestrian_overlap_at_failure",
        "scene_entry_exit",
    )
    candidates = []
    used = set()
    for category in categories:
        category_rows = []
        for sequence_name, data in evaluated.items():
            for identity in data["identities"]:
                if identity.error_score == 0 or _category_count(identity, category) == 0:
                    continue
                category_rows.append((identity, sequence_name))
        category_rows.sort(
            key=lambda item: (
                item[0].error_score,
                item[0].id_switches,
                item[0].fragmentations,
                _category_count(item[0], category),
                item[0].misses,
            ),
            reverse=True,
        )
        selected = next(
            (
                (identity, sequence_name)
                for identity, sequence_name in category_rows
                if (sequence_name, identity.identity_id) not in used
            ),
            None,
        )
        if selected is None:
            print(f"No distinct failure candidate found for {category}.")
            continue
        identity, sequence_name = selected
        used.add((sequence_name, identity.identity_id))
        candidates.append(
            {
                "sequence": sequence_name,
                "identity": identity,
                "category": category,
            }
        )
    return candidates


def save_identity_video(dataset, sequence_data, sequence_name, identity, category):
    """Save a compact full-frame clip with just the chosen GT/GRU trajectory."""
    frame_indices = identity.frame_indices
    first_index, last_index = frame_indices[0], frame_indices[-1]
    dataset_indices = sequence_data["dataset_indices"][first_index : last_index + 1]
    frame_numbers = sequence_data["frame_numbers"][first_index : last_index + 1]
    gt_boxes = sequence_data["gt_boxes"][first_index : last_index + 1]
    gt_ids = sequence_data["gt_ids"][first_index : last_index + 1]
    tracking_results = sequence_data["tracking_results"][first_index : last_index + 1]
    matched_ids = {
        index: predicted_id
        for index, predicted_id in zip(identity.frame_indices, identity.matched_prediction_ids)
    }

    figure, axis = plt.subplots(figsize=(9, 5.1))
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis.set_axis_off()
    image_artist = None
    artists = []
    last_matched_id = None

    def update(local_index):
        nonlocal image_artist, last_matched_id
        dataset_index = dataset_indices[local_index]
        image, _, _ = dataset[dataset_index]
        frame = image.permute(1, 2, 0).mul(255).byte().cpu().numpy()
        if image_artist is None:
            image_artist = axis.imshow(frame)
        else:
            image_artist.set_data(frame)

        for artist in artists:
            artist.remove()
        artists.clear()

        global_index = first_index + local_index
        visible_gt_box = None
        for box, gt_identity in zip(gt_boxes[local_index], gt_ids[local_index]):
            if gt_identity == identity.identity_id:
                visible_gt_box = box
                rectangle = patches.Rectangle(
                    (box[0], box[1]), box[2], box[3], fill=False,
                    edgecolor="#ffd400", linewidth=2, linestyle="--",
                )
                axis.add_patch(rectangle)
                artists.append(rectangle)
                break

        if global_index in matched_ids and matched_ids[global_index] is not None:
            last_matched_id = matched_ids[global_index]

        status = "GT not annotated"
        prediction_to_draw = None
        if visible_gt_box is not None:
            if matched_ids.get(global_index) is not None:
                status = f"matched GRU ID {matched_ids[global_index]}"
            else:
                status = "GT visible, GRU missed"
        elif last_matched_id is not None:
            status = f"GT annotation gap; showing GRU ID {last_matched_id} prediction"

        if last_matched_id is not None:
            prediction_to_draw = next(
                (
                    box
                    for box, track_id in tracking_results[local_index].predictions
                    if track_id == last_matched_id
                ),
                None,
            )
        if prediction_to_draw is not None:
            box, track_id = prediction_to_draw, last_matched_id
            rectangle = patches.Rectangle(
                (box[0], box[1]), box[2], box[3], fill=False,
                edgecolor="#00e5ff", linewidth=2,
            )
            label = axis.text(
                box[0], max(0, box[1] - 5), f"GRU {track_id}",
                color="black", fontsize=9,
                bbox={"facecolor": "#00e5ff", "alpha": 0.8, "pad": 2},
            )
            axis.add_patch(rectangle)
            artists.extend((rectangle, label))

        header = axis.text(
            0.01,
            0.99,
            f"{sequence_name} | GT ID {identity.identity_id} | {category} | "
            f"frame {frame_numbers[local_index]} | {status}",
            transform=axis.transAxes,
            va="top",
            color="white",
            fontsize=10,
            bbox={"facecolor": "black", "alpha": 0.75, "pad": 4},
        )
        artists.append(header)
        return [image_artist, *artists]

    video = animation.FuncAnimation(
        figure, update, frames=len(dataset_indices), interval=50, blit=False
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"gru_failure_{sequence_name}_gt{identity.identity_id}.mp4"
    writer = animation.FFMpegWriter(
        fps=10,
        codec="libx264",
        bitrate=1200,
        extra_args=["-pix_fmt", "yuv420p", "-crf", "25"],
    )
    print(f"Saving selected identity video to {output_path}...", flush=True)
    video.save(output_path, writer=writer, dpi=100)
    plt.close(figure)
    return output_path


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    detector = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    ).to(device).eval()
    model = GRUSequenceModel(input_size=4, hidden_size=32, output_size=4).to(device)
    weights_path = ROOT / "weights" / "mot17_gru.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    dataset, evaluated, ranked_rows = evaluate_training_sequences(detector, model, device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "gru_training_identity_failures.csv"
    fields = (
        "rank", "sequence", "identity_id", "error_score", "id_switches",
        "fragmentations", "misses", "matched_frames", "observed_frames",
        "longest_annotation_gap", "long_gap_events", "crossing_events",
        "boundary_events", "failure_frames",
    )
    with report_path.open("w", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(ranked_rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    **{field: row[field] for field in fields if field in row},
                    "failure_frames": ";".join(map(str, row["failure_frames"])),
                }
            )

    print("\nHighest-ranked GRU identity failures:")
    for rank, row in enumerate(ranked_rows[:10], start=1):
        print(
            f"{rank:>2}. {row['sequence']} GT {row['identity_id']}: "
            f"{row['error_score']} errors "
            f"({row['id_switches']} switches + {row['fragmentations']} fragmentations), "
            f"{row['misses']} visible misses, gap={row['longest_annotation_gap']} frames"
        )

    candidates = select_distinct_candidates(evaluated)
    print("\nDistinct metric-ranked failure candidates:")
    for candidate in candidates:
        identity = candidate["identity"]
        print(
            f"{candidate['category']}: {candidate['sequence']} GT {identity.identity_id}; "
            f"score={identity.error_score}, switches={identity.id_switches}, "
            f"fragmentations={identity.fragmentations}"
        )

    if not candidates:
        raise RuntimeError("No identity had an ID switch or fragmentation to export")
    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["identity"].error_score,
            candidate["identity"].id_switches,
            candidate["identity"].fragmentations,
        ),
    )
    video_path = save_identity_video(
        dataset,
        evaluated[selected["sequence"]],
        selected["sequence"],
        selected["identity"],
        selected["category"],
    )
    print(f"Saved ranked identity report: {report_path}")
    print(f"Saved worst selected candidate video: {video_path}")


if __name__ == "__main__":
    main()
