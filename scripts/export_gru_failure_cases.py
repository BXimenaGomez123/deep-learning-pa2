"""Export focused videos for selected failed GRU identities on MOT17-04-SDP."""

from pathlib import Path
from typing import cast

import torch
from tqdm import tqdm
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

from find_gru_failure_case import IOU_EVALUATION_THRESHOLD, ROOT, save_identity_video
from pa2.datasets import make_mot17_frame_splits
from pa2.gru_model import GRUSequenceModel
from pa2.metrics import Box
from pa2.mot17_dataset import MOT17FrameDataset
from pa2.tracking import GRUTracker, TrackingResult
from pa2.tracking_analysis import analyze_identity_errors
from predict_mot17_tracks import HISTORY_LENGTH, IOU_THRESHOLD, MAX_MISSED_FRAMES, detect_people


SEQUENCE_NAME = "MOT17-04-SDP"
TARGET_IDENTITIES = (86, 93, 74)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUSequenceModel(input_size=5, hidden_size=32, output_size=4).to(device)
    model.load_state_dict(
        torch.load(
            ROOT / "weights" / "mot17_gru_corrupt.pt",
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    subset = make_mot17_frame_splits(root=ROOT / "data")["train"]
    dataset = cast(MOT17FrameDataset, subset.dataset)
    sequence_index = next(
        (
            index
            for index, sequence in enumerate(dataset.sequences)
            if sequence.name == SEQUENCE_NAME
        ),
        None,
    )
    if sequence_index is None:
        raise FileNotFoundError(f"Could not find {SEQUENCE_NAME} in the MOT17 data")

    sequence = dataset.sequences[sequence_index]
    dataset_indices = [
        index
        for index, (source_index, _) in enumerate(dataset.index)
        if source_index == sequence_index
    ]
    cache_path = ROOT / "outputs" / "mot17_04_sdp_frcnn_detections.pt"
    if cache_path.is_file():
        detections_cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if not isinstance(detections_cache, list):
            raise ValueError(f"Invalid detection cache at {cache_path}")
    else:
        detections_cache = []

    if len(detections_cache) < len(dataset_indices):
        detector = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        ).to(device).eval()
        for position in tqdm(
            range(len(detections_cache), len(dataset_indices)),
            desc=f"Detecting {SEQUENCE_NAME}",
            unit="frame",
        ):
            image, _, _ = dataset[dataset_indices[position]]
            detections_cache.append(detect_people(detector, image, device))
            if (position + 1) % 20 == 0 or position + 1 == len(dataset_indices):
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = cache_path.with_suffix(".tmp")
                torch.save(detections_cache, temporary_path)
                temporary_path.replace(cache_path)

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
        include_observation_mask=True,
    )

    with torch.inference_mode():
        sample_image, _, _ = dataset[dataset_indices[0]]
        image_size = (sample_image.shape[1], sample_image.shape[2])
        for position, dataset_index in enumerate(
            tqdm(dataset_indices, desc=f"Tracking {SEQUENCE_NAME}", unit="frame")
        ):
            image, boxes, identities = dataset[dataset_index]
            detections = detections_cache[position]
            result = tracker.update(detections, image_size)
            tracking_results.append(result)
            gt_boxes.append(
                [(int(box[0]), int(box[1]), int(box[2]), int(box[3])) for box in boxes]
            )
            gt_ids.append([int(identity) for identity in identities.tolist()])
            pred_boxes.append([box for box, _ in result.detections])
            pred_ids.append([track_id for _, track_id in result.detections])
            image_sizes.append(image_size)

    identity_errors = analyze_identity_errors(
        gt_boxes,
        gt_ids,
        pred_boxes,
        pred_ids,
        frame_numbers,
        image_sizes,
        iou_threshold=IOU_EVALUATION_THRESHOLD,
    )
    sequence_data = {
        "dataset_indices": dataset_indices,
        "frame_numbers": frame_numbers,
        "gt_boxes": gt_boxes,
        "gt_ids": gt_ids,
        "tracking_results": tracking_results,
        "identities": identity_errors,
    }

    for identity_id in TARGET_IDENTITIES:
        identity = next(
            (item for item in identity_errors if item.identity_id == identity_id),
            None,
        )
        if identity is None:
            print(f"GT identity {identity_id} was not present; skipping.")
            continue
        print(
            f"GT {identity_id}: {identity.id_switches} switches + "
            f"{identity.fragmentations} fragmentations; "
            f"{identity.misses} visible misses."
        )
        save_identity_video(
            dataset,
            sequence_data,
            SEQUENCE_NAME,
            identity,
            "high-switch-and-fragmentation case",
        )


if __name__ == "__main__":
    main()
