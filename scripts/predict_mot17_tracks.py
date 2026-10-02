"""Track MOT17 pedestrians with Faster R-CNN detections and the trained GRU."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from tqdm import tqdm
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

from pa2.datasets import make_mot17_frame_splits
from pa2.gru_model import GRUSequenceModel
from pa2.metrics import Box, match_frame
from pa2.mot17_dataset import MOT17FrameDataset


SEQUENCE_NAME = "MOT17-13-SDP"
MAX_FRAMES = 60
DETECTION_SCORE_THRESHOLD = 0.7
IOU_THRESHOLD = 0.3
MAX_MISSED_FRAMES = 3
HISTORY_LENGTH = 15


@dataclass
class Track:
    """A track's observed/predicted box history and missed-frame count."""

    boxes: list[Box]
    missed_frames: int = 0


def match_predictions_to_detections(
    predicted_boxes: list[Box],
    track_ids: list[int],
    detected_boxes: list[Box],
    iou_threshold: float = IOU_THRESHOLD,
) -> list[tuple[int, int]]:
    """Return (track ID, detection index) pairs matched by Hungarian IoU."""
    detection_indices = list(range(len(detected_boxes)))
    return match_frame(
        predicted_boxes,
        track_ids,
        detected_boxes,
        detection_indices,
        iou_thresh=iou_threshold,
    )


def detect_people(detector, image: torch.Tensor, device: torch.device) -> list[Box]:
    """Run Faster R-CNN and return confident person boxes in (x, y, w, h)."""
    output = detector([image.to(device)])[0]
    people = (output["labels"] == 1) & (
        output["scores"] >= DETECTION_SCORE_THRESHOLD
    )
    xyxy = output["boxes"][people].detach().cpu()
    boxes = []
    for x1, y1, x2, y2 in xyxy.tolist():
        boxes.append(
            (round(x1), round(y1), max(1, round(x2 - x1)), max(1, round(y2 - y1)))
        )
    return boxes


def predict_next_box(
    model: GRUSequenceModel,
    track: Track,
    scale: torch.Tensor,
    image_size: tuple[int, int],
    device: torch.device,
) -> Box:
    """Predict a track's next pixel-space box and clip it to the image."""
    history = torch.tensor(
        track.boxes[-HISTORY_LENGTH:], dtype=torch.float32, device=device
    ).unsqueeze(0)
    prediction, _ = model(history / scale)
    x, y, width, height = (prediction[0, -1] * scale).tolist()

    image_height, image_width = image_size
    x = min(max(round(x), 0), image_width - 1)
    y = min(max(round(y), 0), image_height - 1)
    width = min(max(round(width), 1), image_width - x)
    height = min(max(round(height), 1), image_height - y)
    return x, y, width, height


def load_test_sequence(root: str = "data"):
    """Read the held-out MOT17-13-SDP frames through the shared split factory."""
    test_split = make_mot17_frame_splits(root)["test"]
    dataset = test_split.dataset
    if not isinstance(dataset, MOT17FrameDataset):
        raise TypeError("Expected the MOT17 frame split to wrap MOT17FrameDataset")

    frame_indices = [
        index
        for index in test_split.indices
        if dataset.sequences[dataset.index[index][0]].name == SEQUENCE_NAME
    ][:MAX_FRAMES]
    if not frame_indices:
        raise FileNotFoundError(
            f"No frames for {SEQUENCE_NAME} in the MOT17 test split under {root}"
        )
    return dataset, frame_indices


def show_tracking(frames, detections_per_frame, tracks_per_frame):
    """Animate detector boxes and the GRU's predicted, ID-colored tracks."""
    figure, axis = plt.subplots()
    image = axis.imshow(frames[0])
    axis.set_axis_off()
    box_patches = []
    labels = []

    def update(frame_index):
        image.set_data(frames[frame_index])
        for item in box_patches + labels:
            item.remove()
        box_patches.clear()
        labels.clear()

        # Detector boxes are dashed yellow; GRU track predictions are solid.
        for x, y, width, height in detections_per_frame[frame_index]:
            rect = patches.Rectangle(
                (x, y), width, height, fill=False, edgecolor="yellow", linestyle="--"
            )
            axis.add_patch(rect)
            box_patches.append(rect)

        for box, track_id in tracks_per_frame[frame_index]:
            x, y, width, height = box
            color = plt.cm.tab20(track_id % 20)
            rect = patches.Rectangle(
                (x, y), width, height, fill=False, edgecolor=color, linewidth=2
            )
            label = axis.text(x, y, str(track_id), color=color, fontsize=9)
            axis.add_patch(rect)
            box_patches.append(rect)
            labels.append(label)
        axis.set_title(f"{SEQUENCE_NAME}, frame {frame_index + 1}")
        return [image, *box_patches, *labels]

    video = animation.FuncAnimation(
        figure, update, frames=len(frames), interval=50, blit=False
    )
    plt.show()
    return video


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    print(
        "Loading pretrained Faster R-CNN weights "
        "(the first run downloads about 170 MB)...",
        flush=True,
    )
    detector = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
    print("Faster R-CNN is ready.", flush=True)

    model = GRUSequenceModel(input_size=4, hidden_size=32, output_size=4).to(device)
    weights_path = Path(__file__).resolve().parents[1] / "weights" / "mot17_gru.pt"
    print(f"Loading trained GRU from {weights_path}...", flush=True)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    print(f"Loading frames from {SEQUENCE_NAME}...", flush=True)
    dataset, frame_indices = load_test_sequence()
    print(f"Prepared {len(frame_indices)} frames for tracking.", flush=True)
    scale = torch.tensor([1920.0, 1080.0, 1920.0, 1080.0], device=device)
    tracks: dict[int, Track] = {}
    next_track_id = 0
    frames = []
    detections_per_frame = []
    tracks_per_frame = []

    with torch.inference_mode():
        progress = tqdm(frame_indices, desc="Tracking", unit="frame")
        for frame_index in progress:
            progress.set_postfix_str("loading image and running detector", refresh=True)
            image, _, _ = dataset[frame_index]
            detections = detect_people(detector, image, device)
            image_height, image_width = image.shape[1:]

            track_ids = list(tracks)
            predictions = [
                predict_next_box(
                    model, tracks[track_id], scale, (image_height, image_width), device
                )
                for track_id in track_ids
            ]
            matches = match_predictions_to_detections(
                predictions, track_ids, detections
            )
            detection_for_track = {
                track_id: detection_index
                for track_id, detection_index in matches
            }
            matched_detection_indices = set(detection_for_track.values())

            active_tracks = {}
            visible_tracks = []
            for track_id, predicted_box in zip(track_ids, predictions):
                track = tracks[track_id]
                visible_tracks.append((predicted_box, track_id))
                if track_id in detection_for_track:
                    track.boxes.append(detections[detection_for_track[track_id]])
                    track.missed_frames = 0
                    active_tracks[track_id] = track
                else:
                    track.boxes.append(predicted_box)
                    track.missed_frames += 1
                    if track.missed_frames <= MAX_MISSED_FRAMES:
                        active_tracks[track_id] = track

            for detection_index, box in enumerate(detections):
                if detection_index in matched_detection_indices:
                    continue
                track_id = next_track_id
                next_track_id += 1
                active_tracks[track_id] = Track(boxes=[box])
                visible_tracks.append((box, track_id))

            tracks = active_tracks
            frames.append(
                (image.permute(1, 2, 0).mul(255).byte().cpu().numpy())
            )
            detections_per_frame.append(detections)
            tracks_per_frame.append(visible_tracks)
            progress.set_postfix(
                detections=len(detections),
                matches=len(matches),
                active_tracks=len(tracks),
            )

    print("Inference complete. Opening the tracking animation; close its window to exit.", flush=True)
    show_tracking(frames, detections_per_frame, tracks_per_frame)


if __name__ == "__main__":
    main()
