"""Pretrained Faster R-CNN and GRU inference on MOT17 videos."""

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

from pa2.gru_model import GRUSequenceModel
from pa2.mot17_dataset import MOT17FrameDataset
from pa2.tracking import GRUTracker


DETECTION_SCORE_THRESHOLD = 0.7
IOU_THRESHOLD = 0.3
MAX_MISSED_FRAMES = 3
HISTORY_LENGTH = 15
DEFAULT_MAX_FRAMES = 60


def detect_people(detector, image: torch.Tensor, device: torch.device):
    """Run Faster R-CNN and return confident person boxes as (x, y, w, h)."""
    output = detector([image.to(device)])[0]
    people = (output["labels"] == 1) & (
        output["scores"] >= DETECTION_SCORE_THRESHOLD
    )
    xyxy = output["boxes"][people].detach().cpu()
    return [
        (round(x1), round(y1), max(1, round(x2 - x1)), max(1, round(y2 - y1)))
        for x1, y1, x2, y2 in xyxy.tolist()
    ]


def _sequence_name(video_id: str | int) -> str:
    value = str(video_id).strip()
    if value.startswith("MOT17-"):
        return value
    return f"MOT17-{int(value):02d}-SDP"


def run_mot17_gru(
    video_id: str | int,
    *,
    root: str | Path = "data",
    max_frames: int | None = DEFAULT_MAX_FRAMES,
):
    """Run the pretrained person detector and GRU tracker on one MOT17 SDP video.

    Returns a Matplotlib animation. By default it processes the first 60 frames
    for a quick preview; pass ``max_frames=None`` to process the whole sequence.
    """
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive or None")

    repo_root = Path(__file__).resolve().parents[2]
    data_root = Path(root)
    if not data_root.is_absolute():
        data_root = repo_root / data_root
    sequence_name = _sequence_name(video_id)
    dataset = MOT17FrameDataset(root=data_root, split="train")
    sequence_index = next(
        (
            index
            for index, sequence in enumerate(dataset.sequences)
            if sequence.name == sequence_name
        ),
        None,
    )
    if sequence_index is None:
        raise FileNotFoundError(f"Could not find {sequence_name} under {data_root}")

    sequence = dataset.sequences[sequence_index]
    dataset_indices = [
        index
        for index, (source_index, _) in enumerate(dataset.index)
        if source_index == sequence_index
    ]
    if max_frames is not None:
        dataset_indices = dataset_indices[:max_frames]
    frame_numbers = [
        sequence.frame_numbers[dataset.index[index][1]] for index in dataset_indices
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    ).to(device).eval()
    model = GRUSequenceModel(input_size=4, hidden_size=32, output_size=4).to(device)
    checkpoint = repo_root / "weights" / "mot17_gru.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    tracker = GRUTracker(
        model,
        device,
        iou_threshold=IOU_THRESHOLD,
        max_missed_frames=MAX_MISSED_FRAMES,
        history_length=HISTORY_LENGTH,
    )
    frames, detections_per_frame, tracks_per_frame = [], [], []
    with torch.inference_mode():
        for dataset_index in tqdm(dataset_indices, desc=sequence_name, unit="frame"):
            image, _, _ = dataset[dataset_index]
            detections = detect_people(detector, image, device)
            result = tracker.update(detections, (image.shape[1], image.shape[2]))
            frames.append(image.permute(1, 2, 0).mul(255).byte().cpu().numpy())
            detections_per_frame.append(detections)
            tracks_per_frame.append(result.predictions)

    figure, axis = plt.subplots(figsize=(10, 5.6))
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.92)
    image_artist = axis.imshow(frames[0])
    axis.set_axis_off()
    box_artists, labels = [], []

    def update(frame_index):
        image_artist.set_data(frames[frame_index])
        for artist in box_artists + labels:
            artist.remove()
        box_artists.clear()
        labels.clear()

        for x, y, width, height in detections_per_frame[frame_index]:
            rectangle = patches.Rectangle(
                (x, y), width, height, fill=False, edgecolor="yellow", linestyle="--"
            )
            axis.add_patch(rectangle)
            box_artists.append(rectangle)
        for box, track_id in tracks_per_frame[frame_index]:
            x, y, width, height = box
            color = plt.cm.tab20(track_id % 20)
            rectangle = patches.Rectangle(
                (x, y), width, height, fill=False, edgecolor=color, linewidth=2
            )
            label = axis.text(x, y, str(track_id), color=color, fontsize=9)
            axis.add_patch(rectangle)
            box_artists.append(rectangle)
            labels.append(label)
        axis.set_title(f"{sequence_name}, frame {frame_numbers[frame_index]}")
        return [image_artist, *box_artists, *labels]

    return animation.FuncAnimation(
        figure, update, frames=len(frames), interval=50, blit=False
    )
