"""Box association and lightweight MOT trackers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pa2.gru_model import GRUSequenceModel
from pa2.metrics import Box, box_iou


def iou_matrix(boxes_a: list[Box], boxes_b: list[Box]) -> np.ndarray:
    """Return pairwise IoU for boxes in (x, y, width, height) format."""
    similarities = np.zeros((len(boxes_a), len(boxes_b)), dtype=float)
    for row, box_a in enumerate(boxes_a):
        for column, box_b in enumerate(boxes_b):
            similarities[row, column] = box_iou(box_a, box_b)
    return similarities


def gated_hungarian(
    similarities: np.ndarray, threshold: float
) -> list[tuple[int, int]]:
    """Match valid pairs first, then maximize their total similarity."""
    if similarities.ndim != 2:
        raise ValueError("similarities must be a two-dimensional matrix")
    if similarities.size == 0:
        return []

    valid = similarities >= threshold
    cardinality_bonus = min(similarities.shape) + 1
    rewards = np.where(valid, cardinality_bonus + similarities, 0.0)
    rows, columns = linear_sum_assignment(-rewards)
    return [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if valid[row, column]
    ]


def match_boxes_by_iou(
    boxes_a: list[Box], boxes_b: list[Box], threshold: float
) -> list[tuple[int, int]]:
    """Return index pairs using one-to-one gated Hungarian IoU matching."""
    return gated_hungarian(iou_matrix(boxes_a, boxes_b), threshold)


@dataclass
class TrackingResult:
    """Per-frame observed detections, predicted boxes, and tracker status."""

    detections: list[tuple[Box, int]]
    predictions: list[tuple[Box, int]]
    matches: int
    active_tracks: int


@dataclass
class _Track:
    boxes: list[Box]
    observed: list[bool]
    missed_frames: int = 0


class GRUTracker:
    """Associate detections with next-box predictions from a trained GRU."""

    def __init__(
        self,
        model: GRUSequenceModel,
        device: torch.device,
        scale: tuple[float, float, float, float] = (1920.0, 1080.0, 1920.0, 1080.0),
        iou_threshold: float = 0.3,
        max_missed_frames: int = 3,
        history_length: int = 15,
        include_observation_mask: bool = False,
    ):
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        self.model = model
        self.device = device
        self.scale = torch.tensor(scale, dtype=torch.float32, device=device)
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.history_length = history_length
        self.include_observation_mask = include_observation_mask
        self.tracks: dict[int, _Track] = {}
        self.next_track_id = 1

    def _predict_next_box(
        self, track: _Track, image_size: tuple[int, int]
    ) -> Box:
        boxes = torch.tensor(
            track.boxes[-self.history_length :], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        history = boxes / self.scale
        if self.include_observation_mask:
            observed_values = track.observed[-self.history_length :]
            observed = torch.tensor(
                observed_values,
                dtype=torch.float32,
                device=self.device,
            ).view(1, -1, 1)
            history = history * observed
            history = torch.cat((history, observed), dim=-1)
        predictions, _ = self.model(history)
        x, y, width, height = (predictions[0, -1] * self.scale).tolist()

        image_height, image_width = image_size
        x = min(max(round(x), 0), image_width - 1)
        y = min(max(round(y), 0), image_height - 1)
        width = min(max(round(width), 1), image_width - x)
        height = min(max(round(height), 1), image_height - y)
        return x, y, width, height

    def update(
        self, detections: list[Box], image_size: tuple[int, int]
    ) -> TrackingResult:
        """Match this frame's detections and advance active track histories."""
        track_ids = list(self.tracks)
        predicted_boxes = [
            self._predict_next_box(self.tracks[track_id], image_size)
            for track_id in track_ids
        ]
        matches = match_boxes_by_iou(
            predicted_boxes, detections, self.iou_threshold
        )
        detection_to_track = {
            detection_index: track_ids[track_index]
            for track_index, detection_index in matches
        }
        track_to_detection = {
            track_id: detection_index
            for detection_index, track_id in detection_to_track.items()
        }

        active_tracks: dict[int, _Track] = {}
        visible_predictions = list(zip(predicted_boxes, track_ids))
        for track_id, predicted_box in zip(track_ids, predicted_boxes):
            track = self.tracks[track_id]
            if track_id in track_to_detection:
                track.boxes.append(detections[track_to_detection[track_id]])
                track.observed.append(True)
                track.missed_frames = 0
                active_tracks[track_id] = track
            else:
                track.boxes.append(predicted_box)
                track.observed.append(False)
                track.missed_frames += 1
                if track.missed_frames <= self.max_missed_frames:
                    active_tracks[track_id] = track

        for detection_index, box in enumerate(detections):
            if detection_index in detection_to_track:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            active_tracks[track_id] = _Track(boxes=[box], observed=[True])
            detection_to_track[detection_index] = track_id
            visible_predictions.append((box, track_id))

        self.tracks = active_tracks
        observed = [
            (box, detection_to_track[index])
            for index, box in enumerate(detections)
        ]
        return TrackingResult(
            detections=observed,
            predictions=visible_predictions,
            matches=len(matches),
            active_tracks=len(active_tracks),
        )


@dataclass
class _IoUTrack:
    box: Box
    last_frame: int
    missed_frames: int = 0


class IoUTracker:
    """Baseline tracker matching detections to boxes in the previous frame."""

    def __init__(self, iou_threshold: float = 0.3, max_missed_frames: int = 3):
        if max_missed_frames < 1:
            raise ValueError("max_missed_frames must be at least one")
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.tracks: dict[int, _IoUTrack] = {}
        self.next_track_id = 1
        self.previous_frame: int | None = None

    def update(self, frame: int, detections: list[Box]) -> TrackingResult:
        """Assign IDs to detections using only immediately previous boxes."""
        if self.previous_frame is not None and frame != self.previous_frame + 1:
            raise ValueError("Process every consecutive frame, including empty ones")

        candidate_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.last_frame == frame - 1
        ]
        previous_boxes = [self.tracks[track_id].box for track_id in candidate_ids]
        matches = match_boxes_by_iou(
            previous_boxes, detections, self.iou_threshold
        )
        detection_to_track = {
            detection_index: candidate_ids[track_index]
            for track_index, detection_index in matches
        }

        observed: list[tuple[Box, int]] = []
        observed_ids = set()
        for detection_index, box in enumerate(detections):
            track_id = detection_to_track.get(detection_index)
            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
            self.tracks[track_id] = _IoUTrack(box=box, last_frame=frame)
            observed_ids.add(track_id)
            observed.append((box, track_id))

        for track_id in list(self.tracks):
            if track_id in observed_ids:
                continue
            self.tracks[track_id].missed_frames += 1
            if self.tracks[track_id].missed_frames >= self.max_missed_frames:
                del self.tracks[track_id]

        self.previous_frame = frame
        return TrackingResult(
            detections=observed,
            predictions=list(observed),
            matches=len(matches),
            active_tracks=len(self.tracks),
        )
