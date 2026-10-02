"""Identity-level switch and fragmentation analysis for MOT predictions."""

from __future__ import annotations

from dataclasses import dataclass

from pa2.metrics import Box, box_iou, match_frame


@dataclass
class IdentityError:
    """Tracking errors and diagnostic context for one GT identity."""

    identity_id: int
    first_frame: int
    last_frame: int
    observed_frames: int
    matched_frames: int
    misses: int
    id_switches: int
    fragmentations: int
    longest_annotation_gap: int
    failure_frames: list[int]
    long_gap_events: int
    crossing_events: int
    boundary_events: int
    frame_indices: list[int]
    matched_prediction_ids: list[int | None]

    @property
    def error_score(self) -> int:
        """Requested ranking metric: ID switches plus track fragmentations."""
        return self.id_switches + self.fragmentations


def analyze_identity_errors(
    gt_boxes: list[list[Box]],
    gt_ids: list[list[int]],
    pred_boxes: list[list[Box]],
    pred_ids: list[list[int]],
    frame_numbers: list[int],
    image_sizes: list[tuple[int, int]],
    iou_threshold: float = 0.5,
    long_gap_threshold: int = 10,
) -> list[IdentityError]:
    """Count identity switches/fragments and flag likely failure contexts.

    Fragmentation is a missed run bounded by matched observations while the GT
    identity is annotated. Missing GT annotations are not themselves treated as
    tracker misses. Long GT annotation absences are retained only as an
    occlusion proxy; they can also represent leaving and re-entering the scene.
    """
    frame_count = len(gt_boxes)
    if not all(
        len(values) == frame_count
        for values in (gt_ids, pred_boxes, pred_ids, frame_numbers, image_sizes)
    ):
        raise ValueError("All per-frame inputs must have the same length")

    observations: dict[int, list[tuple[int, Box, int | None]]] = {}
    for frame_index, (boxes, ids, boxes_pred, ids_pred) in enumerate(
        zip(gt_boxes, gt_ids, pred_boxes, pred_ids)
    ):
        match_by_gt = dict(
            match_frame(
                boxes,
                ids,
                boxes_pred,
                ids_pred,
                iou_thresh=iou_threshold,
            )
        )
        for box, identity in zip(boxes, ids):
            observations.setdefault(identity, []).append(
                (frame_index, box, match_by_gt.get(identity))
            )

    results = []
    for identity, identity_observations in observations.items():
        matched_ids = [item[2] for item in identity_observations]
        switches = 0
        fragments = 0
        misses = 0
        failure_indices: set[int] = set()
        previous_matched_id = None
        has_matched = False
        pending_miss = False

        for (frame_index, _, predicted_id) in identity_observations:
            if predicted_id is None:
                misses += 1
                if has_matched:
                    pending_miss = True
                continue

            if previous_matched_id is not None and predicted_id != previous_matched_id:
                switches += 1
                failure_indices.add(frame_index)
            if has_matched and pending_miss:
                fragments += 1
                failure_indices.add(frame_index)
            previous_matched_id = predicted_id
            has_matched = True
            pending_miss = False

        frame_indices = [item[0] for item in identity_observations]
        identity_frame_numbers = [frame_numbers[index] for index in frame_indices]
        gaps = [
            later - earlier - 1
            for earlier, later in zip(identity_frame_numbers, identity_frame_numbers[1:])
        ]
        longest_gap = max(gaps, default=0)

        long_gap_returns = {
            frame_indices[offset + 1]
            for offset, gap in enumerate(gaps)
            if gap >= long_gap_threshold
        }
        long_gap_events = sum(
            any(abs(frame_numbers[failure] - frame_numbers[return_index]) <= 5
                for failure in failure_indices)
            for return_index in long_gap_returns
        )

        crossing_events = 0
        boundary_events = 0
        first_position, last_position = frame_indices[0], frame_indices[-1]
        span = max(1, last_position - first_position)
        for failure_index in failure_indices:
            observation = next(
                (item for item in identity_observations if item[0] == failure_index), None
            )
            if observation is None:
                continue
            _, target_box, _ = observation
            overlaps_another_person = any(
                other_id != identity and box_iou(target_box, other_box) > 0.05
                for other_box, other_id in zip(
                    gt_boxes[failure_index], gt_ids[failure_index]
                )
            )
            if overlaps_another_person:
                crossing_events += 1

            height, width = image_sizes[failure_index]
            x, y, box_width, box_height = target_box
            near_image_edge = (
                x <= 0.02 * width
                or y <= 0.02 * height
                or x + box_width >= 0.98 * width
                or y + box_height >= 0.98 * height
            )
            near_lifetime_end = (
                failure_index - first_position <= 0.1 * span
                or last_position - failure_index <= 0.1 * span
            )
            if near_image_edge or near_lifetime_end:
                boundary_events += 1

        results.append(
            IdentityError(
                identity_id=identity,
                first_frame=identity_frame_numbers[0],
                last_frame=identity_frame_numbers[-1],
                observed_frames=len(identity_observations),
                matched_frames=sum(predicted_id is not None for predicted_id in matched_ids),
                misses=misses,
                id_switches=switches,
                fragmentations=fragments,
                longest_annotation_gap=longest_gap,
                failure_frames=[frame_numbers[index] for index in sorted(failure_indices)],
                long_gap_events=long_gap_events,
                crossing_events=crossing_events,
                boundary_events=boundary_events,
                frame_indices=frame_indices,
                matched_prediction_ids=matched_ids,
            )
        )

    return sorted(
        results,
        key=lambda result: (
            result.error_score,
            result.id_switches,
            result.fragmentations,
            result.misses,
        ),
        reverse=True,
    )
