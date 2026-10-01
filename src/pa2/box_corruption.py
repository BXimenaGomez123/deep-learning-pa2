"""Corrupt ground-truth box tracks to simulate imperfect detections."""

from __future__ import annotations

import math

import torch


def corrupt_track_boxes(
    boxes: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    drop_fraction: float = 0.0,
    noise_std: float = 0.0,
    false_positive_count: int = 0,
    image_size: tuple[int, int] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop, jitter, and add false-positive boxes in a track sample.

    Args:
        boxes: Ground-truth boxes with shape ``[T, 4]`` in ``(x, y, w, h)``
            format, as returned by a track dataset.
        valid_mask: Binary mask with shape ``[T]`` indicating which input
            boxes are real. This follows the track dataset's ``valid`` output.
        drop_fraction: Fraction of real boxes to drop, rounded to the nearest
            whole box. Dropped entries get a zero box and a zero mask value.
        noise_std: Standard deviation of independent Gaussian noise added to
            each coordinate of every retained real box, in pixels.
        false_positive_count: Number of false positives to insert into
            originally invalid time slots. These slots are marked valid in the
            returned mask. If there are not enough invalid slots, raises
            ``ValueError`` rather than changing the sequence length.
        image_size: Optional ``(height, width)`` used to keep boxes inside the
            image. If omitted, the extent of the input boxes is used.
        generator: Optional PyTorch random generator for reproducible results.

    Returns:
        A new ``(boxes, mask)`` pair with the same shapes, dtypes, and devices
        as the inputs. The returned mask describes which boxes are present in
        the corrupted detections, including injected false positives.

    The inputs are not modified. False-positive box sizes are sampled from
    real boxes in the input track, and their positions are sampled uniformly
    within the provided or inferred image extent.
    """
    if not isinstance(boxes, torch.Tensor) or not isinstance(valid_mask, torch.Tensor):
        raise TypeError("boxes and valid_mask must be torch tensors")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must have shape [T, 4], got {tuple(boxes.shape)}")
    if valid_mask.shape != (boxes.shape[0],):
        raise ValueError(
            f"valid_mask must have shape [{boxes.shape[0]}], got {tuple(valid_mask.shape)}"
        )
    if boxes.device != valid_mask.device:
        raise ValueError("boxes and valid_mask must be on the same device")
    if not boxes.is_floating_point():
        raise TypeError("boxes must have a floating-point dtype")
    if not (math.isfinite(drop_fraction) and 0.0 <= drop_fraction <= 1.0):
        raise ValueError("drop_fraction must be finite and between 0 and 1")
    if not (math.isfinite(noise_std) and noise_std >= 0.0):
        raise ValueError("noise_std must be finite and non-negative")
    if isinstance(false_positive_count, bool) or not isinstance(false_positive_count, int):
        raise TypeError("false_positive_count must be an integer")
    if false_positive_count < 0:
        raise ValueError("false_positive_count must be non-negative")
    if image_size is not None:
        if len(image_size) != 2 or image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError("image_size must be a positive (height, width) pair")
    if not bool(torch.all((valid_mask == 0) | (valid_mask == 1))):
        raise ValueError("valid_mask must contain only 0 and 1")

    original_valid = valid_mask.to(torch.bool)
    valid_indices = torch.where(original_valid)[0]
    invalid_indices = torch.where(~original_valid)[0]
    if false_positive_count > len(invalid_indices):
        raise ValueError(
            f"Requested {false_positive_count} false positives, but only "
            f"{len(invalid_indices)} originally invalid slots are available"
        )
    if false_positive_count and len(valid_indices) == 0:
        raise ValueError("Cannot size false-positive boxes without a valid input box")

    corrupted_boxes = boxes.clone()
    corrupted_mask = valid_mask.clone()

    n_dropped = math.floor(len(valid_indices) * drop_fraction + 0.5)
    if n_dropped:
        permutation = torch.randperm(
            len(valid_indices), device=boxes.device, generator=generator
        )
        dropped_indices = valid_indices[permutation[:n_dropped]]
        corrupted_mask[dropped_indices] = 0

    retained_indices = torch.where(corrupted_mask.to(torch.bool))[0]
    if noise_std and len(retained_indices):
        noise = torch.randn(
            (len(retained_indices), 4),
            dtype=boxes.dtype,
            device=boxes.device,
            generator=generator,
        ) * noise_std
        corrupted_boxes[retained_indices] += noise
    if len(retained_indices):
        corrupted_boxes[retained_indices, 2:4].clamp_(min=1.0)

    # Invalid entries from the input are conventionally zero-filled; keep that
    # invariant for dropped entries too, before placing any false positives.
    corrupted_boxes[~corrupted_mask.to(torch.bool)] = 0

    if false_positive_count:
        height, width = image_size if image_size is not None else (None, None)
        source_boxes = boxes[valid_indices]
        source_widths = source_boxes[:, 2].clamp(min=1.0)
        source_heights = source_boxes[:, 3].clamp(min=1.0)

        source_choice = torch.randint(
            len(source_boxes),
            (false_positive_count,),
            device=boxes.device,
            generator=generator,
        )
        fp_widths = source_widths[source_choice]
        fp_heights = source_heights[source_choice]

        if width is None:
            width = max(1.0, float((source_boxes[:, 0] + source_boxes[:, 2]).max()))
        if height is None:
            height = max(1.0, float((source_boxes[:, 1] + source_boxes[:, 3]).max()))

        width = float(width)
        height = float(height)
        fp_widths = fp_widths.clamp(max=width)
        fp_heights = fp_heights.clamp(max=height)
        x = torch.rand(
            false_positive_count, dtype=boxes.dtype, device=boxes.device, generator=generator
        ) * (width - fp_widths).clamp(min=0)
        y = torch.rand(
            false_positive_count, dtype=boxes.dtype, device=boxes.device, generator=generator
        ) * (height - fp_heights).clamp(min=0)

        fp_boxes = torch.stack((x, y, fp_widths, fp_heights), dim=1)
        fp_slots = invalid_indices[
            torch.randperm(
                len(invalid_indices), device=boxes.device, generator=generator
            )[:false_positive_count]
        ]
        corrupted_boxes[fp_slots] = fp_boxes
        corrupted_mask[fp_slots] = 1

    if image_size is not None and len(retained_indices):
        height, width = image_size
        retained_boxes = corrupted_boxes[retained_indices]
        retained_boxes[:, 2].clamp_(min=1.0, max=float(width))
        retained_boxes[:, 3].clamp_(min=1.0, max=float(height))
        retained_boxes[:, 0].clamp_(min=0.0)
        retained_boxes[:, 1].clamp_(min=0.0)
        retained_boxes[:, 0] = torch.minimum(
            retained_boxes[:, 0], float(width) - retained_boxes[:, 2]
        )
        retained_boxes[:, 1] = torch.minimum(
            retained_boxes[:, 1], float(height) - retained_boxes[:, 3]
        )
        corrupted_boxes[retained_indices] = retained_boxes

    return corrupted_boxes, corrupted_mask
