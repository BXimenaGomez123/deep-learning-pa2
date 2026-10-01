import pytest
import numpy as np
from PIL import Image


def make_fake_mot17(
    root,
    sequence_numbers=("02", "04", "05", "09", "10", "11", "13"),
    n_frames=3,
    image_size=(8, 8),
    include_distractor=False,
):
    """Create a small MOT17-style image/label tree for dataset tests."""
    height, width = image_size
    for sequence_number in sequence_numbers:
        name = f"MOT17-{sequence_number}-SDP"
        sequence_dir = root / "MOT17" / "train" / name
        image_dir = sequence_dir / "img1"
        image_dir.mkdir(parents=True)
        (sequence_dir / "seqinfo.ini").write_text(
            f"[Sequence]\nname={name}\nimDir=img1\nimExt=.jpg\n"
        )

        gt_dir = root / "MOT17Labels" / "train" / name / "gt"
        gt_dir.mkdir(parents=True)
        annotations = []
        for frame in range(1, n_frames + 1):
            image = np.full((height, width, 3), frame * 40, dtype=np.uint8)
            Image.fromarray(image).save(image_dir / f"{frame:06d}.jpg")
            annotations.append(f"{frame},7,1,2,3,4,1,1,1")
            if include_distractor:
                annotations.append(f"{frame},99,2,2,3,4,1,7,1")
        (gt_dir / "gt.txt").write_text("\n".join(annotations) + "\n")
    return root


def tensor_like(tensor, shape=None, dtype=None):
    """Checks if a tensor matches a shape spec and dtype.

    shape: tuple of ints; use None as a wildcard for any size in that dim,
           e.g. (None, 3, 224, 224) matches any batch size.
    dtype: e.g. torch.float32 (skipped if None).
    """

    __tracebackhide__ = True  # pytest shows the caller's line, not this one

    if shape is not None:
        actual = tuple(tensor.shape)

        # check dim count
        if len(actual) != len(shape):
            pytest.fail(f'Expected {len(shape)} dims, got {len(actual)}: {actual}')

        # check each dim
        for i, (a, e) in enumerate(zip(actual, shape)):
            if e is not None and a != e:
                pytest.fail(
                    f'Shape mismatch at dim {i}: expected {e}, got {a} '
                    f'(full shape: {actual}, expected: {shape})'
                )

    # check dtype
    if dtype is not None and tensor.dtype != dtype:
        pytest.fail(f'Expected dtype {dtype}, got {tensor.dtype}')

    return True
