import pytest


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
