import pytest
import torch
from torch import float32

from pa2.gru_model import GRUSequenceModel
from utils import tensor_like


def test_gru_sequence_output_and_hidden_shapes():
    batch_size, sequence_length, input_size = 3, 6, 4
    model = GRUSequenceModel(
        input_size=input_size,
        hidden_size=16,
        output_size=4,
        num_layers=2,
    )

    predictions, hidden = model(torch.randn(batch_size, sequence_length, input_size))

    assert tensor_like(predictions, (batch_size, sequence_length, 4), float32)
    assert tensor_like(hidden, (2, batch_size, 16), float32)


def test_gru_hidden_state_can_continue_a_sequence():
    model = GRUSequenceModel(input_size=4, hidden_size=8, output_size=4)
    model.eval()
    sequence = torch.randn(2, 7, 4)

    full_predictions, _ = model(sequence)
    first_predictions, hidden = model(sequence[:, :4])
    continued_predictions, _ = model(sequence[:, 4:], hidden)

    assert torch.allclose(full_predictions[:, :4], first_predictions)
    assert torch.allclose(full_predictions[:, 4:], continued_predictions)


def test_gru_supports_non_batch_first_and_bidirectional_inputs():
    model = GRUSequenceModel(
        input_size=5,
        hidden_size=7,
        output_size=2,
        bidirectional=True,
        batch_first=False,
    )
    predictions, hidden = model(torch.randn(6, 3, 5))

    assert tensor_like(predictions, (6, 3, 2), float32)
    assert tensor_like(hidden, (2, 3, 7), float32)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_size": 0, "hidden_size": 8, "output_size": 4},
        {"input_size": 4, "hidden_size": 0, "output_size": 4},
        {"input_size": 4, "hidden_size": 8, "output_size": 0},
        {"input_size": 4, "hidden_size": 8, "output_size": 4, "num_layers": 0},
        {"input_size": 4, "hidden_size": 8, "output_size": 4, "dropout": 1.0},
    ],
)
def test_gru_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        GRUSequenceModel(**kwargs)


def test_gru_rejects_wrong_input_shape_or_feature_count():
    model = GRUSequenceModel(input_size=4, hidden_size=8, output_size=4)

    with pytest.raises(ValueError, match="three dimensions"):
        model(torch.randn(2, 4))
    with pytest.raises(ValueError, match="input features"):
        model(torch.randn(2, 5, 3))
