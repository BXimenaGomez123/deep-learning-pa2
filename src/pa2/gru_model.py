"""A generic GRU sequence-to-sequence model."""

from __future__ import annotations

import torch
from torch import nn


class GRUSequenceModel(nn.Module):
    """Map each timestep in an input sequence to an output feature vector.

    This module is intentionally independent of datasets and feature meaning.
    For box-motion prediction, for example, inputs and outputs can both be
    four-dimensional box representations. The caller controls how sequences
    and targets are prepared and aligned.

    With the default ``batch_first=True``, inputs have shape
    ``[batch, time, input_size]`` and predictions have shape
    ``[batch, time, output_size]``. The returned hidden state has shape
    ``[num_layers * directions, batch, hidden_size]`` and can be passed back
    to ``forward`` to continue a sequence incrementally.

    For one-step-ahead prediction, train output at timestep ``t`` against the
    target at timestep ``t + 1``. At inference, the final output predicts the
    step after the final input timestep.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        directions = 2 if bidirectional else 1
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_size * directions, output_size)

    def forward(
        self,
        sequence: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-step predictions and the final GRU hidden state."""
        if sequence.ndim != 3:
            raise ValueError(
                "sequence must have three dimensions "
                "(batch, time, features) or (time, batch, features)"
            )
        if sequence.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} input features, got {sequence.shape[-1]}"
            )

        recurrent_output, final_hidden = self.gru(sequence, hidden)
        predictions = self.output_projection(self.dropout(recurrent_output))
        return predictions, final_hidden
