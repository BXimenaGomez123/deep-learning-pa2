"""Measure the trained GRU's effective memory using hidden-state gradients."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from pa2.datasets import make_mot17_track_splits
from pa2.gru_model import GRUSequenceModel
from pa2.mot17_dataset import MOT17TrackDataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
SAMPLE_COUNT = 256
SEED = 0
SCALE = torch.tensor([1920.0, 1080.0, 1920.0, 1080.0])
HORIZON_THRESHOLD = 0.05


def sample_training_windows(sample_count: int, seed: int):
    """Sample track windows evenly across the five training source sequences."""
    subset = make_mot17_track_splits(root=ROOT / "data", T=16, stride=4)["train"]
    dataset = cast(MOT17TrackDataset, subset.dataset)
    source_numbers = ("02", "04", "09", "10", "11")
    groups = {
        number: [
            index
            for index in subset.indices
            if dataset.sample_sequences[index].split("-")[1] == number
        ]
        for number in source_numbers
    }
    groups = {number: indices for number, indices in groups.items() if indices}
    if not groups:
        raise ValueError("No MOT17 training windows were found")

    rng = np.random.default_rng(seed)
    selected = []
    base_count, remainder = divmod(sample_count, len(groups))
    for group_index, indices in enumerate(groups.values()):
        count = min(base_count + (group_index < remainder), len(indices))
        selected.extend(rng.choice(indices, size=count, replace=False).tolist())

    if len(selected) < sample_count:
        selected_set = set(selected)
        remaining = [
            index
            for indices in groups.values()
            for index in indices
            if index not in selected_set
        ]
        extra_count = min(sample_count - len(selected), len(remaining))
        selected.extend(rng.choice(remaining, size=extra_count, replace=False).tolist())

    return dataset, selected


def measure_gradient_profile(model, dataset, window_indices, device):
    """Compute ||d L_t / d h_(t-k)|| for valid one-step prediction losses."""
    raw_by_lag: list[list[float]] = [[] for _ in range(15)]
    relative_by_lag: list[list[float]] = [[] for _ in range(15)]
    target_count_by_lag = [0] * 15
    scale = SCALE.to(device)

    model.eval()
    for window_index in tqdm(window_indices, desc="Gradient windows", unit="window"):
        boxes, valid_mask = dataset[window_index]
        inputs = (boxes.to(device) / scale).unsqueeze(0)
        valid = valid_mask.to(device) > 0

        # Calling the same model once per timestep exposes each intermediate
        # hidden tensor in autograd while preserving the checkpoint's weights.
        hidden = None
        states = []
        predictions = []
        for timestep in range(inputs.shape[1] - 1):
            recurrent_output, _ = model.gru(
                inputs[:, timestep : timestep + 1], hidden
            )
            # Use one explicit tensor as both h_t and the state passed onward.
            # This makes each intermediate h_t a node in the autograd graph.
            hidden = recurrent_output[:, -1, :].unsqueeze(0)
            states.append(hidden)
            predictions.append(
                model.output_projection(model.dropout(hidden[-1]))
            )

        for target_t in range(len(predictions)):
            if not (valid[target_t] and valid[target_t + 1]):
                continue
            loss = (predictions[target_t][0] - inputs[0, target_t + 1]).square().mean()
            gradients = torch.autograd.grad(
                loss,
                states[: target_t + 1],
                retain_graph=True,
            )
            lag_norms = {
                target_t - state_t: float(gradient.detach().norm().cpu())
                for state_t, gradient in enumerate(gradients)
            }
            current_norm = lag_norms[0]
            for lag, norm in lag_norms.items():
                raw_by_lag[lag].append(norm)
                relative_by_lag[lag].append(norm / max(current_norm, 1e-12))
                target_count_by_lag[lag] += 1

    return raw_by_lag, relative_by_lag, target_count_by_lag


def summarize(values):
    if not values:
        return float("nan"), float("nan"), float("nan")
    return tuple(float(value) for value in np.percentile(values, [25, 50, 75]))


def effective_horizon(relative_medians, threshold):
    """First lag after which the median relative gradient remains below threshold."""
    for lag in range(len(relative_medians)):
        if all(value <= threshold for value in relative_medians[lag:]):
            return lag
    return None


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = ROOT / "weights" / "mot17_gru.pt"
    model = GRUSequenceModel(input_size=4, hidden_size=32, output_size=4).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    dataset, window_indices = sample_training_windows(SAMPLE_COUNT, SEED)
    print(
        f"Measuring hidden-state gradients on {len(window_indices)} training windows "
        f"using {checkpoint.name} ({device}).",
        flush=True,
    )
    raw, relative, target_counts = measure_gradient_profile(
        model, dataset, window_indices, device
    )

    raw_stats = [summarize(values) for values in raw]
    relative_stats = [summarize(values) for values in relative]
    relative_medians = [stats[1] for stats in relative_stats]
    horizon = effective_horizon(relative_medians, HORIZON_THRESHOLD)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "gru_analytic_memory.csv"
    with csv_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "lag",
                "target_count",
                "gradient_norm_p25",
                "gradient_norm_median",
                "gradient_norm_p75",
                "relative_norm_p25",
                "relative_norm_median",
                "relative_norm_p75",
            ]
        )
        for lag, (raw_values, relative_values) in enumerate(
            zip(raw_stats, relative_stats)
        ):
            writer.writerow(
                [lag, target_counts[lag], *raw_values, *relative_values]
            )

    lags = np.arange(len(raw_stats))
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    raw_low, raw_median, raw_high = np.array(raw_stats).T
    rel_low, rel_median, rel_high = np.array(relative_stats).T

    axes[0].plot(lags, raw_median, "o-", color="tab:blue")
    axes[0].fill_between(lags, raw_low, raw_high, color="tab:blue", alpha=0.2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Median $\|\partial L_t/\partial h_{t-k}\|_2$")
    axes[0].set_title("Analytical memory horizon of the trained MOT17 GRU")
    axes[0].grid(alpha=0.3)

    axes[1].plot(lags, rel_median, "o-", color="tab:orange")
    axes[1].fill_between(lags, rel_low, rel_high, color="tab:orange", alpha=0.2)
    axes[1].axhline(HORIZON_THRESHOLD, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Lag k (input timesteps into the past)")
    axes[1].set_ylabel("Gradient norm / norm at k=0")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()

    plot_path = OUTPUT_DIR / "gru_analytic_memory.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    if horizon is None:
        print(
            f"Median relative gradient did not remain below {HORIZON_THRESHOLD:.0%} "
            f"within the measured lag range 0–{len(lags) - 1}.",
            flush=True,
        )
    else:
        print(
            f"Effective horizon at the {HORIZON_THRESHOLD:.0%} threshold: "
            f"{horizon} timestep(s).",
            flush=True,
        )
    print(f"Saved plot: {plot_path}")
    print(f"Saved gradient data: {csv_path}")


if __name__ == "__main__":
    main()
