"""Train an observation-aware GRU with corrupted inputs and clean targets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pa2.box_corruption import corrupt_track_boxes
from pa2.datasets import make_mot17_track_splits
from pa2.gru_model import GRUSequenceModel


ROOT = Path(__file__).resolve().parents[1]
SEED = 0
IMAGE_SIZE = (1080, 1920)
SCALE = torch.tensor([1920.0, 1080.0, 1920.0, 1080.0])
BATCH_SIZE = 256


def corrupt_batch(boxes, valid_mask, generator):
    corrupted_boxes = []
    observed_masks = []
    for track_boxes, track_mask in zip(boxes, valid_mask):
        random_values = torch.rand(3, generator=generator)
        invalid_slots = int((track_mask == 0).sum())
        false_positive_count = int(random_values[2] < 0.2 and invalid_slots > 0)
        sample_boxes, sample_mask = corrupt_track_boxes(
            track_boxes,
            track_mask,
            drop_fraction=0.1 + 0.3 * float(random_values[0]),
            noise_std=5.0 + 15.0 * float(random_values[1]),
            false_positive_count=false_positive_count,
            image_size=IMAGE_SIZE,
            generator=generator,
        )
        corrupted_boxes.append(sample_boxes)
        observed_masks.append(sample_mask)
    return torch.stack(corrupted_boxes), torch.stack(observed_masks)


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = make_mot17_track_splits(root=ROOT / "data", T=16, stride=4)
    if not len(splits["train"]) or not len(splits["validation"]):
        raise ValueError("MOT17 train and validation splits must contain track windows")
    train_loader = DataLoader(splits["train"], batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(splits["validation"], batch_size=BATCH_SIZE)

    model = GRUSequenceModel(input_size=5, hidden_size=32, output_size=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scale = SCALE.to(device)
    training_rng = torch.Generator().manual_seed(SEED + 1)

    def run_epoch(loader, training):
        model.train(training)
        corruption_rng = (
            training_rng if training else torch.Generator().manual_seed(SEED + 2)
        )
        loss_sum, coordinate_count = 0.0, 0
        for boxes, clean_mask in loader:
            noisy_boxes, observed_mask = corrupt_batch(
                boxes, clean_mask, corruption_rng
            )
            clean_targets = boxes.to(device) / scale
            input_features = torch.cat(
                (
                    noisy_boxes.to(device) / scale,
                    observed_mask.to(device).unsqueeze(-1),
                ),
                dim=-1,
            )
            target_valid = clean_mask[:, 1:].to(device) > 0
            if not torch.any(target_valid):
                continue

            if training:
                optimizer.zero_grad()
            with torch.set_grad_enabled(training):
                predictions, _ = model(input_features[:, :-1])
                error = (predictions - clean_targets[:, 1:]).square()
                batch_loss = (error * target_valid.unsqueeze(-1)).sum() / (
                    target_valid.sum() * 4
                )
                if training:
                    batch_loss.backward()
                    optimizer.step()

            loss_sum += batch_loss.detach().item() * target_valid.sum().item() * 4
            coordinate_count += target_valid.sum().item() * 4

        if coordinate_count == 0:
            raise ValueError("No valid clean next-box targets were found")
        return loss_sum / coordinate_count

    train_losses, validation_losses = [], []
    best_validation_loss = float("inf")
    best_weights = None
    patience, stale_epochs = 5, 0

    for epoch in tqdm(range(100), desc="Robust GRU epochs"):
        train_loss = run_epoch(train_loader, training=True)
        validation_loss = run_epoch(validation_loader, training=False)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        tqdm.write(
            f"Epoch {epoch + 1:02d}: train MSE={train_loss:.6f}, "
            f"validation MSE={validation_loss:.6f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_weights = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                tqdm.write(f"Early stopping after epoch {epoch + 1}")
                break

    if best_weights is None:
        raise RuntimeError("Robust GRU training did not produce a checkpoint")
    weights_path = ROOT / "weights" / "mot17_gru_corrupt.pt"
    torch.save(best_weights, weights_path)

    OUTPUT_DIR = ROOT / "outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(train_losses, label="Train (corrupted inputs)")
    axis.plot(validation_losses, label="Validation (corrupted inputs)")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE against clean next box")
    axis.set_title("Observation-aware GRU training")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "mot17_gru_corrupt_training.png", dpi=150)
    plt.close(figure)
    print(f"Saved robust GRU checkpoint to {weights_path}")


if __name__ == "__main__":
    main()
