"""Train the GRU to predict the next MOT17 ground-truth box."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pa2.datasets import make_mot17_track_splits
from pa2.gru_model import GRUSequenceModel


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = make_mot17_track_splits(T=16, stride=4)
    train_data, validation_data = splits["train"], splits["validation"]
    if not len(train_data) or not len(validation_data):
        raise ValueError("MOT17 train and validation splits must both contain track windows")
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=256)

    model = GRUSequenceModel(input_size=4, hidden_size=32, output_size=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scale = torch.tensor([1920.0, 1080.0, 1920.0, 1080.0], device=device)

    def run_epoch(loader, training):
        model.train(training)
        loss_sum, coordinate_count = 0.0, 0
        for boxes, mask in loader:
            boxes, mask = boxes.to(device) / scale, mask.to(device)
            valid = (mask[:, :-1] > 0) & (mask[:, 1:] > 0)
            if not torch.any(valid):
                continue

            if training:
                optimizer.zero_grad()
            with torch.set_grad_enabled(training):
                predictions, _ = model(boxes[:, :-1])
                error = (predictions - boxes[:, 1:]).square()
                batch_loss = (error * valid.unsqueeze(-1)).sum() / (valid.sum() * 4)
                if training:
                    batch_loss.backward()
                    optimizer.step()

            loss_sum += batch_loss.detach().item() * valid.sum().item() * 4
            coordinate_count += valid.sum().item() * 4

        if coordinate_count == 0:
            raise ValueError("Dataset contains no adjacent valid box pairs")
        return loss_sum / coordinate_count

    train_errors, validation_errors = [], []
    best_validation_error = float("inf")
    best_weights = None
    patience, epochs_without_improvement = 5, 0
    max_epochs = 100

    for epoch in tqdm(range(max_epochs), desc="Epochs"):
        train_error = run_epoch(train_loader, training=True)
        validation_error = run_epoch(validation_loader, training=False)
        train_errors.append(train_error)
        validation_errors.append(validation_error)
        tqdm.write(
            f"Epoch {epoch + 1:02d}/{max_epochs} — "
            f"train MSE: {train_error:.6f}, validation MSE: {validation_error:.6f}"
        )

        if validation_error < best_validation_error:
            best_validation_error = validation_error
            best_weights = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                tqdm.write(f"Early stopping after epoch {epoch + 1}")
                break

    weights_dir = Path(__file__).resolve().parents[1] / "weights"
    weights_dir.mkdir(exist_ok=True)
    weights_path = weights_dir / "mot17_gru.pt"
    if best_weights is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    torch.save(best_weights, weights_path)
    print(f"Saved best validation checkpoint to {weights_path}")

    epochs = range(1, len(train_errors) + 1)
    plt.plot(epochs, train_errors, label="Training")
    plt.plot(epochs, validation_errors, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Masked MSE (normalized box coordinates)")
    plt.title("MOT17 GRU training and validation error")
    plt.legend()
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
