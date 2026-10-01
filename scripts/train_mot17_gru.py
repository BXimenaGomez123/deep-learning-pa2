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
    dataset = make_mot17_track_splits(T=16, stride=4)["train"]
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    model = GRUSequenceModel(input_size=4, hidden_size=32, output_size=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scale = torch.tensor([1920.0, 1080.0, 1920.0, 1080.0], device=device)

    epoch_errors = []
    for epoch in tqdm(range(20), desc="Epochs"):
        model.train()
        loss_sum, coordinate_count = 0.0, 0
        for boxes, mask in tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False):
            boxes, mask = boxes.to(device) / scale, mask.to(device)
            valid = (mask[:, :-1] > 0) & (mask[:, 1:] > 0)
            if not torch.any(valid):
                continue

            optimizer.zero_grad()
            predictions, _ = model(boxes[:, :-1])
            error = (predictions - boxes[:, 1:]).square()
            batch_loss = (error * valid.unsqueeze(-1)).sum() / (valid.sum() * 4)
            batch_loss.backward()
            optimizer.step()
            loss_sum += batch_loss.detach().item() * valid.sum().item() * 4
            coordinate_count += valid.sum().item() * 4

        epoch_error = loss_sum / coordinate_count
        epoch_errors.append(epoch_error)
        tqdm.write(f"Epoch {epoch + 1:02d}/20 — masked MSE: {epoch_error:.6f}")

    weights_dir = Path(__file__).resolve().parents[1] / "weights"
    weights_dir.mkdir(exist_ok=True)
    weights_path = weights_dir / "mot17_gru.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Saved weights to {weights_path}")

    plt.plot(range(1, len(epoch_errors) + 1), epoch_errors)
    plt.xlabel("Epoch")
    plt.ylabel("Masked MSE (normalized box coordinates)")
    plt.title("MOT17 GRU training error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
