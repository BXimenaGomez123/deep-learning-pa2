"""Train/evaluate the GRU on synthetic box tracks and plot its error."""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from pa2.gru_model import GRUSequenceModel
from pa2.synthetic_dataset import SyntheticTrackDataset


def main():
    torch.manual_seed(0)
    image_size, sequence_length = 128, 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = SyntheticTrackDataset(
        n_videos=8, T=sequence_length, stride=4, seed_offset=0, img_size=image_size
    )
    validation_data = SyntheticTrackDataset(
        n_videos=2, T=sequence_length, stride=4, seed_offset=10_000, img_size=image_size
    )
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=64)

    model = GRUSequenceModel(input_size=4, hidden_size=32, output_size=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def run_epoch(loader, training=False):
        model.train(training)
        squared_error_sum, coordinate_count = 0.0, 0
        for boxes, mask in loader:
            boxes = boxes.to(device) / image_size
            mask = mask.to(device)
            valid_pairs = (mask[:, :-1] > 0) & (mask[:, 1:] > 0)
            if not torch.any(valid_pairs):
                continue
            if training:
                optimizer.zero_grad()
            with torch.set_grad_enabled(training):
                predictions, _ = model(boxes[:, :-1])
                error = (predictions - boxes[:, 1:]).square()
                loss_sum = (error * valid_pairs.unsqueeze(-1)).sum()
                count = valid_pairs.sum() * 4
                loss = loss_sum / count
                if training:
                    loss.backward()
                    optimizer.step()
            squared_error_sum += loss_sum.detach().item()
            coordinate_count += count.item()
        return squared_error_sum / coordinate_count

    training_errors, validation_errors = [], []
    for _ in range(30):
        training_errors.append(run_epoch(train_loader, training=True))
        validation_errors.append(run_epoch(validation_loader))

    print(f"Final validation MSE: {validation_errors[-1]:.6f}")
    epochs = range(1, len(training_errors) + 1)
    plt.plot(epochs, training_errors, label="Training")
    plt.plot(epochs, validation_errors, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Masked MSE (normalized box coordinates)")
    plt.title("GRU next-frame box prediction")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
