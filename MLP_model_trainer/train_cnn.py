# train_cnn.py

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from cnn_dataset import create_cnn_loaders
from cnn_model import ZumiCNN


def train_cnn():
    data_dir = Path("cnn_dataset")
    save_dir = Path("cnn_checkpoints")
    save_dir.mkdir(exist_ok=True)

    train_loader, val_loader = create_cnn_loaders(
        data_dir=data_dir,
        batch_size=32
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ZumiCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")

    for epoch in range(1, 51):
        model.train()
        train_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "input_shape": [1, 3, 120, 160],
                "output_dim": 2
            }, save_dir / "best_cnn.pt")

    print("Entraînement terminé.")


if __name__ == "__main__":
    train_cnn()