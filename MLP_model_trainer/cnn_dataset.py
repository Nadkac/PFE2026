# cnn_dataset.py

import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class ZumiImageDataset(Dataset):
    def __init__(self, data_dir, image_size=(160, 120)):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.labels_path = self.data_dir / "labels.jsonl"

        self.samples = []

        with open(self.labels_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = self.data_dir / sample["image"]
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Image introuvable: {image_path}")

        image = cv2.resize(image, self.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype("float32") / 255.0

        # HWC → CHW pour PyTorch
        image = torch.tensor(image).permute(2, 0, 1)

        label = torch.tensor(
            [sample["left"], sample["right"]],
            dtype=torch.float32
        )

        return image, label


def create_cnn_loaders(data_dir, batch_size=32, val_ratio=0.2):
    dataset = ZumiImageDataset(data_dir)

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader