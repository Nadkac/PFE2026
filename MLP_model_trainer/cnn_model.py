# cnn_model.py

import torch
import torch.nn as nn


class ZumiCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 100),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(100, 50),
            nn.ReLU(),

            nn.Linear(50, 2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x