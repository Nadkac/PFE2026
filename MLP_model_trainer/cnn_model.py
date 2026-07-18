import torch
import torch.nn as nn


class ZumiCNN(nn.Module):
    """
    CNN compact pour la conduite autonome du Zumi.

    Entrée :
        image RGB [batch, 3, 120, 160]

    Sortie :
        [left_speed, right_speed] normalisées dans [-1, 1]
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # 120x160 -> environ 60x80
            nn.Conv2d(
                in_channels=3,
                out_channels=24,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),

            # 60x80 -> environ 30x40
            nn.Conv2d(
                in_channels=24,
                out_channels=36,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),

            # 30x40 -> environ 15x20
            nn.Conv2d(
                in_channels=36,
                out_channels=48,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            # Extraction de caractéristiques plus fines
            nn.Conv2d(
                in_channels=48,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Taille fixe, peu importe les dimensions intermédiaires
            nn.AdaptiveAvgPool2d((3, 4))
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64 * 3 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),

            nn.Linear(64, 16),
            nn.ReLU(inplace=True),

            nn.Linear(16, 2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)