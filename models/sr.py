import torch
import torch.nn as nn


class SRResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class SimpleSRResNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        hidden_channels=64,
        num_blocks=6,
    ):
        super().__init__()

        self.head = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[SRResidualBlock(hidden_channels) for _ in range(num_blocks)]
        )
        self.body_tail = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.tail = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shallow_features = self.head(x)
        deep_features = self.body(shallow_features)
        deep_features = self.body_tail(deep_features) + shallow_features
        residual = self.tail(deep_features)
        return x + residual
