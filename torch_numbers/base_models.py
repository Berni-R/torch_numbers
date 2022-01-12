from typing import Literal
import torch
from torch import nn
from torch import Tensor


EXAMPLE_BATCH_SIZE = 2


class Encoder(nn.Module):
    
    def __init__(
            self,
            relu_negative_slope: float = 0.1,
            norm: Literal['batch', 'instance'] = 'instance',
            out_dim: int = 256,
            layer_dims: tuple[int, int, int] = (12, 32, 256),
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, layer_dims[0], (3, 3)),
            nn.LeakyReLU(relu_negative_slope, inplace=True),
            nn.AvgPool2d(2),
            {'batch': nn.BatchNorm2d(layer_dims[0]), 'instance': nn.InstanceNorm2d(layer_dims[0])}[norm],
            nn.Conv2d(layer_dims[0], layer_dims[1], (6, 6)),
            nn.LeakyReLU(relu_negative_slope, inplace=True),
            {'batch': nn.BatchNorm2d(layer_dims[1]), 'instance': nn.InstanceNorm2d(layer_dims[1])}[norm],
            nn.AvgPool2d(2),
            nn.Conv2d(layer_dims[1], layer_dims[2], (3, 3), groups=2),
            nn.LeakyReLU(relu_negative_slope, inplace=True),
            {'batch': nn.BatchNorm2d(layer_dims[2]), 'instance': nn.InstanceNorm2d(layer_dims[2])}[norm],
            nn.Conv2d(layer_dims[2], out_dim, (2, 2), groups=8),
            nn.Flatten(),
        )

        self.example_input_array = torch.rand((EXAMPLE_BATCH_SIZE, 1, 28, 28))

    def forward(self, imgs: Tensor) -> tuple[Tensor, Tensor]:
        return self.model(imgs)


class Decoder(nn.Module):
    
    def __init__(
            self,
            relu_negative_slope: float = 0.1,
            norm: Literal['batch', 'instance'] = 'instance',
            in_dim: int = 256,
            layer_dims: tuple[int, int, int] = (256, 128, 64, 32, 12),
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, layer_dims[0]),
            nn.LeakyReLU(relu_negative_slope, inplace=True),
            nn.Unflatten(-1, (layer_dims[0], 1, 1)),
            nn.ConvTranspose2d(layer_dims[0], layer_dims[1], kernel_size=2),
            nn.LeakyReLU(relu_negative_slope, inplace=True),
            {'batch': nn.BatchNorm2d(layer_dims[1]), 'instance': nn.InstanceNorm2d(layer_dims[1])}[norm],
            nn.ConvTranspose2d(layer_dims[1], layer_dims[2], kernel_size=3, stride=1),
            nn.LeakyReLU(relu_negative_slope, inplace=True),
            {'batch': nn.BatchNorm2d(layer_dims[2]), 'instance': nn.InstanceNorm2d(layer_dims[2])}[norm],
            nn.ConvTranspose2d(layer_dims[2], layer_dims[3], kernel_size=5, stride=3),
            nn.LeakyReLU(relu_negative_slope, inplace=True),
            {'batch': nn.BatchNorm2d(layer_dims[3]), 'instance': nn.InstanceNorm2d(layer_dims[3])}[norm],
            nn.ConvTranspose2d(layer_dims[3], layer_dims[4], kernel_size=2, stride=2),
            nn.LeakyReLU(relu_negative_slope, inplace=True),
            {'batch': nn.BatchNorm2d(layer_dims[4]), 'instance': nn.InstanceNorm2d(layer_dims[4])}[norm],
            nn.Conv2d(layer_dims[4], 1, kernel_size=1),
        )

        self.example_input_array = torch.rand((EXAMPLE_BATCH_SIZE, in_dim))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.model(x)

