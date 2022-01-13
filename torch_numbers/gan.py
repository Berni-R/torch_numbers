from typing import Optional, Literal
from torch import Tensor
import torch
from torch import nn
from torch.nn import functional as F

from .base_models import Encoder, Decoder


class Discriminator(nn.Module):

    def __init__(
            self,
            encoder: Optional[nn.Module] = None,
            encoded_dim: int = 256,
            final_activation: bool = True,
            relu_negative_slope: float = 0.1,
            norm: Literal['batch', 'instance'] = 'instance',
    ):
        super().__init__()

        self.encode = encoder or Encoder(relu_negative_slope=relu_negative_slope, norm=norm, out_dim=encoded_dim)
        self.activate = nn.LeakyReLU(relu_negative_slope, inplace=True)

        self.verify = nn.Sequential(
            nn.Linear(encoded_dim, 64),
            nn.LeakyReLU(relu_negative_slope, inplace=True),
            {'batch': nn.BatchNorm1d(64), 'instance': nn.Identity()}[norm],
            nn.Linear(64, 2),
            nn.LogSoftmax(dim=1) if final_activation else nn.Identity(),
        )

        self.classify = nn.Sequential(
            nn.Linear(encoded_dim, 64),
            nn.LeakyReLU(relu_negative_slope, inplace=True),
            {'batch': nn.BatchNorm1d(64), 'instance': nn.Identity()}[norm],
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1) if final_activation else nn.Identity(),
        )

        self.example_input_array = torch.rand((2, 1, 28, 28))

    def forward(self, imgs: Tensor) -> tuple[Tensor, Tensor]:
        x = self.encode(imgs)
        x = self.activate(x)
        validity = self.verify(x)
        lbl = self.classify(x)
        return validity, lbl


class Generator(nn.Module):

    def __init__(
            self,
            decoder: Optional[nn.Module] = None,
            num_classes: int = 10,
            relu_negative_slope: float = 0.1,
            norm: Literal['batch', 'instance'] = 'instance',
    ):
        super().__init__()
        self.num_classes = num_classes

        self.decode = decoder or Decoder(relu_negative_slope=relu_negative_slope, norm=norm, in_dim=num_classes)
        self.activate = nn.Sigmoid()

        self.example_input_array = torch.arange(10)

    def forward(self, lbls: Tensor) -> Tensor:
        z = F.one_hot(lbls, num_classes=self.num_classes)
        z = z + 0.1 * torch.randn((len(z), self.num_classes))

        x = self.decode(z)
        x = self.activate(x)

        return x
