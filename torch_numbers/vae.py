from typing import Optional, Literal
from torch import Tensor
import torch
from torch import nn

from .base_models import Encoder, Decoder


class VAEEncoder(nn.Module):

    def __init__(
            self,
            encoder: Optional[nn.Module] = None,
            interims_dim: int = 256,
            encoded_dim: int = 256,
            relu_negative_slope: float = 0.1,
            norm: Literal['batch', 'instance'] = 'instance',
    ):
        super().__init__()

        self.encode = encoder or Encoder(relu_negative_slope=relu_negative_slope, norm=norm, out_dim=interims_dim)
        self.activate = nn.LeakyReLU(relu_negative_slope, inplace=True)
        self.z_mean = nn.Sequential(
            nn.Linear(interims_dim, encoded_dim),
            nn.LeakyReLU(relu_negative_slope, inplace=True),
        )
        self.z_log_var = nn.Sequential(
            nn.Linear(interims_dim, encoded_dim),
            nn.LeakyReLU(relu_negative_slope, inplace=True),
        )

        self.example_input_array = torch.rand((2, 1, 28, 28))

    def forward(self, imgs: Tensor) -> tuple[Tensor, Tensor]:
        x = self.encode(imgs)
        x = self.activate(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var


class Sampling(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, z_mean: Tensor, z_log_var: Tensor) -> Tensor:
        eps = torch.randn(z_mean.shape)
        return z_mean + torch.exp(0.5 * z_log_var) * eps


class VAEDecoder(nn.Module):

    def __init__(
            self,
            decoder: Optional[nn.Module] = None,
            interims_dim: int = 256,
            encoded_dim: int = 256,
            final_activation: bool = True,
            relu_negative_slope: float = 0.1,
            norm: Literal['batch', 'instance'] = 'instance',
    ):
        super().__init__()

        self.dense = nn.Sequential(
            nn.Linear(encoded_dim, interims_dim),
            nn.LeakyReLU(relu_negative_slope, inplace=True),
        )
        self.decode = decoder or Decoder(relu_negative_slope=relu_negative_slope, norm=norm, in_dim=interims_dim)
        self.activate = nn.Sigmoid() if final_activation else nn.Identity()

        self.example_input_array = torch.rand((2, encoded_dim))

    def forward(self, z: Tensor) -> Tensor:
        x = self.dense(z)
        x = self.decode(x)
        x = self.activate(x)

        return x
