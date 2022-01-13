from typing import Optional, Literal
import torch
from torch import nn, Tensor
from torch.nn import functional as F

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
        self.final_layer = nn.Linear(interims_dim, 2 * encoded_dim)

        self.example_input_array = torch.rand((2, 1, 28, 28))

    def forward(self, imgs: Tensor) -> tuple[Tensor, Tensor]:
        x = self.encode(imgs)
        x = self.activate(x)
        mu_log_var = self.final_layer(x).view(len(x), 2, -1)
        mu = mu_log_var[:, 0, :].view(len(x), -1)
        log_var = mu_log_var[:, 1, :].view(len(x), -1)
        return mu, log_var


class Sampling(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, mu: Tensor, log_var: Tensor) -> Tensor:
        if self.training:
            return mu + torch.exp(0.5 * log_var) * torch.randn(log_var.shape)
        else:
            return mu


def vae_losses(reconstruction: Tensor, imgs: Tensor, mu: Tensor, log_var: Tensor) -> tuple[Tensor, Tensor]:
    reconstruction_loss = F.binary_cross_entropy(reconstruction, imgs)
    kl_loss = 0.5 * torch.mean(mu**2 + log_var.exp() - log_var - 1)
    return reconstruction_loss, kl_loss


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
