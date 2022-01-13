import os
from typing import Union, Optional, Callable
from numpy.typing import ArrayLike
from torch import Tensor
import torch
from torch import nn
import numpy as np
from torchvision.utils import save_image
from IPython.display import display, Image


def count_params(module: nn.Module, trainable: bool = False) -> int:
    """Count the number of parameters in a module."""
    return sum(np.prod(p.shape) for p in module.parameters() if (p.requires_grad or not trainable))


def denorm(img: Union[ArrayLike, Tensor]) -> Union[ArrayLike, Tensor]:
    """From image scaled from -1 to +1, to scale from 0 to 1."""
    return (img + 1) / 2


@torch.no_grad()
def display_imgs(
        imgs: Union[ArrayLike, Tensor],
        n_columns: int = 10,
        path: Optional[str] = None,
        delete_file: Optional[bool] = None,
        denorm: Optional[Callable] = None,
        invert: bool = True,
):
    """Display images in notebook and simultaneously store them to a file.

    Args:
        imgs (Tensor, list of these):   The images to show.
        n_columns (int):                Images are ordered in a grid of this number of columns.
        path (str):                     The path to store the image grid to. (Optional.)
        delete_file (bool):             Whether to delete the image file after displaying.
        denorm (Callable):              A function used to denorm the images.
        invert (bool):                  Invert the images (after denorm), i.e. calculate `1-img`.
    """
    if path is None:
        path = f'tmp_imgs_{np.random.randint(16**10):10x}.png'
        delete_file = delete_file or True
    else:
        delete_file = delete_file or False

    if denorm is not None:
        imgs = denorm(imgs)
    if invert:
        imgs = 1 - imgs
    save_image(imgs, path, nrow=n_columns)
    display(Image(path))
    if delete_file:
        os.remove(path)


def display_examples(gan, n_lines: int = 3, n_columns: int = 10, random: bool = False,
                     path: Optional[str] = None, delete_file: bool = True, denorm: Optional[Callable] = None,
                     invert: bool = True):
    if path is None:
        path = f"fake_images_{np.random.randint(16 ** 10):010x}.png"

    if random:
        n = torch.randint(10, size=(n_lines * n_columns,))
    else:
        n = torch.arange(n_lines * n_columns) % 10

    images = gan(n)
    display_imgs(images, n_columns=n_columns, path=path, delete_file=delete_file, denorm=denorm, invert=invert)
