import os
from typing import Union, Optional, Callable
from numpy.typing import ArrayLike
from torch import Tensor
import torch
from torch import nn
import numpy as np
from torchvision.utils import save_image  # type: ignore
from IPython.display import display, Image  # type: ignore


def count_params(module: nn.Module, trainable: bool = False) -> int:
    """Count the number of parameters in a module."""
    return sum(np.prod(p.shape) for p in module.parameters() if (p.requires_grad or not trainable))


def mask_images(
        imgs: Tensor,
        size: tuple[int, int],
        fill: float = 0.0,
        pos: Optional[ArrayLike] = None,
) -> tuple[Tensor, Tensor, np.ndarray]:
    img_shape = tuple(imgs.shape[-2:])

    if pos is None:
        pos = np.random.randint(0, img_shape[0]-size[0], size=(2, len(imgs)))
    pos = np.asarray(pos)

    masked_imgs = torch.empty_like(imgs)
    masked_imgs.copy_(imgs)
    masked = torch.empty(tuple(imgs.shape[:2]) + size).type_as(imgs)

    for i, (x0, y0) in enumerate(pos.T):
        masked[i] = imgs[i, :, x0:x0+size[0], y0:y0+size[1]]
        masked_imgs[i, :, x0:x0+size[0], y0:y0+size[1]] = fill
    return masked_imgs, masked, pos


def fill_mask(imgs: Tensor, fills: Tensor, pos: ArrayLike, size: tuple[int, int]) -> Tensor:
    pos = np.asarray(pos)
    unmasked = torch.empty_like(imgs)
    unmasked.copy_(imgs)

    for i, (x0, y0) in enumerate(pos.T):
        unmasked[i, :, x0:x0+size[0], y0:y0+size[1]] = fills[i]
    return unmasked


def denorm(img: Union[ArrayLike, Tensor]) -> Union[ArrayLike, Tensor]:
    """From image scaled from -1 to +1, to scale from 0 to 1."""
    return (img + 1) / 2  # type: ignore


@torch.no_grad()
def display_imgs(
        imgs: Union[ArrayLike, Tensor],
        n_columns: int = 10,
        path: Optional[str] = None,
        delete_file: Optional[bool] = None,
        ipy_display: bool = True,
        denorm: Optional[Callable[[Union[ArrayLike, Tensor]], Union[ArrayLike, Tensor]]] = None,
        invert: bool = True,
):
    """Display images in notebook and simultaneously store them to a file.

    Args:
        imgs (Tensor, list of these):   The images to show.
        n_columns (int):                Images are ordered in a grid of this number of columns.
        path (str):                     The path to store the image grid to. (Optional.)
        delete_file (bool):             Whether to delete the image file after displaying.
        ipy_display (bool):             Actually call `IPython.display`. Helpful, if used outside notebooks.
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
        imgs = 1 - imgs  # type: ignore
    save_image(imgs, path, nrow=n_columns)
    if ipy_display:
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
