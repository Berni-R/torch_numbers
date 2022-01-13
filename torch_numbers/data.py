from typing import Optional, Literal
from pathlib import Path
from torchvision import datasets
from torchvision import transforms  # type: ignore
from torch.utils.data import Dataset

# number of examples, classes, min(val), max(val), mean(val), std(val)
DS_VALUE_STATS = {
    'MNIST': (60_000, 10, 0.0, 1.0, (0.1307,), (0.3081,)),
    'FashionMNIST': (60_000, 10, 0.0, 1.0, (0.2860,), (0.3530,)),
    'EMNIST': {
        'mnist': (60_000, 10, 0.0, 1.0, (0.1733,), (0.3317,)),
        'digits': (240_000, 10, 0.0, 1.0, (0.1733,), (0.3317,)),
        'letters': (124_800, 27, 0.0, 1.0, (0.1722,), (0.3309,)),
        'balanced': (112_800, 47, 0.0, 1.0, (0.1751,), (0.3332,)),
        'bymerge': (697_932, 47, 0.0, 1.0, (0.1736,), (0.3316,)),
        'byclass': (697_932, 62, 0.0, 1.0, (0.1736,), (0.3316,)),
    },
    'CIFAR10': (50000, 10, 0.0, 1.0, (0.4734,), (0.2515,))
}


def get_dataset(
        which: Literal['MNIST', 'FashionMNIST', 'EMNIST'],
        path: Path = './data',
        normalize: Optional[Literal['tanh', 'normal', 'sigmoid']] = None,
        emnist_split: Literal['byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist'] = 'digits',
) -> Dataset:
    path = Path(path)
    
    stats = DS_VALUE_STATS[which]
    if which == 'EMNIST':
        stats = stats[emnist_split]

    transform = [transforms.ToTensor()]
    if normalize is None or normalize == 'sigmoid':
        pass
    elif normalize == 'tanh':
        transform.append(transforms.Normalize((0.5,), (0.5,)))
    elif normalize == 'normal':
        transform.append(transforms.Normalize(stats[-2], stats[-1]))
    else:
        raise ValueError(f'Unknown `normalize`="{normalize}"')
    transform = transforms.Compose(transform)

    args = dict(root=path, download=True, transform=transform)
    if which == 'EMNIST':
        args['split'] = emnist_split
    ds = getattr(datasets, which)(**args)
    return ds
