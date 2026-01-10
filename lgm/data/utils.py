import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common import spatial_broadcast_index
from ..types import AnyFloat


def dataset_mean_std(dataloader: DataLoader,
                     mode: str = "global") -> tuple[AnyFloat, AnyFloat]:
    """Kinda scuffed way to get mean and std from a dataset for normalize.
    
    Parametes:
        per_dim: If True, compute separate statistics for each data dimension (e.g. pixel). Otherwise, a global mean
                 and standard deviation are returned.
    """
    # TODO do this in one go if at all possible...
    # TODO numericaly questionable and naive
    data_sum = sum(batch[0].sum(dim=0) for batch in dataloader)
    data_mean_per_dim = data_sum / len(dataloader.dataset)
    if mode == "global":
        data_mean = data_mean_per_dim.mean()
    elif mode == "channel":
        example = next(iter(dataloader))[0]
        data_mean = data_mean_per_dim.mean(dim=tuple(range(1, example.dim() - 1)))[spatial_broadcast_index(example)]
    elif mode == "dim":
        data_mean = data_mean_per_dim
    else:
        raise ValueError(f"Unsupported mode {mode}. Allowed are 'global', 'channel', 'dim'.")

    sum_of_squares = sum(((batch[0] - data_mean)**2).sum(dim=0) for batch in dataloader)
    mean_of_squares = sum_of_squares / len(dataloader.dataset)
    if mode == "global":
        data_variance = mean_of_squares.mean()
    elif mode == "channel":
        data_variance = mean_of_squares.mean(dim=tuple(range(1, example.dim() - 1)))[spatial_broadcast_index(example)]
    else:
        data_variance = mean_of_squares

    return data_mean, torch.sqrt(data_variance)
