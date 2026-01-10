from collections.abc import Callable

import numpy as np
import torch

from .trainer import FlowModel
from ..common import interpolate
from ..types import DataBatchFloat, DataFloat


def interpolate_partial(code1: DataFloat,
                        code2: DataFloat,
                        interpolate_levels: int,
                        n_levels_total: int,
                        n_interpolations: int) -> DataBatchFloat:
    """Interpolate up to a certain latent variable level.
    
    This is only appropriate for multi-level Flow models like RealNVP or Glow. This will interpolate the "highest"
    levels from code1 to code2, while keeping the deeper levels as in code1. If take_levels is equal to n_levels_total,
    this just performs full interpolation.

    Parameters:
        code1: Starting point for all levels.
        code2: End point for early levels that are actually interpolated.
        take_levels: How many levels to interpolate. We start counting at 1. That is, passing 1 here will only
                     interpolate the first level, passing 2 the first two, etc.
        n_levels_total: How many levels the Flow model has. Unfortunately we don't know this from anywhere else, and
                        since the last level doesn't follow the halving pattern in terms of size, we need this
                        knowledge for correct behavior.
        n_interpolations: Will create this many steps between code1 and code2, including endpoints

    Returns:
        Batch of interpolated codes.
    """
    if interpolate_levels == n_levels_total:
        take_n_dims = code1.shape[0]
    else:
        take_n_dims = code1.shape[0] // 2
        for level_ind in range(1, interpolate_levels):
            take_n_dims += code1.shape[0] // 2**(level_ind+1)
    code1_take = code1[:take_n_dims]
    code2_take = code2[:take_n_dims]
    interpolated_take = interpolate(code1_take, code2_take, n_interpolations=n_interpolations)
    remainder_tiled = torch.tile(code1[take_n_dims:], [n_interpolations, 1])
    return torch.cat((interpolated_take, remainder_tiled), dim=1)


def corrupt_likelihoods(model: FlowModel,
                        healthy_samples: DataBatchFloat,
                        corrupt_fn: Callable[[DataBatchFloat, int], DataBatchFloat],
                        n_steps: int) -> np.ndarray:
    """Corrupts some data at various levels and computes log likelihoods for each.
    
    Parameters:
        model: Flow model to use for likelihood computations
        healthy_samples: Batch of data samples to evaluate.
        corrupt_fn: Function that takes healthy samples and a "corruption level" between 0 and 1 and returns corrupted
                    samples.
        n_steps: How many corruption steps to fill the interval 0 to 1 in.
    """
    results = []
    for corruption in np.linspace(0, 1, n_steps):
        corrupted_samples = corrupt_fn(healthy_samples, corruption)
        with torch.inference_mode():
            corrupted_bits_per_dim = model.log_p(corrupted_samples) / (model.total_dim * torch.log(torch.tensor(2.)))
        results.append(corrupted_bits_per_dim.cpu().numpy())
    return np.stack(results, axis=1)
