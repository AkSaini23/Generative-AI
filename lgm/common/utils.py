from collections.abc import Iterable

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from ..types import (DataBatchFloat, DataBatchFloatChannelsLast, BroadcastDataBatchFloat, SquareMatrixFloat,
                     TabularBatchFloat, VectorBatchFloat)


def count_parameters(model: nn.Module) -> int:
    """Get number of (trainable) parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_learning_curves(metrics: dict[str, np.ndarray],
                         keys: Iterable[str]):
    """Basic plots for metrics of interest.

    Parameters:
        metrics: Dictionary as returned by a Trainer object's train_model function.
        keys: Plots are made for each metric named in here. Each plot gets one line for training and one for validation.
    """
    for key in keys:
        plt.figure(figsize=(12, 3))
        plt.plot(metrics["train_" + key], label="train")
        plt.plot(metrics["val_" + key], label="validation")
        plt.legend()
        plt.title(key)
        plt.xlabel("Epoch")
        plt.show()


def sum_except(x: DataBatchFloat,
               keepdim: bool = False) -> VectorBatchFloat | BroadcastDataBatchFloat:
    # TODO typing is imprecise
    """Sum over all dimensions of x except the first one."""
    return x.sum(dim=tuple(i for i in range(1, x.ndim)), keepdim=keepdim)


def batched_multiply(tensor: DataBatchFloat,
                     vector: VectorBatchFloat) -> DataBatchFloat:
    """Multiply a (b x ...) tensor with a b-shaped vector."""
    return tensor * vector[(...,) + (None,) * (tensor.ndim - 1)]


def spatial_broadcast_index(inputs: DataBatchFloat) -> tuple:
    """Creates indexing for broadcasting one tensor to another's shape.
    
    The indeded usage is when you have a batch of data, say (b x c x h x w) images. You also have a (b x c) tensor that
    you would like to broadcast over the spatial dimensions. This can be achieved via indexing with [:, :, None, None].
    However, the number of None required depends on the dimensionality of the spatial inputs (images, waveforms, etc.).
    This function returns the index given an arbitrarily-shaped input.

    Parameters:
        inputs: The data tensor you want to broadcast to.
    """
    # TODO figure out type hint lmao
    return (...,) + (torch.newaxis,) * (inputs.dim() - 2)


def interleave(inputs: DataBatchFloat,
               other: DataBatchFloat) -> DataBatchFloat:
    """Interleave rows of two tensors of the same shape.
    
    The result will feature rows from inputs and other in an alternating fashion.
    """
    stacked = torch.stack((inputs, other), dim=1)
    return stacked.view(2*inputs.shape[0], *inputs.shape[1:])
    

def channels_first_to_last(tensor: DataBatchFloat) -> DataBatchFloatChannelsLast:
    """Moves channels of a batch (of e.g. images) to the last axis."""
    n_dims = tensor.dim()
    return tensor.permute(0, *range(2, n_dims), 1)


def channels_last_to_first(tensor: DataBatchFloatChannelsLast) -> DataBatchFloat:
    """Moves channels of a batch (of e.g. images) after the batch axis."""
    last_dim_index = tensor.dim() - 1
    return tensor.permute(0, last_dim_index, *range(1, last_dim_index))


def squared_distances(row_input: TabularBatchFloat,
                      column_input: TabularBatchFloat) -> SquareMatrixFloat:
    """Compute squared distances between two batches of vectors.
    
    We make use of (a - b)**2 = a**2 - 2ab + b**2.
    """
    return ((row_input**2).sum(dim=1)[:, torch.newaxis] 
            - 2 * (row_input @ column_input.T) 
            + (column_input**2).sum(dim=1)[torch.newaxis, :])
