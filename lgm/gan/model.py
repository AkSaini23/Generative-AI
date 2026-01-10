from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from ..types import DataBatchFloat, TabularBatchFloat


class GAN(nn.Module):
    def __init__(self,
                 generator: Generator,
                 discriminator: Discriminator):
        """Container module for GAN training.
        
        Do not try to call this as a module. It's just a place to bundle generator and discriminator parameters.
        """
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator


class Generator(nn.Sequential):
    def __init__(self,
                 noise_fn: Callable[[int], DataBatchFloat],
                 *args,
                 **kwargs):
        """Thin wrapper around Sequential.
        
        Parameters:
            noise_fn: Function accepting a batch size and returning batch_size many noise tensors as input to the
                      generator.
            args, kwargs: Arguments to nn.Sequential.
        """
        super().__init__(*args, **kwargs)
        self.noise_fn = noise_fn

    def generate(self,
                 batch_size: int) -> DataBatchFloat:
        device = next(self.parameters()).device
        noise = self.noise_fn(batch_size).to(device)
        return self(noise)


class Discriminator(nn.Module):
    def __init__(self,
                 root: nn.Module,
                 body: nn.Module,
                 head: nn.Module):
        """GAN discriminator returning hidden features (for feature matching).
        
        We cannot use Sequential due to the multiple outputs.

        Parameters:
            root: Any layers that should be applied before the body, such as an initial convolution.
            body: Should be a CNNBody with return_all=True or some other module giving multiple outputs that can be
                  used for feature matching.
            head: A module that takes the body final output and returns a binary classification (single output). In
                  case you are using binary cross-entropy, do NOT include a sigmoid activation in the head. Often this
                  can just be Flatten layer followed by one or more Linear ones.
        """
        super().__init__()
        self.root = root
        self.body = body
        self.head = head

    def forward(self,
                inputs: DataBatchFloat) -> TabularBatchFloat:
        initial = self.root(inputs)
        body_last_output, body_all_outputs = self.body(initial)
        return self.head(body_last_output), body_all_outputs
                 

class SDLayer(nn.Module):
    """Layer from "Progressive Growing of GANs" paper.
    
    Computes per-feature variances, summarizes to one global value via mean, and concatenates a single feature
    with this value.
    """
    def forward(self,
                inputs: DataBatchFloat) -> DataBatchFloat:
        variances = torch.var(inputs, dim=0)
        global_value = variances.mean()
        broadcast_shape = torch.tensor(inputs.shape)
        broadcast_shape[1] = 1
        broadcast = global_value * torch.ones(*broadcast_shape, device=inputs.device)
        return torch.cat([inputs, broadcast], dim=1)
