from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from ..types import DataBatchFloat, TabularBatchFloat, VectorBatchFloat


class NiceFlow(nn.Module):
    def __init__(self,
                 input_shape: tuple[int, ...],
                 n_layers: int,
                 transform_factory: Callable[[int], nn.Module],
                 base_distribution: torch.distributions.Distribution,
                 scale_fn: Callable[[VectorBatchFloat], VectorBatchFloat] = torch.exp):
        """Full NICE model. It isn't that nice in terms of performance, though.
        
        Parameters:
            input_shape: The basic NICE formulation works on flattened inputs using fully-connected MLP models. To be
                         able to produce outputs with the proper shape (like images), we need to reshape them
                         afterwards. To do that, we need to know the shape.
            n_layers: How many coupling layers to use for the model. Do not confuse this with neural network layers!
            transform_factory: Should be a function accepting a single integer input_dim and returning a neural network.
                               This network should accept batched vectors of that size and return outputs of the same
                               size. Because the coupling layers transform one part of the input given the other, we
                               pass half the total input size. This "factory"" is called for each coupling layer to
                               create the respective transformation network.
            base_distribution: Simple distribution that the flow will aim to transform the data distribution to.
                               NOTE this should be a 1d distribution, i.e. something like torch.distributions.Normal,
                               NOT MultivariateNormal.
            scale_fn: Should be a function mapping real numbers to strictly positive numbers to parameterize the
                      diagonal rescaling at the end. exp is a reasonable choice, but can be numerically unstable. Note
                      that performance can differ significantly for different choices of function.
        """
        super().__init__()
        total_dim = int(np.prod(input_shape))
        self.total_dim = total_dim
        self.input_shape = input_shape
        self.coupling_layers = nn.ModuleList()
        
        transform_options = ["one", "two"]
        for coupling_index in range(n_layers):
            self.coupling_layers.append(NiceLayer(transform_factory(total_dim // 2), transform_options[coupling_index % 2]))
            
        self.base_distribution = base_distribution
        self.scaler = nn.Parameter(torch.zeros(total_dim))  # results in scale of 1 for scale_fn=torch.exp
        self.scale_fn = scale_fn
        
    def forward(self,
                inputs: DataBatchFloat) -> TabularBatchFloat:
        return self.transform_forward(inputs)
    
    def transform_forward(self,
                          x: DataBatchFloat) -> TabularBatchFloat:
        """Forward transformation (data -> noise) of the flow."""
        h = x.view(x.shape[0], self.total_dim)
        for layer in self.coupling_layers:
            h = layer.transform_forward(h)
        h = h * self.scale_fn(self.scaler)
        return h
            
    def transform_backward(self,
                           h: TabularBatchFloat) -> DataBatchFloat:
        """Backward transformation (noise -> data) of the flow."""
        #x = h * torch.exp(-self.scaler)  # only valid if fixing scale_fn to exp
        x = h / self.scale_fn(self.scaler)  # more general
        for layer in reversed(self.coupling_layers):
            x = layer.transform_backward(x)
        x = x.view(-1, *self.input_shape)
        return x
            
    def log_p(self,
              samples: DataBatchFloat) -> VectorBatchFloat:
        """Compute log probability of a batch of data."""
        back_to_basics = self(samples)
        #log_determinant = self.scaler.sum()  # only valid for scale_fn exp
        log_determinant = torch.log(self.scale_fn(self.scaler)).sum()  # more general
        return self.base_distribution.log_prob(back_to_basics).sum(dim=1) + log_determinant

    def generate(self,
                 n_samples: int) -> DataBatchFloat:
        device = next(self.parameters()).device
        base = self.base_distribution.sample((n_samples, self.total_dim)).to(device)
        return self.transform_backward(base)


class NiceLayer(nn.Module):
    def __init__(self,
                 transform_function: nn.Module,
                 transform_part: str):
        """NICE additive coupling layer."""
        super().__init__()
        self.transform_function = transform_function
        
        if transform_part not in ["one", "two"]:
            raise ValueError(f"transform_part must be 'one' or 'two', got {transform_part}")
        self.transform_part = transform_part
        
    def forward(self,
                inputs: TabularBatchFloat) -> TabularBatchFloat:
        return self.transform_forward(inputs)
    
    def transform_forward(self,
                          x: TabularBatchFloat) -> TabularBatchFloat:
        """Data -> noise transformation.
        
        The coupling layers in a model should alternatingly transform the first or second part of the input, leaving the
        rest unchanged. This could be implemented more "elegantly" by always transforming the _same_ part of the input,
        but reversing (or generally permuting) the channel order between coupling layers. However, this is somewhat
        harder to understand and more error-prone, so we stick to the more explicit version.
        """
        x1, x2 = self.split_even_odd(x)
        if self.transform_part == "two":
            h1 = x1
            h2 = x2 + self.shift(x1)
        else:
            h1 = x1 + self.shift(x2)
            h2 = x2
        together = torch.stack([h1, h2], dim=-1)
        return together.view(*x.shape)
    
    def transform_backward(self,
                           h: TabularBatchFloat) -> TabularBatchFloat:
        """Noise -> data transformation."""
        h1, h2 = self.split_even_odd(h)
        if self.transform_part == "two":
            x1 = h1
            x2 = h2 - self.shift(h1)
        else:
            x1 = h1 - self.shift(h2)
            x2 = h2
        together = torch.stack([x1, x2], dim=-1)
        return together.view(*h.shape)
    
    def shift(self,
              input_part: TabularBatchFloat) -> TabularBatchFloat:
        """Compute the shift function based on one part of the input."""
        return self.transform_function(input_part)
    
    def split_even_odd(self,
                       inputs: TabularBatchFloat) -> tuple[TabularBatchFloat, TabularBatchFloat]:
        """Split input into two parts corresponding to even and odd indices.
        
        Note that for images, this will lead to a column-wise split. A checkerboard pattern would likely be better.
        However, that is not as quick to implement.
        """
        even_inds = torch.arange(0, inputs.shape[1], 2)
        odd_inds = torch.arange(1, inputs.shape[1], 2)
        even = inputs[:, even_inds]
        odd = inputs[:, odd_inds]
        return even, odd
