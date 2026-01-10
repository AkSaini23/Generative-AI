from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from ..common import spatial_broadcast_index, sum_except
from ..types import DataBatchFloat, TabularBatchFloat, VectorBatchFloat


class GlowFlow(nn.Module):
    def __init__(self,
                 input_shape: tuple[int, ...],
                 n_levels: int,
                 n_layers_per_level: int,
                 hidden_dim_base: int,
                 transform_factory: Callable[[int, int], nn.Module],
                 base_distribution: torch.distributions.Distribution,
                 scale_fn: Callable[[VectorBatchFloat], VectorBatchFloat] = torch.exp,
                 channel_growth_factor: int = 2):
        """Full Glow model.
        
        Parameters:
            input_shape: As usual, the module is by default not aware of the input size. We need this to figure out the
                         shape of the noise components at the various levels.
            n_levels: How many levels to build the model for. Each level works at a progressively smaller resolution,
                      with part of the input being "factored out".
            n_layers_per_level: Number of coupling layers per level. Do not confuse with neural network layers.
            hidden_dim_base: Number of channels used in the coupling layers for the first level.
                             This is doubled at each successive level.
            transform_factory: Function that takes two ints (input channels and hidden channels) and returns a module
                               accepting that many inputs and returning *twice* as many outputs. This will be used to
                               create the transformation function returning shift and scale values at each coupling
                               layer. the hidden channel argument could technically be ignored, but it's useful so that
                               we can create larger networks at the deeper levels.
            base_distribution: Simple distribution that the flow will aim to transform the data distribution to.
            scale_fn: See notes in NiceFlow.
            channel_growth_factor: By how much to increase number of hidden channels in the coupling layer
                                   transformations at each level.
        """
        super().__init__()
        self.input_shape = input_shape
        self.noise_shapes = []
        self.total_dim = int(np.prod(input_shape))

        self.levels = nn.ModuleList()
        for ind in range(n_levels):
            self.levels.append(GlowLevel(n_layers_per_level, transform_factory,
                                         input_channels=2**ind * 2*input_shape[0],
                                         hidden_dim=hidden_dim_base * channel_growth_factor**ind,
                                         scale_fn=scale_fn))
            # each level increases channel count by times 4, half of which are factored to noise. so base 2 exponential.
            noise_channels = 2**(ind + 1) * input_shape[0]
            if ind == n_levels - 1:
                # final level does not factor out anything, so keeps full dim instead of half.
                noise_channels *= 2
            noise_pixels = input_shape[1] // (2**(ind + 1))  # per dimension
            self.noise_shapes.append((noise_channels, noise_pixels, noise_pixels))
        self.base_distribution = base_distribution
        
    def forward(self,
                inputs: DataBatchFloat) -> tuple[TabularBatchFloat, VectorBatchFloat]:
        return self.transform_forward(inputs)
    
    def transform_forward(self,
                          x: DataBatchFloat) -> tuple[TabularBatchFloat, VectorBatchFloat]:
        """Data -> noise transformation.
        
        Returns the transformed outputs, as well as the total log determinant.
        After each level except, half the outputs are split off and "fast-tracked" to the end. Afterwards, we
        concatenate the final remainder with all the split-off to construct the full noise vector.
        """
        log_determinant = 0
        h = x
        split_off = []
        for ind, level in enumerate(self.levels):
            h, level_logdet = level.transform_forward(h)
            log_determinant += level_logdet
            # since the layer code transforms the second part given the first,
            # we would hope the second part gets a better "chance" to be more transformed.
            # so I choose to split off the second part... I dunno man.
            if ind < len(self.levels) - 1:
                n_channels = h.shape[1]
                h, split_here = torch.split(h, n_channels // 2, dim=1)
                split_off.append(split_here.view(x.shape[0], -1))
            else:
                split_off.append(h.view(x.shape[0], -1))
        h = torch.cat(split_off, dim=1)
        return h, log_determinant
            
    def transform_backward(self,
                           h: TabularBatchFloat) -> DataBatchFloat:
        """Noise -> data transformation.
        
        Since we have to revert the per-level-split operation, we create the correct per-level noise shapes in advance.
        These are then concatenated with the partially transformed outputs after each level.
        """
        level_noises = list(reversed(self.factor_noise(h)))
        x = level_noises[0]
        for ind, level in enumerate(reversed(self.levels)):
            if ind > 0:
                attach_noise = level_noises[ind]
                x = torch.cat((x, attach_noise), dim=1)
            x = level.transform_backward(x)
        return x
            
    def log_p(self,
              samples: TabularBatchFloat) -> VectorBatchFloat:
        """Compute log probability of a batch of data."""
        back_to_basics, log_determinant = self(samples)
        return sum_except(self.base_distribution.log_prob(back_to_basics)) + log_determinant

    def generate(self,
                 n_samples: int,
                 temperature: float = 1.) -> DataBatchFloat:
        device = next(self.parameters()).device
        base = self.base_distribution.sample((n_samples, self.total_dim)).to(device) * temperature
        return self.transform_backward(base)
    
    def factor_noise(self,
                     h: TabularBatchFloat) -> list[DataBatchFloat]:
        """Construct the correctly shaped per-level noise components from a full noise vector."""
        noises = []
        for shape in self.noise_shapes:
            total_size = np.prod(shape)
            part = h[:, :total_size]
            h = h[:, total_size:]
            noises.append(part.view(-1, *shape))
        return noises


class GlowLevel(nn.Module):
    def __init__(self,
                 n_layers: int,
                 factory: Callable[[int, int], nn.Module],
                 input_channels: int,
                 hidden_dim: int,
                 scale_fn: Callable[[VectorBatchFloat], VectorBatchFloat] = torch.exp):
        """Single level where all coupling layers work at the same resolution.
        
        Parameters:
            n_layers: Number of coupling layers (not neural network layers) at this level.
            factory: As passed by GlowFlow.
            input_channels: Number of channels the *affine transformation netowrks at this level* receive. Note that
                            this is NOT the number of channels in the data at this leve, but HALF of that!!
            hidden_dim: Number of channels to use for the networks at this level.
            scale_fn: See notes in NiceFlow.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(GlowLayer(2*input_channels, factory(input_channels, hidden_dim), scale_fn))
        self.squeezer = nn.PixelUnshuffle(2)
        self.unsqueezer = nn.PixelShuffle(2)

    def forward(self,
                inputs: DataBatchFloat) -> tuple[DataBatchFloat, VectorBatchFloat]:
        return self.transform_forward(inputs)
    
    def transform_forward(self,
                          x: DataBatchFloat) -> tuple[DataBatchFloat, VectorBatchFloat]:
        """Data -> noise transformation.
        
        Returns the transformed outputs, as well as the total log determinant at this level. The first operation at each
        level is to downsample the spatial dimensions and move the variables into the channel dimension.

        NOTE that this function will initialize the activation normalization component of each coupling layer the first
        time it is called. Thus, you should make sure the first time you call the model is on actual training data, not
        some placeholder dummy data.
        """
        log_determinant = 0
        h = self.squeezer(x)
        for layer in self.layers:
            if not layer.initialized:
                layer.initialize(h)
            h, layer_logdet = layer.transform_forward(h)
            log_determinant += layer_logdet
        return h, log_determinant
        
    def transform_backward(self,
                           h: DataBatchFloat) -> DataBatchFloat:
        """Noise -> data transformation."""
        x = h
        for layer in reversed(self.layers):
            x = layer.transform_backward(x)
        return self.unsqueezer(x)


class GlowLayer(nn.Module):
    def __init__(self,
                 total_channels,
                 transform_function: nn.Module,
                 scale_fn: Callable[[VectorBatchFloat], VectorBatchFloat] = torch.exp):
        """Single Glow affine coupling layer with intertible 1x1 convolution.
        
        Parameters:
            total_channels: Number of channels in the input to this coupling layer. Note that is NOT the number of
                            channels that the transformation network receives, as that will only receive half the
                            channels.
            transform_function: Neural network module to be used for the affine transformation.
            scale_fn: See notes in NiceFlow.
        """
        super().__init__()
        self.transform_function = transform_function
        self.actnorm_s = None
        self.actnorm_b = None
        self.initialized = False
        self.scale_fn = scale_fn
        
        # initialize to random orthonormal matrix
        w_init = np.linalg.qr(np.random.randn(total_channels, total_channels))[0]
        self.w = nn.Parameter(torch.tensor(w_init, dtype=torch.float32))
        
    def forward(self,
                inputs: DataBatchFloat) -> tuple[DataBatchFloat, VectorBatchFloat]:
        return self.transform_forward(inputs)
    
    def transform_forward(self,
                          x: DataBatchFloat) -> tuple[DataBatchFloat, VectorBatchFloat]:
        """Data -> noise transformation.
        
        The layer has to have been initialized beforehand to populate the activation normalization component. If you use
        the model as intended, this should happen automatically the first time you call it.
        """
        if not self.initialized:
            raise ValueError("Initialize model plz")
        x = self.actnorm_forward(x)
        x = self.conv_permute(x)
        x1, x2 = self.split_channels(x)
        h1 = x1
        scale_shift = self.transform_function(x1)
        log_scale, shift = torch.split(scale_shift, scale_shift.shape[1] // 2, dim=1)
        #h2 = torch.exp(log_scale) * x2 + shift
        h2 = self.scale_fn(log_scale) * x2 + shift
        
        n_spatial_dims = torch.prod(torch.tensor(x.shape[2:]))
        #scale_logdet = sum_except(log_scale)
        scale_logdet = sum_except(torch.log(self.scale_fn(log_scale)))
        actnorm_logdet = torch.log(self.actnorm_s.abs()).sum() * n_spatial_dims
        conv_logdet = torch.log(torch.linalg.det(self.w).abs()) * n_spatial_dims
        h = torch.cat([h1, h2], dim=1)
        return h, scale_logdet + actnorm_logdet + conv_logdet
    
    def transform_backward(self,
                           h: DataBatchFloat) -> DataBatchFloat:
        """Noise -> data transformation."""
        if not self.initialized:
            raise ValueError("Initialize model plz")
        h1, h2 = self.split_channels(h)
        x1 = h1
        scale_shift = self.transform_function(h1)
        log_scale, shift = torch.split(scale_shift, scale_shift.shape[1] // 2, dim=1)
        #x2 = (h2 - shift) * torch.exp(-log_scale)
        x2 = (h2 - shift) / self.scale_fn(log_scale)

        x = torch.cat([x1, x2], dim=1)
        x = self.conv_inverse(x)
        return self.actnorm_backward(x)
    
    def conv_permute(self,
             inputs: DataBatchFloat) -> DataBatchFloat:
        """Mix channels using invertible 1x1 convolution."""
        return torch.nn.functional.conv2d(inputs, self.w[spatial_broadcast_index(inputs)])
    
    def conv_inverse(self,
                     inputs: DataBatchFloat) -> DataBatchFloat:
        """Reverse channel mixing.
        
        Computing the inverse is expensive, but only needs to be done when generating.
        TODO could compute it once and store the result.
        """
        # TODO use LU decomposition
        inverse_w = torch.linalg.inv(self.w)
        return torch.nn.functional.conv2d(inputs, inverse_w[spatial_broadcast_index(inputs)])
    
    def split_channels(self,
                       inputs: DataBatchFloat) -> DataBatchFloat:
        """Split data into two parts.
        
        Since we rely on convolutions for channel mixing, we can always split in the same way.
        """
        n_channels = inputs.shape[1]
        return torch.split(inputs, n_channels // 2, dim=1)
    
    def actnorm_forward(self,
                        x: DataBatchFloat) -> DataBatchFloat:
        # TODO parameterize s with exp?
        return (self.actnorm_s + self.eps) * x + self.actnorm_b
    
    def actnorm_backward(self,
                         h: DataBatchFloat) -> DataBatchFloat:
        return (h - self.actnorm_b) / (self.actnorm_s + self.eps)
    
    def initialize(self,
                   input_batch: DataBatchFloat) -> DataBatchFloat:
        """Initialize activation norm parameters using input statistics."""
        mean_dims = (0, *range(2, input_batch.dim()))
        broadcast_index = spatial_broadcast_index(input_batch)
        means = input_batch.mean(dim=mean_dims)[broadcast_index]
        stds = torch.std(input_batch, dim=mean_dims)[broadcast_index]

        # TODO UGH!!
        device = next(self.parameters()).device
        self.actnorm_b = nn.Parameter(-means).to(device)
        self.actnorm_s = nn.Parameter(1/stds).to(device)
        self.eps = torch.tensor(1e-8)
        self.initialized = True
