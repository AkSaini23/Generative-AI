from collections.abc import Iterable

import torch
from torch import nn

from .norms import BNWrap1d, BNWrap2d, DummyNorm, n_groups_convert
from ..common import spatial_broadcast_index
from ..types import DataBatchFloat, TabularBatchFloat


INTERPOLATION_MAP = {"1d": "linear", "2d": "bilinear"}


class CNNBody(nn.Module):
    def __init__(self,
                 blocks_per_level: int | Iterable[int],
                 level_filters: Iterable[int],
                 level_strides: Iterable[int],
                 input_channels: int,
                 direction: str,
                 block_fn: type[nn.Module],
                 final_normact: bool = True,
                 return_all: bool = False,
                 take_skips: bool = False,
                 **block_kwargs):
        """A sequence of CNN "levels", which in turn consist of sequences of "blocks".

        Each level can have a different number of filters, and optional down-/up-sampling.
    
        Parameters:
            blocks_per_level: How many blocks each level consists of. Can be int, then all levels will use the same
                              number of blocks. Or an iterable like filters and strides.
            level_filters: Iterable (e.g. list) with number of filters for each level.
            level_strides: Stride (up- or downsampling factor) at each level.
            input_channels: Number of channels for the input. Duh.
            direction: One of "up" or "down". "down" would usually be for encoder-type models, "up" for
                       decoder/generator-like.
            block_fn: Which type of convolutional block to use.
            final_normact: If True, apply a final sequence of normaliztion & activation after the last level. This can
                           improve performance somewhat. But note that this sometimes be superfluous in models where we
                           apply multiple CNNBodys in sequence, e.g. U-Nets. Here, the first Body would end with a final
                           norm-act, and the second body would also start with norm-act. In such cases, you could set
                           this to False for the first Body.
            return_all: If True, additionally to the final output, will return a list of lists of tensors, containing
                        the outputs of all blocks at all levels. Useful for feature matching in GANs, or skip
                        connections in U-Net architectures (encoder part).
            take_skips: If True, each block expects an additional input besides the output of the previous block. These
                        will have to be provided in the forward call as a list of lists of tensors skip_input, such as
                        returned by another CNNBody with return_all == True. Useful for skip-connections in U-Net
                        architectures (decoder part).
                        NOTE a naive implementation will duplicate the input for the first block of the first level.
                        This is because it receives the same "regular" input as the skip input. We avoid this by special
                        arguments to the CNNLevel class. Also NOTE that you are responsible for making sure no skip 
                        input is passed to the first block of the first level when actually calling the model! Else you
                        will get a shape mismatch.
            block_kwargs: Arguments for the blocks.
        """
        super().__init__()
        if (len(level_filters) != len(level_strides)
            or (isinstance(blocks_per_level, Iterable) and len(blocks_per_level) != len(level_filters))):
            raise ValueError("Blocks, filters and strides should have same number of entries!")
        if isinstance(blocks_per_level, int):
            blocks_per_level = [blocks_per_level] * len(level_filters)

        self.levels = nn.ModuleList()
        previous_filters = input_channels
        for level_ind in range(len(level_filters)):
            current_filters = level_filters[level_ind]
            self.levels.append(CNNLevel(blocks_per_level[level_ind], previous_filters, current_filters,
                                        stride=level_strides[level_ind], direction=direction,
                                        block_fn=block_fn,
                                        return_all=return_all,
                                        take_skips=2*take_skips if level_ind == 0 else take_skips,
                                        **block_kwargs))
            previous_filters = current_filters

        self.return_all = return_all
        self.final_normact = final_normact
        if final_normact:
            norm_groups = n_groups_convert(block_kwargs.get("norm_groups", None), level_filters[-1])
            data_format = block_kwargs.get("data_format", "2d")
            default_norm = BNWrap1d if data_format == "1d" else BNWrap2d
            self.final_norm = block_kwargs.get("norm", default_norm)(norm_groups, level_filters[-1])
            self.final_activation = block_kwargs["activation"]()

    def forward(self,
                inputs: DataBatchFloat,
                skip_inputs: Iterable[Iterable[DataBatchFloat]] | None = None,
                conditioning: TabularBatchFloat | None = None)\
                    -> DataBatchFloat | tuple[DataBatchFloat, Iterable[Iterable[DataBatchFloat]]]:
        outputs = inputs
        if self.return_all:
            all_body_outputs = []
        for ind, level in enumerate(self.levels):
            skip_input = None if skip_inputs is None else skip_inputs[ind]
            if self.return_all:
                outputs, all_level_outputs = level(outputs, skip_input, conditioning)
                all_body_outputs.append(all_level_outputs)
            else:
                outputs = level(outputs, skip_input, conditioning)

        if self.final_normact:
            outputs = self.final_activation(self.final_norm(outputs))
        return (outputs, all_body_outputs) if self.return_all else outputs


class CNNLevel(nn.Module):
    def __init__(self,
                 n_blocks: int,
                 previous_filters: int,
                 current_filters: int,
                 stride: int,
                 direction: str,
                 block_fn: type[nn.Module],
                 return_all: bool = False,
                 take_skips: int = 0,
                 **block_kwargs):
        """One "level" of a CNN consists of an arbitrary number of convolutional "blocks" as well as up-/down-sampling.

        All blocks at a level operate on the same image size and output the same number of filters.
        
        Parameters:
            n_blocks: How many blocks the level should consist of.
            previous_filters: Number of channels of the input to this level.
            current_filters: Number of channels on this level.
            stride: Up- or down-sampling factor applied by the first block. NOTE: In case of downsampling, the stride
                    is applied at the start of the first block. In case of upsampling, bilinear upsampling is added at
                    the *end* of the level.
            direction: One of "up" or "down". "down" would usually be for encoder-type models, "up" for
                       decoder/generator-like.
            block_fn: Which type of convolutional block to use.
            return_all: See doc of CNNBody. If True, return a list of tensors of all the block outputs in this level.
            take_skips: See doc of CNNBody. If > 0, forward must receive a list of tensors in skip_inputs. A value of 1
                        will assume *all* blocks get a skip input. A value of 2 will build the *first block* in the
                        level to *not* get skip inputs! Usually you want to set this to 2 for the first level in the
                        decoder of a U-Net-style model.
            block_kwargs: Arguments for the blocks.
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        for layer_ind in range(n_blocks):
            do_take_skips = 0 if layer_ind == 0 and take_skips == 2 else take_skips
            self.blocks.append(block_fn(previous_filters=previous_filters if layer_ind == 0 else current_filters,
                                        current_filters=current_filters,
                                        stride=stride if (layer_ind == 0 and direction == "down") else 1,
                                        skip_dim=current_filters if do_take_skips else 0,
                                        **block_kwargs))
        if direction == "up" and stride > 1:
            data_format = block_kwargs.get("data_format", "2d")
            self.upsample = UpsampleAA(scale_factor=stride, mode=INTERPOLATION_MAP[data_format])
        else:
            self.upsample = None
        self.return_all = return_all

    def forward(self,
                inputs: DataBatchFloat,
                skip_inputs: Iterable[DataBatchFloat] | None = None,
                conditioning: TabularBatchFloat | None = None)\
                    -> DataBatchFloat | tuple[DataBatchFloat, Iterable[DataBatchFloat]]:
        outputs = inputs
        if self.return_all:
            all_level_outputs = []
        for ind, block in enumerate(self.blocks):
            skip_input = None if skip_inputs is None else skip_inputs[ind]
            outputs = block(outputs, skip_input, conditioning)
            if self.return_all:
                all_level_outputs.append(outputs)
            
        if self.upsample is not None:
            outputs = self.upsample(outputs)
        return (outputs, all_level_outputs) if self.return_all else outputs


class ConvBlockPrenorm(nn.Module):
    def __init__(self,
                 n_layers: int,
                 previous_filters: int,
                 current_filters: int,
                 kernel_size: int,
                 activation: type[nn.Module],
                 stride: int = 1,
                 aa_downsample: bool = True,
                 data_format: str = "2d",
                 bottleneck_factor: int = 1,
                 norm: type[nn.Module] = BNWrap2d,
                 norm_groups: int = None,
                 use_residual: bool = True,
                 force_1x1: bool = False,
                 use_se: bool = True,
                 se_reduction_factor: int | None = None,
                 se_bias: float = 0.,
                 skip_dim: int = 0,
                 cond_dim: int = 0):
        """Multi-purpose CNN block.

        Implements an arbitrary number of Normalization-Activation-Convolution operations. Optional features include a
        residual connection, squeeze-excite, and bottlenecks using 1x1 convolutions. For example, the blocks proposed
        in the original ResNet paper could be implemented by either
        - n_layers=2 and bottleneck_factor=1
        - n_layers=3 and bottleneck_factor=4

        Parameters:
            n_layers: Number of Norm-Act-Conv layers.
            previous_filters: Number of channels in input.
            current_filters: Output channels.
            kernel_size: What it says.
            activation: Guess what. See notes in hidden_linear function.
            stride: Stride applied by the first convolution.
            aa_downsample: If True, instead of downsampling using strided convolutions, we use a stride-1 convolution
                           and then apply bilinear downsampling with anti-aliasing.
            data_format: One of "1d" or "2d". Determines the convolution used. 2d is for images, 1d for time series,
                         e.g. audio.
            bottleneck_factor: If > 1, first and last convolutions will be 1x1 convolutions. The first one will map to
                               channel number current_filters // bottleneck_factor. All "middle" convolutions will
                               operate at this channel number. The final convolution will map back to current_filters.
            norm: See notes in hidden_linear function.
            norm_groups: See notes in hidden_linear function.
            use_residual: If True, input is added to output of the block.
            force_1x1: If True, input is always transformed by a 1x1 convolution before being added in the residual 
                       connection. Otherwise, a 1x1 convolution is still used if necessary (stride >1, or channel
                       numbers differ). Ignored if use_residual is False.
            use_se: If True, apply a squeeze-excite operation at the end of the block (in the residual part, if that is
                    used).
            se_reduction_factor: The hidden layer in the SE "mini network" will use current_filters // reduction units.
            se_bias: Pass a number > 0 to bias the final SE layer (sigmoid output) towards 1, to let through more
                     activation early during training.
            skip_dim: If this block is used in a level that takes skip inputs, this needs to be the number of channels
                      of the respective input.
            cond_dim: If this block is used in a level that takes conditioning inputs (e.g. class information for
                      conditional generation, and/or time in diffusion models), this must be the dimensionality of that
                      input.
        """
        super().__init__()
        if data_format not in ["1d", "2d"]:
            raise ValueError(f"Invalid data format {data_format}. Allowed are '1d', '2d'.")
        conv_fn = nn.Conv2d if data_format == "2d" else nn.Conv1d
        # HACK overwrite 2D batchnorm default for 1d data for convenience.
        # TODO refactor to have norm be e.g. a string
        if norm is BNWrap2d and data_format == "1d":
            norm = BNWrap1d

        use_bias = norm is DummyNorm
        input_dim = previous_filters + skip_dim
        norm_groups_first = n_groups_convert(norm_groups, input_dim)
        self.first_norm = norm(norm_groups_first, input_dim)

        def first_layer(_filters, _kernel_size):
            # helper to reduce code duplication
            if stride > 1 and aa_downsample:
                return nn.Sequential(activation(),
                                     conv_fn(input_dim, _filters, _kernel_size, bias=use_bias,
                                             stride=1, padding="same"),
                                     DownsampleAA(scale_factor=stride, mode=INTERPOLATION_MAP[data_format]))
            else:
                return nn.Sequential(activation(),
                                     conv_fn(input_dim, _filters, _kernel_size, bias=use_bias,
                                             stride=stride, padding=_kernel_size // 2))
        if n_layers < 1:
            raise ValueError("Must have at least one layer!")
        elif n_layers == 1:
            if bottleneck_factor > 1:
                print("WARNING! You have created a single-layer block with bottleneck. The bottleneck will be ignored.")
            self.layers = first_layer(current_filters, kernel_size)
        else:
            bottleneck_filters = current_filters // bottleneck_factor
            use_bottleneck = bottleneck_factor > 1
            bottleneck_kernel_size = 1 if use_bottleneck else kernel_size
            norm_groups = n_groups_convert(norm_groups, bottleneck_filters)
            if use_bottleneck and n_layers == 2:
                print("WARNING! You have created a bottleneck block with only two layers. "
                      "This block consists purely of 1x1 convolutions!")

            self.layers = first_layer(bottleneck_filters, bottleneck_kernel_size)
            for _ in range(n_layers - 2):
                self.layers.append(norm(norm_groups, bottleneck_filters))
                self.layers.append(activation())
                self.layers.append(conv_fn(bottleneck_filters, bottleneck_filters, kernel_size, bias=use_bias,
                                           padding="same"))
            self.layers.append(norm(norm_groups, bottleneck_filters))
            self.layers.append(activation())
            self.layers.append(conv_fn(bottleneck_filters, current_filters, bottleneck_kernel_size, bias=use_bias,
                                       padding="same"))

        if use_residual:
            if force_1x1 or previous_filters != current_filters or stride > 1:
                self.shortcut = conv_fn(previous_filters, current_filters, 1, stride=stride)
            else:
                self.shortcut = nn.Identity()
        self.use_residual = use_residual

        self.stride = stride
        if use_se:
            self.squeeze_excite = SqueezeExcite(current_filters, current_filters // se_reduction_factor,
                                                activation, se_bias)
        else:
            self.squeeze_excite = nn.Identity()

        if cond_dim:
            self.cond_add = nn.Linear(cond_dim, input_dim)
            self.cond_mult = nn.Linear(cond_dim, input_dim)
            nn.init.zeros_(self.cond_add.weight)
            nn.init.zeros_(self.cond_add.bias)
            nn.init.zeros_(self.cond_mult.weight)
            nn.init.zeros_(self.cond_mult.bias)
        self.has_cond = cond_dim > 0

    def forward(self,
                inputs: DataBatchFloat,
                skip_inputs: DataBatchFloat | None = None,
                conditioning: TabularBatchFloat | None = None) -> DataBatchFloat:
        features = self.residual_se(inputs, skip_inputs, conditioning)
        if self.use_residual:
            return features + self.shortcut(inputs)
        else:
            return features
        
    def residual_se(self,
                    inputs: DataBatchFloat,
                    skip_inputs: DataBatchFloat | None = None,
                    conditioning: TabularBatchFloat | None = None) -> DataBatchFloat:
        full_inputs = torch.cat((inputs, skip_inputs), dim=1) if skip_inputs is not None else inputs
        normalized = self.first_norm(full_inputs)
        if self.has_cond:
            broadcast = spatial_broadcast_index(inputs)
            normalized = ((1 + self.cond_mult(conditioning))[broadcast] * normalized 
                          + self.cond_add(conditioning)[broadcast])
        features = self.layers(normalized)
        features = self.squeeze_excite(features)
        return features


class SqueezeExcite(nn.Module):
    def __init__(self,
                 input_dim: int,
                 squeeze_dim: int,
                 activation: type[nn.Module],
                 bias: float):
        """Squeeze-and-excite module: https://arxiv.org/abs/1709.01507

        Used as a component in the block functions above.

        Parameters:
            input_dim: Number of channels in the input.
            squeeze_dim: Dimension of the bottleneck.
            activation: Hidden activation to use.
            bias: Bias of the last layer is initialized to this value. Since there is a sigmoid function after, starting
                  with a positive bias will result in sigmoid outputs being closer to 1 at the start.
        """
        super().__init__()
        self.mapper = nn.Sequential(nn.Linear(input_dim, squeeze_dim),
                                    activation(),
                                    nn.Linear(squeeze_dim, input_dim),
                                    nn.Sigmoid())
        nn.init.constant_(self.mapper[-2].bias, bias)

    def forward(self,
                inputs: DataBatchFloat) -> DataBatchFloat:
        spatial_dims = tuple(range(2, inputs.dim()))
        squeezed = inputs.mean(spatial_dims)
        excitation = self.mapper(squeezed)[spatial_broadcast_index(inputs)]
        return inputs * excitation
    

class UpsampleAA(nn.Upsample):
    """Standard upsampling layer, but forcing antialias=True.
    
    I don't think this does anything for upsampling LMAO.
    """
    def forward(self,
                inputs: DataBatchFloat) -> DataBatchFloat:
        return torch.nn.functional.interpolate(
            inputs,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
            antialias=self.mode in ["bilinear", "bicubic"])


class DownsampleAA(UpsampleAA):
    """Bilinear downsampling layer with anti-aliasing.
    
    We abuse the fact that upsampling layers just use interpolate, which can up- OR downsample. It also doesn't check
    whether the scale_factor is > 1. So we can use an upsampling layer for downsampling. Yay!
    # TODO possibly protect against floating point inaccuracy when dividing here.
    """
    def __init__(self,
                 scale_factor: float,
                 **kwargs):
        super().__init__(scale_factor=1/scale_factor, **kwargs)
