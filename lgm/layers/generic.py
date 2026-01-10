import torch
from torch import nn

from .norms import BNWrap1d, DummyNorm, n_groups_convert
from ..types import AnyFloat, TabularBatchFloat, VectorBatchFloat


def hidden_linear(dim: int,
                  activation: type[nn.Module],
                  norm: type[nn.Module] = BNWrap1d,
                  norm_groups: int = None) -> nn.Module:
    """Basic linear-norm-activation sequence.

    Using lazy layers is a bit dangerous, but given that we often use linear layers after flattened conv layers,
    I think it's worth it -- otherwise, computing the number of inputs would often be cumbersome.

    Parameters:
        dim: Desired output size.
        activation: Activation function. Should be passed like nn.ReLU, NOT nn.ReLU()!
        norm: Desired normalization. Should be passed just like the activation function.
        norm_groups: If norm is GroupNorm, this needs to be passed. If this is a positive number, use this many groups
                     for GroupNorm. If this is a negative number, we will use this as *group size* instead.
    """
    use_bias = isinstance(norm, DummyNorm)
    norm_groups = n_groups_convert(norm_groups, dim)
    return nn.Sequential(nn.LazyLinear(dim, bias=use_bias),
                         norm(norm_groups, dim),
                         activation())


class Reshape(nn.Module):
    def __init__(self,
                 target_shape: tuple[int, ...]):
        """Just a wrapper around the reshape function.
        # TODO can be replaced by nn.Unflatten.

        Parameters:
            target_shape: Input will be reshaped to this. Must include batch dimension; can use -1 for this.
        """
        super().__init__()
        self.target_shape = target_shape

    def forward(self,
                inputs: AnyFloat) -> AnyFloat:
        return inputs.view(*self.target_shape)
    

class OneHot(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def forward(self,
                inputs: VectorBatchFloat) -> TabularBatchFloat:
        return nn.functional.one_hot(inputs, self.n_classes).to(torch.float32)
