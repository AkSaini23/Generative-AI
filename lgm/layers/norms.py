from torch import nn

from ..types import DataBatchFloat


class DummyNorm(nn.Module):
    def __init__(self,
                 *args,
                 **kwargs):
        """A no-op that just passes the input through unchanged.

        Used out of sheer laziness so we if we want to build a model without normalization, the code doesn't need
        if/else cases every time. Just use DummyNorm instead of e.g. Batchnorm2d or whatever.
        """
        super().__init__()
    
    def forward(self,
                inputs: DataBatchFloat) -> DataBatchFloat:
        return inputs


class BNWrap2d(nn.BatchNorm2d):
    def __init__(self,
                 n_groups: int,
                 *args,
                 **kwargs):
        """Dumb wrapper around Batchnorm.

        Sometimes we want to use a different normalization instead, like GroupNormalization.  Unfortunately, that one
        requires an additional argument (number of groups). To avoid having different interfaces depending on which
        normalization we use, we just always pass the n_groups argument... but then ignore it for Batchnorm.

        This is bad! Don't use this as an example of good/smart code! I'm not a software engineer!

        Parameters:
            n_groups: The parameter we will ignore.
            args, kwargs: Additional arguments to Batchnorm.
        """
        super().__init__(*args, **kwargs)


class BNWrap1d(nn.BatchNorm1d):
    def __init__(self,
                 n_groups: int,
                 *args,
                 **kwargs):
        """The same as above... but the 1d version. Ugh."""
        super().__init__(*args, **kwargs)


class LNWrap(nn.LayerNorm):
    def __init__(self,
                 n_groups: int,
                 *args,
                 **kwargs):
        """LayerNorm wrapper with n_groups."""
        super().__init__(*args, **kwargs)


def n_groups_convert(n_groups: int,
                     n_filters: int) -> int | None:
    """Helper to figure out how many groups we want for GroupNorm.
    
    If n_groups is a positive number, it will simply be returned. If n_groups is negative, this is interpreted as a
    desired *group size*, so we use n_filters to figure out how many groups this will come out to be. If n_groups is
    None, we just return None.

    Parameters:
        n_groups: Perhaps badly named, either the number of groups (if positive) or group size (if negative).
        n_filters: Expected input size in terms of number of filters. Needed to compute number of groups from desired
        group size.

    Returns:
        Number of groups for GroupNorm.
    """
    if n_groups is None:
        return None
    elif n_groups < 0:
        group_size = -n_groups
        n_groups = n_filters // group_size
    return n_groups
