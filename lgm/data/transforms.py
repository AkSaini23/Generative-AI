from typing import Any

import torch
from torch import nn
from torchaudio.transforms import TimeStretch
from torchaudio.functional import pitch_shift
from torchvision.transforms.v2 import Transform

from ..types import DataFloat, ImageFloat, SequenceFloat


class PadTransform(Transform):
    def __init__(self,
                 pad: int):
        """Just a wrapper around the pad function.
        
        Useful to have this on the data processing side. ONLY FOR IMAGES right now!
        # TODO generalize to non-images e.g. sequences
        
        Parameters:
            pad: Pad this many pixels on each side.
        """
        super().__init__()
        self.pad = pad

    def transform(self,
                  inputs: ImageFloat,
                  _params: Any) -> ImageFloat:
        return nn.functional.pad(inputs, (self.pad, self.pad, self.pad, self.pad))
    

class PadToFixed(Transform):
    def __init__(self,
                 target_size: int):
        """Pad OR TRUNCATE to desired length.
        
        This is different from PadTransform, which always *adds* a fixed amount. In contrast, this one will result in
        the output *always* having length target_size. Also, this currently only works for sequences.
        # TODO generalize to other shapes.
        """
        super().__init__()
        self.target_size = target_size

    def transform(self,
                  inputs: SequenceFloat,
                  _params: Any) -> SequenceFloat:
        to_pad = self.target_size - inputs.shape[-1]
        if to_pad < 0:
            return inputs[:, :self.target_size]
        else:
            return torch.nn.functional.pad(inputs, (0, to_pad))


class Binarize(Transform):
    def __init__(self,
                 threshold: float = 0.5):
        """Turn inputs into binary numbers according to threshold.

        Parameters:
            threshold: Everything above this value will become 1; everything else 0.
        """
        super().__init__()
        self.threshold = threshold

    def transform(self,
                  inputs: DataFloat,
                  _params: Any) -> DataFloat:
        return torch.where(inputs > self.threshold, 1., 0.)


class ToGrayscale(Transform):
    """Convert color image(s) to grayscale.

    We count axes from the back so it should work for both batches or unbatched single images. The grayscale conversion
    is "naive" in that we simply use the mean of the RGB channels. There are other, more sophisticated methods that
    properly respect the differences in luminance between the different colors.

    NOTE that we invert colors by subtracting the mean value from 1. This is because I hardcoded the "Greys" colormap
    in most places that do plotting, and this colormap assigns black, rather than white, to larger values. This is a
    very hacky solution and subject to change.
    """
    def transform(self,
                  inputs: DataFloat,
                  _params: Any) -> DataFloat:
        return 1 - inputs.mean(dim=-3, keepdim=True)


class Normalize(Transform):
    def __init__(self,
                 mean: float,
                 stddev: float):
        """Normalize inputs by a given mean and standard deviation.

        There is already such a transform in torchvision, but that seems to be a bit restrictive with regards to what
        shapes it accepts for the parameters. This one should be more flexible.
        """
        super().__init__()
        self.mean = mean
        self.stddev = stddev

    def transform(self,
                  inputs: DataFloat,
                  _params: Any) -> DataFloat:
        return (inputs - self.mean) / self.stddev
    

class UniformDequantize(Transform):
    """Dequantize 8bit images by "filling in" the space between the integer values.
    
    We assume the images were already normalized to range [0, 1] beforehand, simply because it matches the way the
    preprocessing is implemented in out repository. This WILL NOT BE CORRECT if they are not in that range.
    There isn't one correct way to do this. We randomly add up to 1 pixel value, and then re-normalize such that the
    maximum is 1. This is the accepted version in the literature. Another version would be to add or subtract up to 0.5
    pixel values (normalized to the correct range) and then clip to [0, 1]. This is the commented-out version.
    """
    def transform(self,
                  inputs: DataFloat,
                  _params: Any) -> DataFloat:
        #return torch.clamp(inputs + (torch.rand_like(inputs) - 0.5)/255, 0., 1.)
        return (inputs*255 + torch.rand_like(inputs)) / 256
    

class Transpose(Transform):
    """Flip height/width dimensions."""
    def transform(self,
                  inputs: ImageFloat,
                  _params: Any) -> ImageFloat:
        return torch.permute(inputs, (-3, -1, -2))
    

class PhaseFlip(Transform):
    def transform(self,
                  inputs: SequenceFloat,
                  _params: Any) -> SequenceFloat:
        # either 1 or -1
        multiplier = 2*torch.randint(low=0, high=2, size=(1,))[0].to(torch.float32) - 1
        return inputs * multiplier
    

class PitchShiftRandom(nn.Module):
    def __init__(self, 
                 shift_max: int,
                 sr: int):
        super().__init__()
        self.sr = sr
        self.shift_max = shift_max

    def forward(self,
                inputs: torch.Tensor) -> torch.Tensor:
        random_shift = torch.randint(-self.shift_max, self.shift_max + 1, (1,))[0]
        return pitch_shift(inputs, self.sr, random_shift)


class TimeStretchRandom(nn.Module):
    def __init__(self,
                 stretch_max: float,
                 hop_length: int,
                 n_fft: int):
        super().__init__()
        self.stretcher = TimeStretch(hop_length=hop_length, n_freq=n_fft)
        self.stretch_max = stretch_max

    def forward(self,
                inputs: torch.Tensor) -> torch.Tensor:
        random_stretch = (2.*torch.rand(1)[0] - 1.) * self.stretch_max
        factor = 1 + random_stretch
        return self.stretcher(inputs, factor)
    

class ToLogit(Transform):
    def __init__(self,
                 alpha: float = 0.05):
        """Transform inputs in [0, 1] to logit space.
        
        Parameters:
            alpha: Should be a small number to pull inputs away from 0 or 1, since the logit function is infinite for
            those values.
        """
        super().__init__()
        self.alpha = alpha

    def transform(self,
                  inputs: ImageFloat,
                  _params: Any) -> ImageFloat:
        squeezed = self.alpha + (1 - 2*self.alpha) * inputs
        return torch.logit(squeezed)
    

class FromLogit(Transform):
    def __init__(self,
                 alpha: float = 0.05):
        """Reverse the ToLogit transform.
        
        Parameters:
            alpha: Analogous to ToLogit. Since you are likely using both transforms in tandem, the same alpha should be
                   used for both.
        """
        super().__init__()
        self.alpha = alpha

    def transform(self,
                  inputs: ImageFloat,
                  _params: Any) -> ImageFloat:
        squeezed = torch.nn.functional.sigmoid(inputs)
        return (squeezed - self.alpha) / (1 - 2*self.alpha)
