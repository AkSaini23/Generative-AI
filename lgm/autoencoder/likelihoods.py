from collections.abc import Callable

import torch
from torch import nn

from ..common import sum_except
from ..types import DataBatchFloat, ScalarFloat


def map_likelihood(outputs: DataBatchFloat,
                   likelihood: str) -> DataBatchFloat:
    """Converts network outputs (distribution parameters) to images outputs (expected values).

    Parameters:
        likelihood: The likelihood we assume for the data, which leads to a loss function equivalent to maximum
                    likelihood. Any of the ones listed in the loss_likelihood function further below are valid.
        outputs: The values to convert.
    """
    if likelihood in ["gaussian_fixed_sigma", "gaussian_optimal_image_sigma", "gaussian_optimal_pixel_sigma",
                      "laplace"]:
        return outputs
    elif likelihood == "bernoulli":
        return nn.functional.sigmoid(outputs)
    elif likelihood == "continuous_bernoulli":
        return continuous_bernoulli_expected_value(nn.functional.sigmoid(outputs))
    elif likelihood == "beta":
        alphas, betas = beta_parameters(outputs)
        return alphas / (alphas + betas)
    else:
        raise ValueError(f"Invalid likelihood {likelihood}. Check your spelling or check autoencoder.py for "
                         "allowed values!")
    

def loss_likelihood(likelihood: str,
                    **additional_args) -> Callable[[DataBatchFloat, DataBatchFloat], ScalarFloat]:
    """Pick the correct loss for a given likelihood.

    Parameters:
        additional_args: Optional arguments for the loss besides outputs and targets. For example, these could be
                         constants needed for numerical stability.
    """
    if likelihood == "gaussian_fixed_sigma":
        function = squared_loss
    elif likelihood == "gaussian_optimal_pixel_sigma":
        function = log_loss
    elif likelihood == "gaussian_optimal_image_sigma":
        function = log_loss2
    elif likelihood == "laplace":
        function = abs_loss
    elif likelihood == "bernoulli":
        function = bernoulli_loss
    elif likelihood == "continuous_bernoulli":
        function = continuous_bernoulli_loss
    elif likelihood == "beta":
        function = beta_loss
    else:
        raise ValueError(f"Invalid likelihood {likelihood}. Check your spelling or check the autoencoder module for "
                         "allowed values!")

    if additional_args:
        def partial_function(x, y):
            return function(x, y, **additional_args)
        return partial_function

    return function


def squared_loss(outputs: DataBatchFloat,
                 targets: DataBatchFloat) -> ScalarFloat:
    """Corresponds to Gaussian likelihood with fixed sigma."""
    squared_difference = (outputs - targets)**2
    return sum_except(squared_difference).mean()


def log_loss(outputs: DataBatchFloat,
             targets: DataBatchFloat,
             lower_lim: float = 0.001) -> ScalarFloat:
    """Gaussian likelihood with the *optimal sigma* chosen separately for each pixel position AND example in the batch.

    Very unstable!

    Parameters:
        lower_lim: We cap the "optimal standard deviation" at this value to prevent collapse and numerical problems.
    """
    absolute_difference = (outputs - targets).abs()
    absolute_difference = torch.clamp(absolute_difference, min=lower_lim)
    return sum_except(torch.log(absolute_difference)).mean()


def log_loss2(outputs: DataBatchFloat,
              targets: DataBatchFloat,
              lower_lim: float = 0.001) -> ScalarFloat:
    """Gaussian likelihood with the *optimal sigma* chosen separately per example in the batch, but SHARED over pixels.

    Parameters:
        lower_lim: We cap the "optimal standard deviation" at this value to prevent collapse and numerical problems.
    """
    feature_dim = torch.prod(outputs.shape[1:])
    sum_of_squares = sum_except((outputs - targets)**2)
    sum_of_squares = torch.clamp(sum_of_squares, min=lower_lim)
    return 0.5*feature_dim * torch.log(sum_of_squares).mean()


def abs_loss(outputs: DataBatchFloat,
             targets: DataBatchFloat) -> ScalarFloat:
    """Corresponds to Laplacian likelihood with fixed b.

    Fun fact: Solving for the optimal per-pixel/position b seems to result in the same log_loss as in the Gaussian case.
    So we don't include another version of that.
    """
    absolute_difference = (outputs - targets).abs()
    return sum_except(absolute_difference).mean()


def bernoulli_loss(outputs: DataBatchFloat,
                   targets: DataBatchFloat) -> ScalarFloat:
    """Corresponds to Bernoulli likelihood.

    You *can* use it for targets in the range [0, 1], but that is no longer a proper likelihood.
    This is intended for binary targets.

    Outputs should be logits! No sigmoid in your model!
    """
    cross_entropy = nn.functional.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
    return sum_except(cross_entropy).mean()


def continuous_bernoulli_loss(outputs: DataBatchFloat,
                              targets: DataBatchFloat,
                              safety_eps: float = 1e-4) -> ScalarFloat:
    """Corresponds to the Continuous Bernoulli likelihood, a properly normalized Bernoulli on the range [0, 1].

    Adapted from the TF code: https://github.com/cunningham-lab/cb_and_cc

    Note that outputs should be logits!

    Parameters:
        safety_eps: Sigmoid outputs smaller than this, or larger than 1-eps, will be clamped. This is because the log
                    normalizer becomes unstable at the edges and will diverge to infinity. The smallest tolerable value
                    seems to be 1e-7.
    """
    cross_entropy = nn.functional.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
    normalizer = continuous_bernoulli_log_normalizer(torch.clamp(nn.functional.sigmoid(outputs),
                                                                 safety_eps, 1 - safety_eps))
    return sum_except(cross_entropy - normalizer).mean()


def continuous_bernoulli_log_normalizer(prob: DataBatchFloat,
                                        lower_lim: float = 0.49,
                                        upper_lim: float = 0.51) -> DataBatchFloat:
    """Normalizer for the Continuous Bernoulli distribution.
    
    This is numerically problematic around 0.5, so use a Taylor expansion around that value. Note that the value for 
    cut_prob where the condition is False is actually irrelevant since the result at those points is not used.

    Parameters:
        prob: Parameter of the distribution.
        lower_lim, upper_lim: Values in this range use the Taylor expansion.
    """
    numerical_condition = torch.logical_or(torch.less(prob, lower_lim), torch.greater(prob, upper_lim))
    cut_prob = torch.where(numerical_condition, prob, lower_lim * torch.ones_like(prob))
    log_normalizer = (torch.log(torch.abs(2 * torch.atanh(1 - 2 * cut_prob))) 
                      - torch.log(torch.abs(1 - 2 * cut_prob)))
    taylor_expansion = (torch.log(torch.tensor(2.)) + 4/3 * torch.pow(prob - 0.5, 2) 
                        + 104/45 * torch.pow(prob - 0.5, 4))
    return torch.where(numerical_condition, log_normalizer, taylor_expansion)


def continuous_bernoulli_expected_value(prob: DataBatchFloat,
                                        lower_lim: float = 0.49,
                                        upper_lim: float = 0.51) -> DataBatchFloat:
    """Returns expected value for a Continuous Bernoulli distribution with a given probability parameter.

    For the regular Bernoulli, this is just equal to the probability; but not here. This one is also unstable around
    0.5, and the authors seem to have settled for a "zeroeth-order" Taylor expansion.

    Parameters:
        prob: Parameter of the distribution.
        lower_lim, upper_lim: Values in this range use the Taylor expansion.
    """
    numerical_condition = torch.logical_or(torch.less(prob, lower_lim), torch.greater(prob, upper_lim))
    cut_prob = torch.where(numerical_condition, prob, lower_lim * torch.ones_like(prob))
    expected = (cut_prob / (2*cut_prob - 1) 
                + 1 / (2 * torch.atanh(1 - 2*cut_prob)))
    return torch.where(numerical_condition, expected, 0.5*torch.ones_like(expected))


def beta_loss(outputs: DataBatchFloat,
              targets: DataBatchFloat,
              safety_eps: float = 1e-3) -> ScalarFloat:
    """Corresponds to Beta likelihood.

    NOTE that there must be two outputs for each target, as the Beta distribution has two parameters. E.g. for a
    three-channel color image, outputs should have six channels. Also note that the Beta distribution is not defined
    for the limits 0/1, so we clip targets into the open interval.

    Parameters:
        safety_eps: target values below this are clipped to this value. 
                    target values above 1-eps are clipped to that.
              
    """
    targets = torch.clamp(targets, safety_eps, 1-safety_eps)
    alphas, betas = beta_parameters(outputs)
    nll = -torch.distributions.Beta(alphas, betas).log_prob(targets)
    nll = nll.view(nll.shape[0], -1)
    return sum_except(nll).mean()


def beta_parameters(outputs: DataBatchFloat,
                    positive_function: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.softplus)\
                        -> tuple[DataBatchFloat, DataBatchFloat]:
    """Turn network outputs into proper Beta distribution parameters.
    
    We do this by splitting along axis 1 (valid for Linear layers or channels-first convolutions) and applying a
    function that should have strictly positive outputs. Candidates are softplus (default) or exp (potentially
    unstable).
    """
    outputs = positive_function(outputs)
    n_params = outputs.shape[1]
    alphas, betas = torch.split(outputs, n_params // 2, dim=1)
    return alphas, betas


def fourier_loss(outputs: DataBatchFloat,
                 targets: DataBatchFloat) -> ScalarFloat:
    """This does not correspond to any likelihood I know of, so it's down here. :)"""
    output_fft = torch.fft.rfft2(outputs, norm="ortho")
    target_fft = torch.fft.rfft2(targets, norm="ortho")
    diff = torch.abs(output_fft - target_fft)
    return diff.mean()
