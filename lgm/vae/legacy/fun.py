import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from .model import VAEGauss
from ...types import DataBatchFloat, LabelBatchFloat, TabularBatchFloat


def show_latent_distributions(model: VAEGauss,
                              dataloader: DataLoader[tuple[DataBatchFloat, LabelBatchFloat]],
                              n_rows: int,
                              figsize: tuple[int, int] = (12, 12)):
    """Display histograms of encoder output values over a dataset.
    
    Parameters:
        model: Any VAE-type model that supports the respective encoding/sampling functions.
        dataloader: Provides the dataset to get distributions for.
        n_rows: How many rows to plot. You are responsible for making sure this divideds the number of dimensions.
        figsize: Total size of the plot.
    """
    device = next(model.parameters()).device
    all_codes = []
    with torch.inference_mode():
        for input_batch, _ in dataloader:
            batch_encoded = model.sampler(*model.encoding_parameters(input_batch.to(device)))
            all_codes.append(batch_encoded)
    all_codes = torch.cat(all_codes, dim=0)

    if all_codes.dim() != 2:
        raise ValueError("Codes must be 1D vectors.")
    code_dim = all_codes.shape[1]
    n_cols = code_dim // n_rows
    plt.figure(figsize=figsize)
    for ind, codes_dim in enumerate(all_codes.T):
        plt.subplot(n_rows, n_cols, ind+1)
        plt.hist(codes_dim.cpu().numpy(), bins=100)
    plt.show()


def check_code_usage(model: VAEGauss,
                     dataloader: DataLoader[tuple[DataBatchFloat, LabelBatchFloat]]) -> TabularBatchFloat:
    """For each code dimension, see how much it diverges from a standard normal.
    
    This is a proxy for how much information this dimension contains. A code dimension that is always near standard
    normal for all inputs (indicated by low KL) contains little to no information about what the input was. The larger
    the average divergence, the more informative this dimension tends to be. This is basically a more formal version of
    our very fun per_dim_latent_walk.

    Parameters:
        model: VAE-like model that returns Gaussian parameters.
        dataloader: Provides the dataset to get distributions for.
    
    Return:
        Matrix showing KL-divergence for each input in the dataloader and each dimension.
    """
    def kl_loss_gauss_per_dim(means: DataBatchFloat,
                            log_variances: DataBatchFloat) -> DataBatchFloat:
        means = means.view(means.shape[0], -1)
        log_variances = log_variances.view(log_variances.shape[0], -1)

        return 0.5 * (means**2 - 1 + torch.exp(log_variances) - log_variances)
    
    device = next(model.parameters()).device
    all_klds = []
    with torch.inference_mode():
        for input_batch, _ in dataloader:
            means, logvars = model.encoding_parameters(input_batch.to(device))
            all_klds.append(kl_loss_gauss_per_dim(means, logvars))
    return torch.cat(all_klds, dim=0)
