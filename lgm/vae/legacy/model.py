import torch
from torch import nn

from ...autoencoder import Autoencoder
from ...types import AnyBatchFloat, DataBatchFloat


class GaussSampler(nn.Module):
    def forward(self,
                means: DataBatchFloat,
                log_variances: DataBatchFloat,
                samples_per_q: int = 1) -> DataBatchFloat:
        """Reparameterized sampler for Gaussian distributions.

        The returned samples are drawn from the respective per-dimension distributions given mean and variance.
        """
        stddevs = torch.exp(0.5 * log_variances)

        if samples_per_q > 1:
            means = torch.repeat_interleave(means, samples_per_q, dim=0)
            log_variances = torch.repeat_interleave(log_variances, samples_per_q, dim=0)
        
        stddevs = torch.exp(0.5 * log_variances)
        return torch.randn_like(means) * stddevs + means


class VAEGauss(Autoencoder):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 likelihood: str,
                 prior_dim: tuple[int, ...]):
        """Wrapper for VAE with Gaussian prior/posterior.

        Parameters:
            See docs in vae.model.
        """
        super().__init__(encoder, decoder, likelihood)
        self.sampler = GaussSampler()
        self.prior_dim = prior_dim

    def forward(self,
                inputs: DataBatchFloat,
                map_to_expected: bool = True) -> DataBatchFloat:
        reconstructions = self.decoder(self.sampler(*self.encoding_parameters(inputs)))
        if map_to_expected:
            reconstructions = self.map_likelihood(reconstructions)
        return reconstructions

    def encoding_parameters(self,
                            inputs: DataBatchFloat) -> tuple[AnyBatchFloat, AnyBatchFloat]:
        """Split encoder outputs into separate means and log variances."""
        parameters = self.encoder(inputs)
        split_size = parameters.shape[1] // 2
        mean, log_variance = torch.split(parameters, split_size, dim=1)
        return mean, log_variance

    def generate(self, 
                 n_samples: int) -> DataBatchFloat:
        device = next(self.parameters()).device
        codes = torch.randn(n_samples, *self.prior_dim, device=device)
        return self.map_likelihood(self.decoder(codes))
