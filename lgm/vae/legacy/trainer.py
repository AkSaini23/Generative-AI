from typing import Generic, TypeVar

import torch

from .model import VAEGauss
from ...autoencoder import AETrainer
from ...common import sum_except
from ...types import AnyBatchFloat, DataBatchFloat, LabelBatchFloat, ScalarFloat


VAEType = TypeVar("VAEType", bound=VAEGauss)


class VAETrainer(AETrainer[VAEType], Generic[VAEType]):
    def __init__(self,
                 beta: float = 1.,
                 samples_per_q: int = 1,
                 **kwargs):
        """Trainer for a (Gaussian) VAE.
        
        Parameters:
            beta: Multiplier for the KL-divergence added to the reconstruction loss.
            samples_per_q: You could take more than one sample from each q(z|x) to better approximate the expected
                           values. Doesn't seem worth it in practice.
            Other parameters: Like in lgm.autoencoder.AETrainer.
        """
        super().__init__(**kwargs)
        self.kl_loss_fn = kl_loss_gauss
        self.beta = beta
        self.samples_per_q = samples_per_q
    
    def core_step(self, 
                  data_batch: tuple[DataBatchFloat, LabelBatchFloat]) -> tuple[ScalarFloat, dict[str, ScalarFloat]]:
        """Core step computing reconstruction loss and KL-Divergence."""
        input_batch, output_batch, means, log_variances = self.core_step_forward(data_batch)
        return self.core_step_loss(input_batch, output_batch, means, log_variances)
    
    def core_step_forward(self,
                          data_batch: tuple[DataBatchFloat, LabelBatchFloat])\
                            -> tuple[DataBatchFloat, DataBatchFloat, AnyBatchFloat, AnyBatchFloat]:
        """Input -> output part of the core step."""
        input_batch, _ = data_batch
        input_batch = input_batch.to(self.device)
        means, log_variances = self.model.encoding_parameters(input_batch)
        output_batch = self.model.decoder(self.model.sampler(means, log_variances, self.samples_per_q))
        return input_batch, output_batch, means, log_variances
    
    def core_step_loss(self,
                       input_batch: DataBatchFloat,
                       output_batch: DataBatchFloat,
                       means: AnyBatchFloat,
                       log_variances: AnyBatchFloat) -> tuple[ScalarFloat, dict[str, ScalarFloat]]:
        """Input + output -> loss part of the core step."""
        if self.samples_per_q > 1:
            input_batch = torch.repeat_interleave(input_batch, self.samples_per_q, dim=0)
        reconstruction_loss = self.reconstruction_loss_fn(output_batch, input_batch)
        kl_loss = self.kl_loss_fn(means, log_variances)
        full_loss = reconstruction_loss + self.beta * kl_loss
        return full_loss, {"reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}

    def plot_examples(self,
                      epoch_ind: int | None = None):
        """Plot AE reconstructions as well as images generated from the latent prior."""
        self.reconstruction_examples(epoch_ind)
        with torch.inference_mode():
            generated = self.autoencoder.generate(self.plot_n_rows**2)
        self.plot_generated_grid(generated, epoch_ind)


def kl_loss_gauss(means: DataBatchFloat,
                  log_variances: DataBatchFloat) -> ScalarFloat:
    """KL-Divergence for Gaussian posterior and Standard Gaussian prior. This can be solved in closed form."""
    return 0.5 * sum_except(means**2 - 1 + torch.exp(log_variances) - log_variances).mean()
