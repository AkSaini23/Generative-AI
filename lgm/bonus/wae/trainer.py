import torch

from ...common import squared_distances
from ...types import DataBatchFloat, LabelBatchFloat, ScalarFloat, SquareMatrixFloat, TabularBatchFloat
from ...vae.legacy import VAEGauss, VAETrainer
WAE = VAEGauss


class WAETrainer(VAETrainer[WAE]):
    def __init__(self,
                 beta: float = 1.,
                 **kwargs):
        """Trainer for a (Gaussian) Wasserstein AE.
        
        Parameters:
            beta: Multiplier for the MMD added to the reconstruction loss. In practice it seems like this needs to be
                  absurdly high (100s of 1000s).
            Other parameters: Like in lgm.autoencoder.AETrainer.
        """
        super().__init__(**kwargs)
        self.beta = beta
    
    def core_step(self, 
                  data_batch: tuple[DataBatchFloat, LabelBatchFloat]) -> tuple[ScalarFloat, dict[str, ScalarFloat]]:
        """Core step computing reconstruction loss and KL-Divergence."""
        input_batch, _ = data_batch
        input_batch = input_batch.to(self.device)
        means, log_variances = self.model.encoding_parameters(input_batch)
        q_samples = self.model.sampler(means, log_variances)
        output_batch = self.model.decoder(q_samples)

        reconstruction_loss = self.reconstruction_loss_fn(output_batch, input_batch)
        mmd_loss = mmd(q_samples)
        full_loss = reconstruction_loss + self.beta * mmd_loss
        return full_loss, {"reconstruction_loss": reconstruction_loss, "mmd_loss": mmd_loss}


def mmd(q_samples: TabularBatchFloat) -> ScalarFloat:
    """Kernel-based Maximum Mean Discrepancy (MMD).
    
    Apparently it's important to exclude the diagonals for the self-comparisons. We do this by zeroing them out and
    adjusting the denominator for the mean accordingly.
    """
    p_samples = torch.randn_like(q_samples)
    k1 = kernel(q_samples, q_samples)
    k2 = kernel(p_samples, p_samples)
    k3 = kernel(p_samples, q_samples)
    k1_eyed = k1 * (1 - torch.eye(k1.shape[0], device=k1.device))
    k2_eyed = k2 * (1 - torch.eye(k2.shape[0], device=k2.device))
    denom = q_samples.shape[0] * (q_samples.shape[0] - 1)
    return k1_eyed.sum() / denom + k2_eyed.sum() / denom - 2*k3.mean()


def kernel(inp1: TabularBatchFloat,
           inp2: TabularBatchFloat) -> SquareMatrixFloat:
    """Inverse (multi?)quadratics kernel or something."""
    c = 2*inp1.shape[1]
    distance = squared_distances(inp1, inp2)
    return c / (c + distance)
