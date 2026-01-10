import numpy as np
import torch
from torch import nn

from ..common import batched_multiply
from ..types import DataBatchFloat, ScalarFloat, VectorBatchFloat


class ScoreModel(nn.Module):
    def __init__(self,
                 root: nn.Module,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 head: nn.Module,
                 input_shape: tuple[int, ...],
                 sigma_sequence: np.ndarray,
                 steps_per_sigma: int,
                 epsilon: float):
        """Score matching generative model.
        
        Parameters:
            root: Initial module before the encoder, such as a single convolution.
            encoder: "Down" part of the U-Net. We expect this to be something like a CNNBody with return_all=True. This
                     is necessary for skip connections.
            decoder: "Up" part of the U-Net. Should take skip inputs that come in from the encoder.
            head: Final layer(s) transforming to the correct data shape.
            input_shape: Like so often, we have to remind the model what shape its input has for generation.
            sigma_sequence: Array-like containing the noise levels to run annealed Langevin dynamics over.
                            To stay consistent wth the notation in the paper, this should be a *descending* sequence,
                            i.e. the first entry should be the largest noise level, and the last entry the smallest.
            steps_per_sigma: How many steps to run Langevin sampling for at each noise level.
            epsilon: Step size for Langevin dynamics.
        """
        super().__init__()
        self.root = root
        self.encoder = encoder
        self.decoder = decoder
        self.head = head
        
        self.input_shape = input_shape
        self.register_buffer("sigma_sequence", torch.tensor(sigma_sequence, dtype=torch.float32))
        self.num_sigmas = len(sigma_sequence)
        self.steps_per_sigma = steps_per_sigma
        self.epsilon = epsilon

    def forward(self,
                inputs: DataBatchFloat) -> DataBatchFloat:
        """Applies the U-Net."""
        initial = self.root(inputs)
        encoder_final, encoder_all = self.encoder(initial)
        # gotta revert encoder outputs to serve as decoder skip inputs
        encoder_all = [[feature for feature in level[::-1]] for level in encoder_all[::-1]]
        # the output of the final block becomes the input for the first decoder block.
        # so it's pointless to add this as a skip connection, as well -> we remove it.
        encoder_all[0][0] = None
        decoded = self.decoder(encoder_final, encoder_all)
        return self.head(decoded)
    
    def noise_conditional_score(self,
                                inputs: DataBatchFloat,
                                sigma: VectorBatchFloat):
        """Compute noise-conditional score prediction.
        
        This is just the unconditional model output divided by sigma, as proposed in the improved paper. Note that in
        practice we can almost always avoid using this, as in most applications of the model there will be some other
        sigma-related terms that cancel out the division. But if you want to stick closer to the given formulas, feel
        free to use it.
        """
        raw = self(inputs)
        return batched_multiply(raw, 1/sigma)
    
    def generate(self,
                 n_samples: int,
                 denoise: bool = True,
                 steps_per_sigma: int | None = None,
                 epsilon: float | None = None) -> DataBatchFloat:
        """Generate a batch of data.
        
        Parameters:
            n_samples: How many independent samples to generate.
            denoise: If True, apply a final denoising step at the end.
            steps_per_sigma: Optionally override the model default.
            epsilon: Optionally override the model default.
        """
        if steps_per_sigma is None:
            steps_per_sigma = self.steps_per_sigma
        if epsilon is None:
            epsilon = self.epsilon
        device = next(self.parameters()).device
        #raise NotImplementedError("How should we initialize sample? It should follow a normal distribution with "
        #                          "whatever standard deviation we chose for the largest noise scale. Recall that "
        #                          "self.sigma_sequence exists, and this contains all noise levels in descending order. "
        #                          "Make sure to respect n_samples and self.input_shape.")
        sample = torch.randn(n_samples, *self.input_shape, device=device) * self.sigma_sequence[0]

        for sigma in self.sigma_sequence:
            #raise NotImplementedError("What should the step size alpha be? Both core papers are referenced in __init__. "
            #                          "You can technically check Algorithm 1 in either, BUT they use slightly different "
            #                          "notations for alpha! For consistency, you should stick to the definition from "
            #                          "the 'improved' version.")
            alpha = epsilon * (sigma ** 2 / self.sigma_sequence[-1]**2)
            for _ in range(steps_per_sigma):
                sample = self.langevin_step(sample, alpha, sigma)
        if denoise:
            # technically sigma**2 * conditional model output.
            # but if we condition the model to divide output by sigma, this partially cancels out.
            sample = sample + self.sigma_sequence[-1] * self(sample)
        return sample

    def langevin_step(self,
                      sample: DataBatchFloat,
                      alpha: ScalarFloat,
                      sigma: ScalarFloat) -> DataBatchFloat:
        """Take a single step for Langevin dynamics.
        
        Note that we could cancel out some terms with alpha and the model output being divided by sigma. But in this
        case, this would complicate the code too much, as we need the original alpha for the noise component.
        """
        #raise NotImplementedError("We are missing the part where the model is called. Note that this has to be a "
        #                          "'noise conditional' model that technically gets the noise level sigma as input. "
        #                          "But the improved paper proposes that the conditioning can be implemented by simply "
        #                          "dividing model outputs by sigma. Maybe you can find a useful function close-by... "
        #                          "And remember the correct step size!")
        score = self.noise_conditional_score(sample, torch.full((sample.shape[0],), sigma, device=sample.device))
        sample = sample + alpha * score + torch.sqrt(2 * alpha) * torch.randn_like(sample)
       
        return sample
