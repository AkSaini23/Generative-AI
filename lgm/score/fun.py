import torch

from .model import ScoreModel
from ..types import DataBatchFloat


def generate_omega(model: ScoreModel,
                   n_samples: int | None = None,
                   start_samples: DataBatchFloat = None,
                   mask = None,  # TODO typing? IndexImage or sth doesn't exist
                   start_sigma_index: int = 0,
                   return_sequence: bool = False,
                   denoise: bool = True,
                   steps_per_sigma: int | None = None,
                   epsilon: float | None = None) -> DataBatchFloat | list[DataBatchFloat]:
    """Swiss army knife of generator functions.

    Lots of code duplication with ScoreModel.generate; we could have integrated all of this in there. But to keep that
    function a bit more clean, we instead put it all in here.
    
    Parameters:
        start_samples: If given, used as starting points for generation OR masked inputs for inpainting.
                       In the first case, you likely want to pass partially diffused samples and also pass the
                       corresponding start_sigma_index. In the second case, these should be non-diffused samples and you
                       also should pass a mask. In either case, n_samples is ignored.
        mask: If given, do inpainting. At each step, the parts where the mask is 1 are replaced by noisy versions of
              start_samples (which also must be given).
        start_sigma_index: Start generation from this sigma. Should be used if you passed start_samples diffused to some
                           sigma. Then you can start generation at that index + 1.
        return_sequence: If True, returns the entire Markov chain (i.e. sequence of samples) instead of just the final
                         one.
        Other parameters: as in ScoreModel.generate.
    """
    if steps_per_sigma is None:
        steps_per_sigma = model.steps_per_sigma
    if epsilon is None:
        epsilon = model.epsilon
    device = next(model.parameters()).device

    if start_samples is None or mask is not None:
        if mask is not None:
            print("Mask detected -- inpainting mode!")
            if start_samples is None:
                raise ValueError("Inpainting requires start_samples to be given!")
            n_samples = start_samples.shape[0]
        else:
            print("No mask or start samples -- random generation mode!")
        sample = model.sigma_sequence[0] * torch.randn(n_samples, *model.input_shape, device=device)
    else:
        print("Start samples detected -- partial denoising mode!")
        sample = start_samples
    
    if return_sequence:
        full_chain = [sample]
    for sigma in model.sigma_sequence[start_sigma_index:]:
        alpha = epsilon * (sigma / model.sigma_sequence[-1])**2
        if mask is not None:
            noisy_real = start_samples + torch.randn_like(start_samples) * sigma
        for _ in range(steps_per_sigma):
            sample = model.langevin_step(sample, alpha, sigma)
            if mask is not None:
                sample = mask * noisy_real + (1 - mask) * sample
            if return_sequence:
                full_chain.append(sample)
    if denoise:
        # technically sigma**2 * conditional model output.
        # but if we condition the model to divide output by sigma, this partially cancels out.
        sample = sample + model.sigma_sequence[-1] * model(sample)
        if return_sequence:
            full_chain.append(sample)
        
    if return_sequence:
        return full_chain
    else:
        return sample
