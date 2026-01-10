from collections.abc import Callable

import numpy as np
import torch
from matplotlib import pyplot as plt

from .model import Generator
from ..common import sum_except
from ..types import DataBatchFloat, TabularBatchFloat


def optimization_inference(target_data: DataBatchFloat,
                           generator: Generator,
                           n_steps: int,
                           descale_data: Callable[[DataBatchFloat], DataBatchFloat] | None = None,
                           regularization: float = 0.) -> TabularBatchFloat:
    """Do inference via optimization for GANs.
    
    This means that given some target data, we try to find points in the latent space that, once generated from, match
    the target data as much as possible. This is not guaranteed to actually find good matches for all inputs, and in
    fact will often result in garbage.

    Parameters:
        target_data: Batch of inputs we want to approximate.
        generator: GAN generator.
        n_steps: Number of gradient steps we take for optimization.
        descale_data: Function that undoes scaling in case data was resacled for training. Pass None to use identity.
        regularization: There is no guarantee that the optimized noise actually represents good samples from the noise
                        function. If the true noise is standard normal, we can infer the expected norm of a sample.
                        Using this, we can quantify the difference between that norm and the actual norm of the
                        optimized noise, and use that as a regularizer. Note that this is only valid for standard normal
                        noise functions. Also, this can severely reduce the chance of finding a good match in the noise
                        space.
    """
    device = next(generator.parameters()).device
    optimized_noise = generator.noise_fn(target_data.shape[0]).to(device)
    optimized_noise.requires_grad = True
    noise_optimizer = torch.optim.Adam([optimized_noise], lr=1.)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(noise_optimizer, n_steps)

    noise_dim = torch.prod(torch.tensor(optimized_noise.shape[1:]))
    non_batch_data_dims = tuple(range(1, target_data.dim()))
    if descale_data is None:
        descale_data = lambda x: x

    for step in range(n_steps + 1):
        outputs = generator(optimized_noise)
        diff = (outputs - target_data).abs().mean(dim=non_batch_data_dims).sum()
        ideal_norm = torch.sqrt(noise_dim)
        actual_norms = torch.sqrt(sum_except(optimized_noise**2))
        diff = diff + regularization * ((ideal_norm - actual_norms)**2).mean()

        diff.backward(inputs=[optimized_noise])
        noise_optimizer.step()
        noise_optimizer.zero_grad()
        schedule.step()

        if not step % (n_steps // 10):
            plt.figure(figsize=(8,8))
            for ind, img in enumerate(descale_data(outputs).detach().cpu().numpy()[:32]):
                target_here = descale_data(target_data[ind]).cpu().numpy()
                combined = np.concatenate((img, target_here), axis=2)
                plt.subplot(8, 4, ind + 1)
                plt.imshow(combined.transpose(1,2,0), cmap="Greys", vmin=0, vmax=1)
                plt.axis("off")
            plt.suptitle(f"After Step {step + 1}")
            plt.show()
    return optimized_noise.detach()
