import os
from collections.abc import Callable
from PIL import Image

import numpy as np
import torch
from matplotlib import pyplot as plt

from ..types import DataBatchFloat, DataFloat, ImageBatchFloat, TabularBatchFloat


def interpolate(start: DataFloat,
                end: DataFloat,
                n_interpolations: int,
                style: str = "slerp") -> DataBatchFloat:
    """Compute a series of interpolations between two points.
    
    Parameters:
        start: Starting point for interpolation. Currently does not support batches, i.e. start and end have to be
               single data points.
        end: End point for interpolation.
        n_interpolations: Will generate this many interpolations, including start and end.
        style: One of 'slerp' or 'linear'. Interpolation style. 'linear' simply interpolates in a straight line.
               'slerp' goes in a circular arc, which can be more appropriate for the unintuitive geometry of high-
               dimensional spaces. This is particularly true when interpolating in latent spaces that should be
               distributed according to normal or uniform distributions, for example.
    """
    coefficients = torch.linspace(0, 1, n_interpolations)
    interpolations = []
    for coeff in coefficients:
        if style == "linear":
            interpolations.append(coeff*end + (1 - coeff)*start)
        elif style == "slerp":
            angle = torch.acos(torch.dot(start.view(-1), end.view(-1))
                               / (torch.linalg.vector_norm(start)*torch.linalg.vector_norm(end)))
            interpolations.append(torch.sin((1 - coeff) * angle) / torch.sin(angle) * start 
                                  + torch.sin(coeff * angle) / torch.sin(angle) * end)
        else:
            raise ValueError(f"Invalid style {style}. Allowed are 'slerp', 'linear'.")
    return torch.stack(interpolations)


def make_animation(folder: str,
                   image_series: ImageBatchFloat,
                   frame_rate: int = 10,
                   format: str = "webp"):
    """Create a GIF/WEBP from a batch of images.

    For color images WEBP is preferred, as the GIF format is very limited with respect to how many colors it can
    display.
    
    Parameters:
        folder: Frame images and the final animation will be stored here.
        image_series: Batched image tensor. Each row will become one frame. These should be scaled/clipped to range [0, 1].
        frame_rate: Number of images per second.
    """
    if format not in ["gif", "webp"]:
        raise ValueError("Invalid format{}. Allowed are 'gif', 'web'")
    frame_duration = 1000 // frame_rate
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    image_series = image_series.detach().cpu().numpy().transpose(0, 2, 3, 1)
    if image_series.shape[-1] == 1:  # grayscale image
        image_series = image_series.repeat(3, axis=-1)
    image_series = (image_series * 255).astype(np.uint8)
    pil_images = [Image.fromarray(image, mode="RGB") for image in image_series]
    pil_images[0].save(os.path.join(folder, f"interpolation.{format}"), save_all=True, append_images=pil_images[1:],
                       duration=frame_duration, loop=0)
    

def per_dim_latent_walk(generator: Callable[[TabularBatchFloat], DataBatchFloat],
                        anchor_code: TabularBatchFloat,
                        range_lim: float,
                        threshold: float) -> list[float]:
    """Fix all but one dimension of the latent space and "walk" along that one.
    
    Dimensions where a meaningful difference has occured are plotted.

    Parameters:
        generator: Despite the name, any callable (so can also be an nn.Module) that takes a latent code input and
                   returns images. So for example a VAE decoder + map_likelihood chain, or a GAN generaor object.
        anchor_code: The "base code" we walk from. This must be on the correct device for the generator already! Since
                     we can't guarantee that generator is a Module, we cannot rely on inferring the device from it.
        threshold: We compute the average (i.e. per-pixel) absolute difference between the start and end points of the
                   walk. If that is above this threshold, we count that as a "used" dimension, else we skip it. The
                   optimal value likely depends on the dataset.
    """
    n_steps = 64  # hardcoded, what are you going to do about it?
    # TODO don't hardcode that
    if anchor_code.dim() != 1:
        raise ValueError("anchor_code must be 1d vector.")
    differences = []
    for latent_dim in range(anchor_code.shape[0]):
        code_repeated = torch.tile(anchor_code, [n_steps, 1])
        value_range = torch.linspace(-range_lim, range_lim, n_steps)
        for dim in range(len(code_repeated)):
            code_repeated[dim, latent_dim] = value_range[dim]

        with torch.inference_mode():
            interpolated_images = generator(code_repeated).cpu().numpy()
        avg_abs_diff = (np.abs(interpolated_images[0] - interpolated_images[-1]).sum()
                        / np.prod(interpolated_images[0].shape))
        differences.append(avg_abs_diff)

        if avg_abs_diff < threshold:
            print(f"Dim {latent_dim} not used")
        else:
            print(avg_abs_diff)
            plt.figure(figsize=(5, 5))
            for ind in range(n_steps):
                plt.subplot(int(np.sqrt(n_steps)), int(np.sqrt(n_steps)), ind+1)
                plt.imshow(interpolated_images[ind].transpose(1, 2, 0), vmin=0, vmax=1, cmap="Greys")
                plt.axis("off")
            plt.show()
    return differences
