from collections.abc import Callable, Iterable

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from ..types import ImageBatchFloat


def plot_image_grid(images: ImageBatchFloat,
                    figure_size: tuple[int, int],
                    title: str,
                    n_rows: int,
                    n_cols: int | None = None,
                    subtitles: Iterable[str] | None = None,
                    colormap="Greys",
                    plot_descale: Callable[[ImageBatchFloat], ImageBatchFloat] | None = None,
                    writer: SummaryWriter | None = None,
                    epoch_ind: int | None = None,
                    tensorboard_figures: bool = False,
                    suppress_plots: bool = False):
    """Make a square grid from images.

    Parameters:
        images: Batch of images we want to plot. We assume that this has rows*cols many entries. If this is not true,
                this might crash. Sorry not sorry!
        figure_size: The size of the figure.
        title: Will be used as figure title as well as for naming Tensorboard summaries.
        n_rows: Will plot this many rows, and n_rows**2 many examples in total if n_cols is not igven.
        n_cols: Will plot this many columns of examples. Defaults to n_rows.
        subtitles: If given, should be an iterable of strings of the same length as images. Each string will be used as
                   title for the respective image's subplot.
        colormap: Which colormap to use to display images. Only used for single-channel images.
        Other arguments: Please see lgm.common.TrainerBase. Everything below writer is only used if that is not
                            None.
    """
    if n_cols is None:
            n_cols = n_rows
    with torch.inference_mode():
        if plot_descale is not None:
            images = plot_descale(images)
        images = np.clip(images.cpu().numpy(), 0, 1)

    plt.figure(figsize=figure_size)
    for ind, img in enumerate(images):
        plt.subplot(n_rows, n_cols, ind + 1)
        plt.imshow(img.transpose(1, 2, 0), vmin=0, vmax=1, cmap=colormap)
        plt.axis("off")
        if subtitles is not None:
             plt.title(subtitles[ind], fontsize=8)
    plt.suptitle(title)

    if writer is not None and tensorboard_figures and epoch_ind is not None:
        writer.add_figure(title, plt.gcf(), epoch_ind, close=suppress_plots)
    plt.show()
