from collections.abc import Callable, Iterable

import numpy as np
import torch
from IPython.display import Audio, display
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchaudio.transforms import MelSpectrogram

from ..types import SequenceBatchFloat


def plot_audio_grid(audios: SequenceBatchFloat,
                    figure_size: tuple[int, int],
                    title: str,
                    n_rows: int,
                    n_cols: int | None = None,
                    n_audios: int | None = None,
                    subtitles: Iterable[str] | None = None,
                    colormap="magma",
                    plot_descale: Callable[[SequenceBatchFloat], SequenceBatchFloat] | None = None,
                    plot_waveforms: bool = True,
                    plot_spectrograms: bool = True,
                    play_audio: bool = True,
                    sampling_rate: int | None = None,
                    n_fft: int = 1024,
                    writer: SummaryWriter | None = None,
                    epoch_ind: int | None = None,
                    tensorboard_figures: bool = False,
                    suppress_plots: bool = False):
    """Display generated audio: Waveforms, spectrograms and/or audio widgets.

    NOTE Due to the large amount of plots, we heavily recommend turning down n_rows/cols compared to usual image plots.
    TODO write audio to tensorboard.

    Parameters:
        audios: Batch of audio. Should be 1d waveforms with size-1 channel axis, i.e. we assume mono audio.
                Stereo should also work, but we will only display waveforms/spectrograms for the first channel.
                We assume that this has n_rows*n_cols entries.
        title: Will be used as part of figure titles as well as for naming Tensorboard summaries.
        figure_size: The size of the figures.
        n_rows: Will plot this many rows, and n_rows**2 many examples in total if n_cols is not igven.
        n_cols: Will plot this many columns of examples. Defaults to n_rows.
        n_audios: Will plot this many audios. If not given, default to n_rows * n_cols.
        subtitles: If given, should be an iterable of strings of the same length as audios. Each string will be used as
                   title for the respective example's subplot.
        colormap: Which colormap to use to display spectrograms, if those are plotted.
        plot_waveforms: If True, display example waveforms.
        plot_spectrograms: If True, display example spectrograms.
        play_audio: If True, display audio widgets. Keep n_rows very small if using this! Will make your notebooks
                    very big.
        sampling_rate: Sampling rate of the audio. Needed for correct spectrogram plots and audio playback.
        n_fft: Window size for spectrograms.
        Other arguments (also plot_descale): Please see lgm.common.TrainerBase. Everything below writer is only used
                        if that is not None.
    """
    if n_cols is None:
        n_cols = n_rows
    with torch.inference_mode():
        if plot_descale is not None:
            audios = plot_descale(audios)
        audios = audios[:, 0].cpu()

    if plot_waveforms:
        plt.figure(figsize=figure_size)
        for ind, wave in enumerate(audios.numpy()):
            plt.subplot(n_rows, n_cols, ind+1)
            plt.plot(wave)
            plt.axis("off")
            if subtitles is not None:
                plt.title(subtitles[ind], fontsize=8)
        plt.suptitle(f"Waveform {title}")
        plt.show()

    # TODO axis labels for spectrograms?
    if plot_spectrograms:
        to_spectro = MelSpectrogram(sample_rate=sampling_rate, n_fft=n_fft, normalized=True)
        spectrograms = to_spectro(audios)
        plt.figure(figsize=figure_size)
        for ind, spectro in enumerate(spectrograms.numpy()):
            plt.subplot(n_rows, n_cols, ind+1)
            plt.pcolormesh(np.log(spectro + 1), cmap=colormap)
            plt.axis("off")
            if subtitles is not None:
                plt.title(subtitles[ind], fontsize=8)
        plt.suptitle(f"Mel Spectrogram {title}")
        plt.show()

    if play_audio:
        if n_audios is None:
            n_audios = n_rows * n_cols
        for wave in audios.numpy()[:n_audios]:
            if subtitles is not None:
                print(subtitles[ind])
            display(Audio(wave, rate=sampling_rate))
