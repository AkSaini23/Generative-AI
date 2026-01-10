from collections.abc import Callable

import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio import datasets
from torchvision.transforms.v2 import Compose, Transform

from .transforms import PadToFixed, PhaseFlip
from ..types import DataBatchFloat, DataFloat, LabelBatchFloat, LabelFloat, SequenceBatchFloat, SequenceFloat
from ..visualization import plot_audio_grid


SAMPLING_RATE = {"speechcommands": 16000}
N_CLASSES = {"speechcommands": 35}

SPEECH_COMMANDS_LABELS = {'zero': 0, 'five': 1, 'yes': 2, 'seven': 3, 'nine': 4, 'one': 5, 'down': 6, 'no': 7,
                          'stop': 8, 'two': 9, 'go': 10, 'six': 11, 'on': 12, 'left': 13, 'eight': 14, 'right': 15,
                          'off': 16, 'three': 17, 'four': 18, 'up': 19, 'house': 20, 'wow': 21, 'dog': 22, 'marvin': 23,
                          'bird': 24, 'cat': 25, 'happy': 26, 'sheila': 27, 'bed': 28, 'tree': 29, 'backward': 30,
                          'visual': 31, 'learn': 32, 'follow': 33, 'forward': 34}


def get_datasets_and_loaders(dataset: str,
                             batch_size: int,
                             num_workers: int = 0,
                             root: str = "data",
                             augment_transforms: list[Transform] | None = None,
                             additional_transforms: list[Transform] | None = None,
                             verbose: bool = True,
                             plot_descale: Callable[[DataBatchFloat], DataBatchFloat] | None = None) \
                                -> tuple[Dataset,
                                         Dataset,
                                         DataLoader[tuple[DataBatchFloat, LabelBatchFloat]],
                                         DataLoader[tuple[DataBatchFloat, LabelBatchFloat]]]:
    """NOTE!!! THIS IS ALL VERY EXPERIMENTAL AND SUBJECT TO CHANGE AND BREAK NOTE!!!

    Parameters:
        dataset: Name of the dataset. Only 'speechcommands' is allowed.
                 speechcommands: The single audio dataset we have implemented right now. Collection of short clips of
                                 people saying single-word commands like "stop", "go", "left", "one", etc. Clips were
                                 crowdsourced and are often very low quality. 16 kHz sampling rate. This dataset is not
                                 intended for generative modeling, but can serve as a simplistic example to get started
                                 with audio datasets.
        batch_size: Guess what!
        num_workers: Used by DataLoader.
        root: Base path where datasets should be stored/looked for.
        augment_transforms: Transforms you want to be applied only to the training data, i.e. data augmentation.
                            Note that most augmentations negatively affect data quality, and would make your
                            generative model learn to generate such data. As such, this is only really useful for
                            training classifiers for evaluation.
        additional_transforms: Any transforms you want on BOTH training and testing data besides standard ones.
                               For example, re-scaling data to some other range from [-1, 1].
        verbose: If True, print some info about the dataset elements (shape and dtype), as well as plotting some example
                 data.
        plot_descale: If you have supplied additional transforms that change the scale of the data, you likely want to
                      undo that (scale back to [-1, 1]) before plotting. This function should do that. Only needed if
                      verbose is True.
    """
    phase_flip = PhaseFlip()
    train_transforms = [phase_flip]
    test_transforms = [phase_flip]
    if dataset == "speechcommands":
        pad_crop = PadToFixed(16384)
        train_transforms.append(pad_crop)
        test_transforms.append(pad_crop)

    if augment_transforms is not None:
        train_transforms += augment_transforms
    if additional_transforms is not None:
        train_transforms += additional_transforms
        test_transforms += additional_transforms
    train_transforms = Compose(train_transforms)
    test_transforms = Compose(test_transforms)

    if dataset == "speechcommands":
        train_data = datasets.SPEECHCOMMANDS(root=root, subset="training", download=True)
        test_data = datasets.SPEECHCOMMANDS(root=root, subset="testing", download=True)

        def collate_fn_train(sample_list: list[tuple[SequenceFloat, int, str, str, int]])\
            -> tuple[SequenceBatchFloat, LabelBatchFloat]:
            audios, _, labels, _, _ = zip(*sample_list)
            audios = [train_transforms(audio) for audio in audios]
            labels = [SPEECH_COMMANDS_LABELS[label] for label in labels]
            audios = torch.stack(audios)
            labels = torch.tensor(labels, dtype=torch.int64)
            return audios, labels
        
        def collate_fn_test(sample_list: list[tuple[SequenceFloat, int, str, str, int]])\
            -> tuple[SequenceBatchFloat, LabelBatchFloat]:
            audios, _, labels, _, _ = zip(*sample_list)
            audios = [test_transforms(audio) for audio in audios]
            labels = [SPEECH_COMMANDS_LABELS[label] for label in labels]
            audios = torch.stack(audios)
            labels = torch.tensor(labels, dtype=torch.int64)
            return audios, labels
        
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  drop_last=True, num_workers=num_workers, collate_fn=collate_fn_train)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, pin_memory=True,
                                 num_workers=num_workers, collate_fn=collate_fn_test)
    if verbose:
        audios, y = next(iter(train_dataloader))
        print(f"Shape/dtype of batch X [N, C, T]: {audios.shape}, {audios.dtype}")
        print(f"Shape/dtype of batch y: {y.shape}, {y.dtype}")
        plot_audio_grid(audios[:32], figure_size=(14, 7), title="Examples", n_rows=4, n_cols=8, n_audios=4,
                        plot_descale=plot_descale, sampling_rate=SAMPLING_RATE[dataset])

    return train_data, test_data, train_dataloader, test_dataloader
