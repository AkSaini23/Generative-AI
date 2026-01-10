import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms.v2 import (ColorJitter, GaussianNoise, InterpolationMode, RandAugment, RandomAffine,
                                       Transform)

from ..common import TrainerBase
from ..types import (DataBatchFloat, IndexSequenceBatch, LabelBatchFloat, ScalarFloat, SequenceBatchFloat,
                     TabularBatchFloat, VectorBatchFloat)


class ClassifierTrainer(TrainerBase):
    def __init__(self,
                 label_smoothing: float = 0.,
                 classes: list[str] | None = None,
                 **kwargs):
        """Trainer class for classifiers.

        Parameters:
            label_smoothing: If given a float > 0, apply that amount of label smoothing. Not recommended if you use
                             cutmix/mixup augmentations.
            classes: If given, use these class names for the plot_examples function. Otherwise, classes will just be
                     numbered starting from 0.
        """
        super().__init__(**kwargs)
        self.classes = classes
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def core_step(self,
                  data_batch: tuple[DataBatchFloat, LabelBatchFloat]) -> tuple[ScalarFloat, dict[str, ScalarFloat]]:
        """Standard classifier step. Computes cross-entropy and accuracy."""
        input_batch, label_batch = data_batch
        input_batch = input_batch.to(self.device)
        label_batch = label_batch.to(self.device)
        logits = self.model(input_batch)
        loss = self.loss_fn(logits, label_batch)
        with torch.inference_mode():
            batch_accuracy = accuracy(logits, label_batch)
        return loss, {"cross_entropy": loss, "accuracy": batch_accuracy}

    def plot_examples(self,
                      epoch_ind: int | None = None):
        """Show classification example results.
        
        For each image, shows the true class and argmax prediction, with associated probability.
        """
        inputs, y = next(iter(self.validation_loader))
        inputs = inputs[:self.plot_n_rows**2]
        y = y[:self.plot_n_rows**2]
        with torch.inference_mode():
            probabilities = torch.nn.functional.softmax(self.model(inputs.to(self.device)), dim=1).cpu()
            predictions = probabilities.argmax(axis=1)
        max_class = probabilities.shape[1]
        classes = list(range(max_class)) if self.classes is None else self.classes

        subtitles = []
        for ind in range(inputs.shape[0]):
            pred_here = predictions[ind]
            prob_here = probabilities[ind, pred_here].item()
            true_here = y[ind]
            subtitles.append(f"true: {classes[true_here]} pred: {classes[pred_here]}\nprob: {prob_here:.3f}")
        self.plot_generated_grid(inputs, epoch_ind, title="Classifications", subtitles=subtitles)


def accuracy(outputs: TabularBatchFloat,
             labels: VectorBatchFloat | TabularBatchFloat) -> ScalarFloat:
    """Accuracy for multiclass (softmax) classification.
    
    Labels can be given in sparse (index) format, or as one-hot. Does not work for multilabel classification (i.e.
    sigmoid outputs). Outputs must be 2D, batch x classes.
    """
    return top_k_accuracy(outputs, labels, k=1)


def top_k_accuracy(outputs: TabularBatchFloat,
                   labels: VectorBatchFloat | TabularBatchFloat,
                   k: int) -> ScalarFloat:
    """Top-k accuracy. Counts a hit if one of the k largest outputs matches the label."""
    top_k_predictions = torch.topk(outputs, k, dim=1, sorted=False).indices
    if labels.ndim == 2:
        labels = labels.argmax(dim=1)
    labels = labels[:, None]
    matches = torch.any(labels == top_k_predictions, dim=1)
    return matches.float().mean()


def accuracy_sequence(outputs: SequenceBatchFloat,
                      labels: IndexSequenceBatch) -> ScalarFloat:
    """Accuracy for multiclass (softmax) classification for sequences.
    
    Labels must be given in sparse (index) format. One-hot labels currently not supported. Does not work for multilabel
    classification (i.e. sigmoid outputs).
    """
    return top_k_accuracy_sequence(outputs, labels, k=1)


def top_k_accuracy_sequence(outputs: DataBatchFloat,
                            labels: IndexSequenceBatch,
                            k: int) -> ScalarFloat:
    """Top-k accuracy for sequence inputs."""
    top_k_predictions = torch.topk(outputs, k, dim=1, sorted=False).indices
    labels = labels[:, None]
    matches = torch.any(labels == top_k_predictions, dim=1)
    return matches.float().mean()


def augmentations(dataset: str,
                  noise_scale: float = 0.) -> list[Transform]:
    """Wrapper to provide some data augmentations for training classifiers.
    
    These can then be provided to the get_datasets_and_loaders function to apply them to the training data. We use
    basic affine transformations for greyscale datasets. For color data, we use the more powerful RandAugment instead.
    We can also add a little bit of noise onto the data. This might make the classifiers slightly less sensitive to
    tiny (invisible) amounts of noise in generated data.
    """
    rotation = 15
    scale = 0.15
    translate = 0.1
    shear = 10
    if dataset not in ["mnist", "fashion", "emnist"]:  # no color for grayscale images
        # we try RandAugment. Alternative would be to use generic augmentations with color jitter
        transforms = [RandAugment(interpolation=InterpolationMode.BILINEAR)]
    else:
        brightness = contrast = saturation = hue = None
        transforms = augmentations_generic(rotation, scale, translate, shear,
                                           brightness, contrast, saturation, hue)
    if noise_scale > 0.:
        transforms.append(GaussianNoise(sigma=noise_scale))
    return transforms


def augmentations_generic(rotation: int | None,
                          scale: float | None,
                          translate: float | None,
                          shear: int | None,
                          brightness: float | None = None,
                          contrast: float | None = None,
                          saturation: float | None = None,
                          hue: float | None = None) -> list[Transform]:
    """Applies affine transforms and color jitter.
    
    For parameters, see the respective functions.
    """
    if scale is not None:
        scale = (1 - scale, 1 + scale)
    if translate is not None:
        translate = (translate, translate)
    if shear is not None:
        shear = (-shear, shear, -shear, shear)

    transforms = []
    if rotation is not None or scale is not None or translate is not None or shear is not None:
        transforms.append(RandomAffine(degrees=rotation, translate=translate, scale=scale, shear=shear,
                                       interpolation=InterpolationMode.BILINEAR))
    if brightness is not None or contrast is not None or saturation is not None or hue is not None:
        transforms.append(ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue))
    return transforms
