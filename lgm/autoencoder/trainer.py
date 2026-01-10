from collections.abc import Callable
from typing import  Generic, TypeVar

import torch

from .model import Autoencoder
from ..common import TrainerBase, interleave
from ..types import DataBatchFloat, LabelBatchFloat, ScalarFloat


AEType = TypeVar("AEType", bound=Autoencoder)


class AETrainer(TrainerBase[AEType], Generic[AEType]):
    def __init__(self,
                 reconstruction_loss_fn: Callable[[DataBatchFloat, DataBatchFloat], ScalarFloat] | None = None,
                 **kwargs):
        """Trainer for a standard autoencoder.
        
        Parameters:
            reconstruction_loss_fn: If desired, you can overwrite the likelihood-based loss here.
        """
        super().__init__(**kwargs)
        if reconstruction_loss_fn is None:
            reconstruction_loss_fn = self.model.loss_likelihood()
        self.reconstruction_loss_fn = reconstruction_loss_fn
        # this may seem pointless, but having this name available can reduce code duplication in some subclasses
        self.autoencoder = self.model
    
    def core_step(self, 
                  data_batch: tuple[DataBatchFloat, LabelBatchFloat]) -> tuple[ScalarFloat, dict[str, ScalarFloat]]:
        """Compute reconstructions and corresponding loss.

        Parameters:
            data_batch: Expected to be a *tuple* of inputs, labels. The labels are ignored! But we keep them since we
                        may later want to implement conditional generative models, and this way we can keep using the
                        same datasets. Still, this means your datasets should include labels!
        """
        input_batch, _ = data_batch
        input_batch = input_batch.to(self.device)
        output_batch = self.model(input_batch)
        reconstruction_loss = self.reconstruction_loss_fn(output_batch, input_batch)
        return reconstruction_loss, {"reconstruction_loss": reconstruction_loss}

    def plot_examples(self,
                      epoch_ind: int | None = None):
        self.reconstruction_examples(epoch_ind)

    def reconstruction_examples(self,
                                epoch_ind: int | None = None):
        """Plot reconstructions for some validation images."""
        inputs, _ =  next(iter(self.validation_loader))
        inputs = inputs[:self.plot_n_rows**2 // 2].to(self.device)
        with torch.inference_mode():
            reconstructions = self.autoencoder(inputs, map_to_expected=True)
            self.reconstruction_grid(inputs, reconstructions, epoch_ind)

    def reconstruction_grid(self,
                            inputs: DataBatchFloat,
                            reconstructions: DataBatchFloat,
                            epoch_ind: int | None = None):
        alternating = interleave(inputs, reconstructions)
        self.plot_generated_grid(alternating, title="Reconstructions", epoch_ind=epoch_ind)
