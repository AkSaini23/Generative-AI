from typing import TypeVar

import torch

from .glow import GlowFlow
from .nice import NiceFlow
from ..common import TrainerBase
from ..types import DataBatchFloat, LabelBatchFloat, ScalarFloat


FlowModel = TypeVar("FlowModel", NiceFlow, GlowFlow)


class FlowTrainer(TrainerBase[FlowModel]):
    def __init__(self,
                 use_bits_per_dim: bool = False,
                 **kwargs):
        """Trainer for Flow models, minimizing negative log-likelihood.
        
        Parameters:
            use_bits_per_dim: If True, use the average per-dimension log-probability, with base-2 logarithm. This can
                              make values more interpretable. Otherwise, just use the sum of natural log over all
                              dimensions.
            kwargs: Arguments to TrainerBase.
        """
        super().__init__(**kwargs)
        self.use_bits_per_dim = use_bits_per_dim
        if use_bits_per_dim:
            n_dims = self.model.total_dim
            log2_convert = torch.log(torch.tensor(2))
            self.loss_divisor = n_dims * log2_convert

    def core_step(self, 
                 data_batch: tuple[DataBatchFloat, LabelBatchFloat]) -> tuple[ScalarFloat, dict[str, ScalarFloat]]:
        input_batch, _ = data_batch
        input_batch = input_batch.to(self.device)
        nll = -self.model.log_p(input_batch).mean()
        if self.use_bits_per_dim:
            nll = nll / self.loss_divisor
        return nll, {"negative_log_likelihood": nll}

    def plot_examples(self,
                      epoch_ind: int | None = None):
        with torch.inference_mode():
            generated = self.model.generate(self.plot_n_rows**2)
        self.plot_generated_grid(generated, epoch_ind)
