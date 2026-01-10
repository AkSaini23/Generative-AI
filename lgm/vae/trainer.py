import torch

from .legacy import VAETrainer
from .model import ConditionalVAEGauss
from ..types import AnyBatchFloat, DataBatchFloat, LabelBatchFloat


class ConditionalVAETrainer(VAETrainer[ConditionalVAEGauss]):
    def core_step_forward(self,
                          data_batch: tuple[DataBatchFloat, LabelBatchFloat])\
                            -> tuple[DataBatchFloat, DataBatchFloat, AnyBatchFloat, AnyBatchFloat]:
        """Input -> output part of the core step."""
        input_batch, label_batch = data_batch
        input_batch = input_batch.to(self.device)
        if self.model.is_conditional:
            label_batch = label_batch.to(self.device)
            embedded_conditioning = self.model.conditioning_net(label_batch)
        else:
            embedded_conditioning = None

        means, log_variances = self.model.encoding_parameters(input_batch, embedded_conditioning)
        if self.samples_per_q > 1:
            embedded_conditioning = torch.repeat_interleave(embedded_conditioning, self.samples_per_q, dim=0)
        output_batch = self.model.decoder(self.model.sampler(means, log_variances, self.samples_per_q),
                                          embedded_conditioning)

        return input_batch, output_batch, means, log_variances

    def plot_examples(self,
                      epoch_ind: int | None = None):
        """Plot AE reconstructions as well as images generated from the latent prior."""
        self.reconstruction_examples()
        
        if self.model.is_conditional:
            n_per_class = self.plot_n_rows**2 // self.model.n_classes
            classes = torch.repeat_interleave(torch.arange(self.model.n_classes), n_per_class)
        with torch.inference_mode():
            if self.model.is_conditional:
                generated = self.model.generate(conditioning=classes)
            else:
                generated = self.model.generate(n_samples=self.plot_n_rows**2)
        self.plot_generated_grid(generated, epoch_ind)

    def reconstruction_examples(self,
                                epoch_ind: int | None = None):
        """Shows a few images and their reconstructions by the autoencoder."""
        inputs, labels =  next(iter(self.validation_loader))
        with torch.inference_mode():
            classes = labels.to(self.device) if self.model.is_conditional else None
            reconstructions = self.model(inputs.to(self.device), classes, map_to_expected=True)
            self.reconstruction_grid(inputs, reconstructions, epoch_ind)
