from collections.abc import Callable
from typing import Generic, TypeVar

import torch

from .model import Discriminator, GAN
from ..common import TrainerBase
from ..types import DataBatchFloat, LabelBatchFloat, ScalarFloat


GANType = TypeVar("GANType", bound=GAN)


class GANTrainer(TrainerBase[GANType], Generic[GANType]):
    def __init__(self,
                 d_loss_fn: Callable[[DataBatchFloat, DataBatchFloat], ScalarFloat],
                 noise_scale: float = 0.,
                 label_smoothing: float = 1.,
                 feature_matching: str | None = None,
                 **kwargs):
        """Trainer for GANs.
        
        Due to the dual-network training of GANs, unfortunately we can't re-use quite as many components from the
        general trainer as usual. In particular, we implement separate train_step and eval_step functions. This class
        implements a somewhat optimized training step where the same generator batch is used for generator and
        discriminator training, and the two networks are trained "in parallel" rather than first updating the
        discriminator, and then the generator based on the new discriminator. Such a "sequential" training step would
        need an entirely separate Trainer class, since we would need two optimizers etc.

        Parameters:
            d_loss_fn: The loss function to use for the discriminator outputs. Classic examples are cross-entropy, or
                       squared error for LS-GAN.
            noise_scale: If a number > 0, we add Gaussian noise with that standard deviation onto data as well as
                         generated samples. This "de-quantizes" the data and fills in the gaps between the uint8 numbers
                         the data is usually stored in. Should help against the discriminator "winning" too easily.
            label_smoothing: Supply a number < 1 (e.g. 0.9) to apply one-sided label-smoothing, were 1 labels are
                             replaced by this number.  Supposed to prevent the discriminator from saturating.
            feature_matching: If given, apply feature matching loss to the generator. Heavily recommended. Note that
                              whether this is given or not, the code expects the discriminator to return a list of
                              hidden features. Options are 'sum' (sum over dimensions) or 'mean' (average over
                              dimensions).
        """
        super().__init__(**kwargs)
        self.d_loss_fn = d_loss_fn
        self.noise_scale = noise_scale
        self.label_smoothing = label_smoothing
        if feature_matching is not None and feature_matching not in ["sum", "mean"]:
            raise ValueError("If feature_matching is supplied, it must be one of 'sum' or 'mean'.")
        self.feature_matching = feature_matching

    def train_step(self,
                   data_batch: tuple[DataBatchFloat, LabelBatchFloat]) -> dict[str, ScalarFloat]:
        """GAN training step.

        Importantly, the discriminator loss is only propagated into the discriminator, and the same goes for the
        generator.
        
        Parameters:
            data_batch: A tuple of images, labels.
        """
        losses = self.core_step(data_batch)
        discriminator_loss = losses["discriminator_loss"]
        generator_loss = losses["generator_loss"]
        # retain graph needed so we can backward again for the generator
        discriminator_loss.backward(inputs=list(self.model.discriminator.parameters()), retain_graph=True)
        generator_loss.backward(inputs=list(self.model.generator.parameters()))
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.ema is not None:
            self.ema.update()
        if self.feature_matching is not None:
            # if using FM, the total loss is just the sum of the FM loss and the regular classification loss.
            # so, we discard it for the metrics.
            del losses["generator_loss"]
        return losses
    
    def eval_step(self,
                  data_batch: tuple[DataBatchFloat, LabelBatchFloat]) -> tuple[ScalarFloat, dict[str, ScalarFloat]]:
        """GAN eval step.
        
        I decided to use the discriminator loss as the "main" loss just because the interface expects something here.
        Do no use EarlyStopping or ReduceLROnPlateau with GANs! Then this doesn't matter.
        """
        with torch.inference_mode():
            losses = self.core_step(data_batch)
        discriminator_loss = losses["discriminator_loss"]
        if self.feature_matching is not None:  # see note in train_step
            del losses["generator_loss"]
        return discriminator_loss, losses

    def core_step(self,
                  data_batch: tuple[DataBatchFloat, LabelBatchFloat]) -> dict[str, ScalarFloat]:
        input_batch, _ = data_batch
        input_batch = input_batch.to(self.device)
        batch_dim = input_batch.shape[0]
        generated_batch = self.model.generator.generate(batch_dim)
        return self.core_step_generic(input_batch, generated_batch, self.model.discriminator)
    
    def core_step_generic(self,
                          real_batch: DataBatchFloat,
                          fake_batch: DataBatchFloat,
                          discriminator: Discriminator) -> dict[str, ScalarFloat]:
        """Factored out for later re-use.
        
        Implements a "parallel" GAN training where generator and discriminator are only called once each. As such, the
        discriminator is updated, and the generator will be updated according to the *old* discriminator, rather than
        the updated one. This should only cause the generator to be "behind" by a single traning step, while saving
        massive amounts of compute.
        """
        full_batch = torch.cat((real_batch, fake_batch), dim=0)
        if self.noise_scale > 0:
            noise = torch.randn_like(full_batch) * self.noise_scale
            full_batch = full_batch + noise
        discriminator_output, discriminator_features = discriminator(full_batch)

        batch_dim_real = real_batch.shape[0]
        batch_dim_fake = fake_batch.shape[0]
        real_labels = (torch.ones(batch_dim_real, *discriminator_output.shape[1:], device=self.device)
                       * self.label_smoothing)
        generated_labels = torch.zeros(batch_dim_fake, *discriminator_output.shape[1:], device=self.device)
        full_labels = torch.cat((real_labels, generated_labels), dim=0)
        discriminator_loss = self.d_loss_fn(discriminator_output, full_labels)

        fool_labels = (torch.ones(batch_dim_fake, *discriminator_output.shape[1:], device=self.device)
                       * self.label_smoothing)
        # re-use D output for generated data from above
        generator_loss = self.d_loss_fn(discriminator_output[batch_dim_real:], fool_labels)

        if self.feature_matching is not None:
            feature_match_loss = 0
            features_flat = [feature for level_feature in discriminator_features for feature in level_feature]
            for feature in features_flat:
                real_feature, fake_feature = torch.split(feature, batch_dim_real, dim=0)
                feature_difference = fake_feature.mean(dim=0) - real_feature.mean(dim=0)
                summary_fn = torch.sum if self.feature_matching == "sum" else torch.mean
                feature_match_loss += summary_fn(feature_difference**2) 
            generator_loss_full = generator_loss + feature_match_loss
        else:
            generator_loss_full = generator_loss

        loss_dict = {"discriminator_loss": discriminator_loss, "generator_loss": generator_loss_full}
        if self.feature_matching:
            loss_dict.update({"generator_loss_classification": generator_loss,
                              "feature_matching_loss": feature_match_loss})
        return loss_dict

    def plot_examples(self,
                      epoch_ind: int | None = None):
        with torch.inference_mode():
            generated = self.model.generator.generate(self.plot_n_rows**2)
        self.plot_generated_grid(generated, epoch_ind)
