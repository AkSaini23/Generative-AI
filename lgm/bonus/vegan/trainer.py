from collections.abc import Callable

import torch

from .model import VEGAN
from ...gan import GANTrainer
from ...vae.legacy import VAETrainer, kl_loss_gauss
from ...types import DataBatchFloat, LabelBatchFloat, ScalarFloat


class VEGANTrainer(VAETrainer[VEGAN], GANTrainer[VEGAN]):
    def __init__(self,
                 reconstruction_loss_fn:  Callable[[DataBatchFloat, DataBatchFloat], ScalarFloat] | None = None,
                 beta: float = 1.,
                 data_gan_loss_scale: float = 1.,
                 latent_gan_loss_scale: float = 1.,
                 **kwargs):
        """Trainer for a (Gaussian) VAE with auxilliary GAN losses.
        
        Parameters:
            gan_loss_scale: Multiplier for "data space GAN" compared to other losses. Pass 0 to turn this off.
            latent_gan_loss_scale: Multiplier for "latent space GAN" compared to other losses. Pass 0 to turn this off.
            beta, reconstruction_loss_fn: Like for VAETrainer.
            kwargs: Arguments to TrainerBase *and* GANTrainer.
        """
        GANTrainer.__init__(self, **kwargs)
        # can't call VAETrainer.__init__ because of multiple inheritance issues...
        if reconstruction_loss_fn is None:
            reconstruction_loss_fn = self.model.vae.loss_likelihood()
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.kl_loss_fn = kl_loss_gauss
        self.beta = beta
        self.samples_per_q = 1  # not dealing with this
        self.autoencoder = self.model.vae

        self.data_gan_loss_scale = data_gan_loss_scale
        self.latent_gan_loss_scale = latent_gan_loss_scale

    def train_step(self,
                   data_batch: tuple[DataBatchFloat, LabelBatchFloat]) -> dict[str, ScalarFloat]:
        """VEGAN training step.

        We have the standard variational autoencoder. On top of that, the encoder doubles as a generator for a "latent
        space GAN", and the decoder is the generator for a "data space GAN". We have discriminators for both spaces.
        
        Parameters:
            data_batch: A tuple of images, labels.
        """
        losses = self.core_step(data_batch)
        discriminator_loss = losses["data_discriminator_loss"]
        generator_loss = losses["vae_loss"]
        latent_d_loss = losses["latent_discriminator_loss"]
        # retain graph needed so we can backward again for the generator
        if self.data_gan_loss_scale > 0.:
            discriminator_loss.backward(inputs=list(self.model.data_discriminator.parameters()), retain_graph=True)
        if self.latent_gan_loss_scale > 0.:
            latent_d_loss.backward(inputs=list(self.model.latent_discriminator.parameters()), retain_graph=True)
        generator_loss.backward(inputs=list(self.model.vae.parameters()))
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.ema is not None:
            self.ema.update()
        del losses["vae_loss"]
        return losses
    
    def eval_step(self,
                  data_batch: tuple[DataBatchFloat, LabelBatchFloat]) -> tuple[ScalarFloat, dict[str, ScalarFloat]]:
        """GAN eval step.
        
        I decided to use the discriminator loss as the "main" loss just because the interface expects something here.
        Do no use EarlyStopping or ReduceLROnPlateau with GANs! Then this doesn't matter.
        """
        with torch.inference_mode():
            losses = self.core_step(data_batch)
        discriminator_loss = losses["data_discriminator_loss"]
        del losses["vae_loss"]
        return discriminator_loss, losses
    
    def core_step(self, 
                  data_batch: tuple[DataBatchFloat, LabelBatchFloat]) -> dict[str, ScalarFloat]:
        """Core step computing reconstruction loss and KL-Divergence."""
        input_batch, _ = data_batch
        input_batch = input_batch.to(self.device)
        reconstruction_batch, reconstruction_loss, kl_loss, latent_gan_losses = self.core_step_vae(input_batch)
        vae_loss = reconstruction_loss + self.beta * kl_loss
        data_gan_losses = self.core_step_data_gan(input_batch, reconstruction_batch)
        total_loss = (vae_loss
                      + self.data_gan_loss_scale * data_gan_losses["generator_loss"]
                      + self.latent_gan_loss_scale * latent_gan_losses["generator_loss"])

        loss_dict = {"data_discriminator_loss": data_gan_losses["discriminator_loss"], "vae_loss": total_loss,
                     "kl_loss": kl_loss, "reconstruction_loss": reconstruction_loss,
                     "latent_discriminator_loss": latent_gan_losses["discriminator_loss"]}
        if self.feature_matching:
            loss_dict.update({"data_generator_loss_classification": data_gan_losses["generator_loss_classification"],
                              "data_feature_matching_loss": data_gan_losses["feature_matching_loss"]})
            loss_dict.update({"latent_feature_matching_loss": latent_gan_losses["feature_matching_loss"],
                              "latent_generator_loss_classification": latent_gan_losses["generator_loss_classification"]})
        return loss_dict
    
    def core_step_vae(self,
                      input_batch: DataBatchFloat)\
                        -> tuple[DataBatchFloat, ScalarFloat, ScalarFloat, dict[str, ScalarFloat]]:
        """Unfortunately not really worth inheriting this from the VAETrainer."""
        means, log_variances = self.model.vae.encoding_parameters(input_batch)
        samples = self.model.vae.sampler(means, log_variances, self.samples_per_q)
        reconstruction_batch = self.model.vae.decoder(samples)

        reconstruction_loss = self.reconstruction_loss_fn(reconstruction_batch, input_batch)
        kl_loss = self.kl_loss_fn(means, log_variances)
        latent_gan_losses = self.core_step_latent_gan(samples)
        return reconstruction_batch, reconstruction_loss, kl_loss, latent_gan_losses
    
    def core_step_latent_gan(self,
                             posterior_samples: DataBatchFloat) -> dict[str, ScalarFloat]:
        if self.latent_gan_loss_scale > 0.:
            prior_samples = torch.randn_like(posterior_samples)
            return self.core_step_generic(prior_samples, posterior_samples, self.model.latent_discriminator)
        else:
            return {"discriminator_loss": torch.tensor(0.), "generator_loss": torch.tensor(0.),
                    "generator_loss_classification": torch.tensor(0.), "feature_matching_loss": torch.tensor(0.)}
    
    def core_step_data_gan(self,
                           input_batch: DataBatchFloat,
                           reconstruction_batch: DataBatchFloat) -> dict[str, ScalarFloat]:
        if self.data_gan_loss_scale > 0.:
            generated_batch = self.model.vae.map_likelihood(reconstruction_batch)
            return self.core_step_generic(input_batch, generated_batch, self.model.data_discriminator)
        else:
            return {"discriminator_loss": torch.tensor(0.), "generator_loss": torch.tensor(0.),
                    "generator_loss_classification": torch.tensor(0.), "feature_matching_loss": torch.tensor(0.)}
