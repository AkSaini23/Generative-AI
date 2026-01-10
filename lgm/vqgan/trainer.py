from collections.abc import Callable

import torch

from .model import VQGAN
from ..autoencoder import AETrainer
from ..common import sum_except
from ..gan import GANTrainer
from ..types import DataBatchFloat, LabelBatchFloat, ScalarFloat


class VQGANTrainer(AETrainer[VQGAN], GANTrainer[VQGAN]):
    def __init__(self,
                 reconstruction_loss_fn:  Callable[[DataBatchFloat, DataBatchFloat], ScalarFloat] | None = None,
                 beta: float = 1.,
                 gan_loss_scale: float = 1.,
                 **kwargs):
        """Trainer for a basic VQVAE with auxilliary GAN loss for better reconstructions.
        
        Parameters:
            beta: Multiplier for the commitment loss.
            gan_loss_scale: Multiplier for the GAN loss.
            Other parameters: Like in lgm.autoencoder.AETrainer.
        """
        GANTrainer.__init__(self, **kwargs)
        self.beta = beta
        # unfortunately we cannot use AETrainer.__init__ due to multiple inheritance shenanigangs
        if reconstruction_loss_fn is None:
            reconstruction_loss_fn = self.model.vqvae.loss_likelihood()
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.autoencoder = self.model.vqvae
        self.gan_loss_scale = gan_loss_scale

    def train_step(self,
                   data_batch: tuple[DataBatchFloat, LabelBatchFloat]) -> dict[str, ScalarFloat]:
        """VQGAN training step.

        We have the VQVAE training step, consisting of reconstruction loss, commitment loss and codebook update. In
        addition, we train a GAN discrminator and backpropagate a GAN generator loss into the VQVAE, as well.
        """
        losses = self.core_step(data_batch)
        discriminator_loss = losses["discriminator_loss"]
        generator_loss = losses["vqvae_loss"]
        # retain graph needed so we can backward again for the generator
        if self.gan_loss_scale > 0.:
            discriminator_loss.backward(inputs=list(self.model.discriminator.parameters()), retain_graph=True)
        generator_loss.backward(inputs=list(self.model.vqvae.parameters()))
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.ema is not None:
            self.ema.update()
        del losses["vqvae_loss"]
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
        del losses["vqvae_loss"]
        return discriminator_loss, losses
    
    def core_step(self, 
                  data_batch: tuple[DataBatchFloat, LabelBatchFloat]) -> dict[str, ScalarFloat]:
        """Core step computing all necessary losses.
        
        We always compute the VQVAE losses (reconstruction + commitment). GAN loss is also always called, but will be a
        dummy in case the gan_loss_scale is 0.
        """
        input_batch, _ = data_batch
        input_batch = input_batch.to(self.device)
        reconstruction_batch, reconstruction_loss, commitment_loss = self.core_step_vqvae(input_batch)
        vae_loss = reconstruction_loss + self.beta * commitment_loss
        gan_losses = self.core_step_gan(input_batch, reconstruction_batch)
        total_loss = vae_loss + self.gan_loss_scale * gan_losses["generator_loss"]

        loss_dict = {"discriminator_loss": gan_losses["discriminator_loss"], "vqvae_loss": total_loss,
                     "commitment_loss": commitment_loss, "reconstruction_loss": reconstruction_loss}
        if self.feature_matching:
            loss_dict.update({"generator_loss_classification": gan_losses["generator_loss_classification"],
                              "feature_matching_loss": gan_losses["feature_matching_loss"]})
        return loss_dict
    
    def core_step_vqvae(self, 
                        input_batch: DataBatchFloat) -> tuple[DataBatchFloat, ScalarFloat, ScalarFloat]:
        """Core step for the VQVAE part.
        
        Returns the output, not just the losses, since the former then goes into the GAN in the next step.
        """
        encodings = self.model.vqvae.encoder(input_batch)
        quantized = self.model.vqvae.quantizer(encodings)
        output_batch = self.model.vqvae.decoder(quantized)

        reconstruction_loss = self.reconstruction_loss_fn(output_batch, input_batch)
        commitment_loss = self.commitment_loss(encodings, quantized)
        return output_batch, reconstruction_loss, commitment_loss
    
    def core_step_gan(self,
                      input_batch: DataBatchFloat,
                      reconstruction_batch: DataBatchFloat) -> dict[str, ScalarFloat]:
        """Core step for the GAN part."""
        if self.gan_loss_scale > 0.:
            generated_batch = self.model.vqvae.map_likelihood(reconstruction_batch)
            return self.core_step_generic(input_batch, generated_batch, self.model.discriminator)
        else:
            return {"discriminator_loss": torch.tensor(0.), "generator_loss": torch.tensor(0.),
                    "generator_loss_classification": torch.tensor(0.), "feature_matching_loss": torch.tensor(0.)}
    
    def commitment_loss(self,
                        encodings: DataBatchFloat,
                        quantized: DataBatchFloat) -> ScalarFloat:
        """Prevents encoder outputs from "escaping" arbitrarily far.
        
        This is especially important to alleviate issues caused by inaccurate gradients due to quantization.
        """
        return sum_except((encodings - quantized.detach())**2).mean()
