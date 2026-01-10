from torch import nn

from ...gan import Discriminator, GAN
from ...vae.legacy import VAEGauss


class VEGAN(GAN):
    def __init__(self,
                 vae: VAEGauss,
                 data_discriminator: Discriminator,
                 latent_discriminator: Discriminator):
        """Container module for GAN training.
        
        Do not try to call this as a module.
        It's just a place to bundle generator and discriminator parameters.
        """
        # cursed... but I don't want to call GAN.__init__.
        # the inheritance from GAN is only due to the mess of generic types I have made.
        nn.Module.__init__(self)
        self.vae = vae
        self.data_discriminator = data_discriminator
        self.latent_discriminator = latent_discriminator
