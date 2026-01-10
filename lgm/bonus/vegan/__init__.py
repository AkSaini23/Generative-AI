"""This module implements various VAE/GAN hybrids.

You have the standard likelihood-based reconstruction loss and KL-divergence scaled by beta. But then we have
discriminators both for the data as well as the latent space. These components can be turned on or off. 

For example, you can implement a GAN-based Wasserstein autoeoncder by only using the latent space discriminator and
setting beta (KLD scale) to 0. Or use the more general InfoVAE (https://arxiv.org/abs/1706.02262) which combines the
latent GAN and Kl divergence. Or you can augment all these with a discriminator in the data space, which tends to
result in shaper reconstructions.

A note on the code itself: This is a horrible monstrosity absuing diamond-shaped multiple inheritance. Sorry.
"""
from .model import Discriminator, VAEGauss, VEGAN
from .trainer import VEGANTrainer
