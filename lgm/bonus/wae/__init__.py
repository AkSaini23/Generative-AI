"""This implements a Wasserstein autoencoder: https://arxiv.org/abs/1711.01558

To be precise, it only implements the MMD-based version. For the GAN-based version, check the bonus.vegan module.
This honestly doesn't work very well.

Note that MMD scales very badly with batch size, so you will likely need very large batches for any degree of success.
"""
from .trainer import WAE, WAETrainer
