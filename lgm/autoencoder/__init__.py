"""This module provides functionalities for standard autoencoders.

These are not generative models, but serve as a good warmup. We will also be able to reuse parts of the code for some
actual generative models, such as variational autoencoders and their variants.
"""
from .likelihoods import loss_likelihood, map_likelihood
from .model import Autoencoder
from .trainer import AETrainer
