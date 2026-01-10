"""In this module, we create and train Latent AutoRegressive Priors.

Given a pretrained VQVAE, we learn a model that generates appropriate latent code "images" that can the be decoded back
to RGB image space.
"""
from .model import ARTransformer, LARP, PositionalEncoding
from .trainer import LARPTrainer
