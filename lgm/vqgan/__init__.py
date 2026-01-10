"""This module implements a vector-quantized autoencder.

The codebook is learned through exponential moving average and can be optionally initialized with k-means.
We use straight-through estimation for learning, i.e. no Gumbel-Softmax tricks.

REFERENCES
VQVAE: https://arxiv.org/abs/1711.00937
VQGAN: https://arxiv.org/abs/2012.09841
L1 loss + PatchGAN combo: https://arxiv.org/abs/1611.07004
Residual VQ: https://ieeexplore.ieee.org/document/9625818
"""
from .model import VQVAE, VectorQuantizer, VQGAN
from .residual import ResidualVQ
from .trainer import VQGANTrainer
