"""This module contains functionalities for variational autoencoders.

The main module showcases a *conditional* VAE that takes (for example) classes as a guiding input. Unconditional
training is also supported. The legacy module contains the old, purely unconditional VAE code. This can be easier to
understand, and the code is reused in other places.

REFERENCES
Original paper: https://arxiv.org/abs/1312.6114
beta-VAE: https://openreview.net/pdf?id=Sy2fzU9gl
sigma-VAE: https://arxiv.org/abs/2006.13202
Continuous Bernoulli (bad :'<): https://arxiv.org/abs/1907.06845
"""
from .legacy.trainer import kl_loss_gauss
from .model import ConditionalVAEGauss, ConditionalNet, GaussSampler
from .trainer import ConditionalVAETrainer
