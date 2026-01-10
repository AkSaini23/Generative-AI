"""This module contains functionalities for binary Restricted Boltzmann Machines.

This is a rather "old-school" model that can't really compete with modern generative models, but it can stil be
instructive. We use MCMC methods for training. Other training algorithms are possible, but haven't been implemented.
"""
from .model import RBM
from .trainer import RBMTrainer
