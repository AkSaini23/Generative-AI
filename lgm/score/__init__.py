"""This module implements classic score-based models as introduced by Yang Song et al.

This is the discrete-time variant only. Continuous time variants (SDEs, ODEs) can be found in a different module.

References:
Original paper: https://arxiv.org/pdf/1907.05600
Improved version: https://arxiv.org/pdf/2006.09011
"""
from .fun import generate_omega
from .model import ScoreModel
from .trainer import ScoreTrainer
