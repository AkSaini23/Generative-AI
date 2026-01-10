"""This module provides code for the classic NICE and Glow models.

Both are instances of normalizing flows. We have skipped RealNVP, which could be considered an intermediate step
between the two. However, Glow made some simplifications to certain aspects of RealNVP, so it is actually simpler in
some aspects, while also performing better.
"""
from .fun import corrupt_likelihoods, interpolate_partial
from .glow import GlowFlow
from .nice import NiceFlow
from .trainer import FlowTrainer
