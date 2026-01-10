"""This module contains functionalities for training Generative Adversarial Networks.

REFERENCES
Original paper: https://proceedings.neurips.cc/paper_files/paper/2014/file/f033ed80deb0234979a61f95710dbe25-Paper.pdf
Feature matching & other tricks: https://proceedings.neurips.cc/paper_files/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf
"""
from .fun import optimization_inference
from .model import Discriminator, GAN, Generator, SDLayer
from .trainer import GANTrainer
