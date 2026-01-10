"""This module contains functionalities that are reused over and over in different generative models.

This mostly concerns the "Trainer" class and various training functionalities (such as early stopping or checkpointing)
on one hand, and a bunch of general utility functions on the other.
"""
from.fun import interpolate, make_animation, per_dim_latent_walk
from .training import Checkpointer, EarlyStopping, EMA, TrainerBase, cosine_decay_warmup
from .utils import (batched_multiply, channels_first_to_last, channels_last_to_first, count_parameters, interleave,
                    plot_learning_curves, spatial_broadcast_index, squared_distances, sum_except)
